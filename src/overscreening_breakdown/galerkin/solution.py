"""Independent mixed fields and diagnostics in the physical measure dx.

These norms use explicit positive quadrature plans and retain their numerical
integration error.  They are distinct from the legacy mapped-coordinate
``L2_error`` and from algebraic nonlinear residual norms.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Tuple

import numpy as np

from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.problems.flux import BoundFluxProblem
from overscreening_breakdown.quadrature import QuadraturePlan


def _finite(arithmetic, value, name):
    value = arithmetic.scalar(value)
    if not arithmetic.isfinite(value):
        raise OverflowError("{} must be finite".format(name))
    return value


def _sum(arithmetic, values):
    values = tuple(_finite(arithmetic, value, "field evaluation term") for value in values)
    return _finite(arithmetic, arithmetic.context.fsum(values) if arithmetic.use_mpmath
                   else math.fsum(values), "field evaluation sum")


def _distance_polynomial(arithmetic, coefficients):
    """Convert sum(c_k*T_k(1-r)) using exact integer recurrence factors."""
    polynomials = [(1,), (1, -1)]
    for _ in range(2, len(coefficients)):
        current, previous = polynomials[-1], polynomials[-2]
        row = []
        for degree in range(len(current)+1):
            row.append((2*current[degree] if degree < len(current) else 0)
                       -(2*current[degree-1] if degree else 0)
                       -(previous[degree] if degree < len(previous) else 0))
        polynomials.append(tuple(row))
    return tuple(_sum(arithmetic, (coefficient*polynomial[degree]
                                  for coefficient, polynomial in zip(coefficients, polynomials)
                                  if degree < len(polynomial)))
                 for degree in range(len(coefficients)))


def _horner_with_derivative(arithmetic, coefficients, coordinate):
    value = derivative = arithmetic.scalar(0)
    for coefficient in reversed(coefficients):
        derivative = derivative*coordinate+value
        value = value*coordinate+coefficient
    return value, derivative


def _check_norm_plan(plan):
    if not isinstance(plan, QuadraturePlan):
        raise TypeError("physical norms require a QuadraturePlan")
    plan.check_precision()
    count = len(plan.physical_nodes)
    if not count or count != len(plan.physical_weights):
        raise ValueError("physical quadrature nodes and weights must have equal nonzero lengths")
    arithmetic = plan.arithmetic
    previous = arithmetic.scalar(0)
    for x, weight in zip(plan.physical_nodes, plan.physical_weights):
        x = _finite(arithmetic, x, "physical quadrature coordinate")
        weight = _finite(arithmetic, weight, "physical quadrature weight")
        if x <= previous or weight <= 0:
            raise ValueError("physical norms require increasing positive nodes and positive weights")
        previous = x


def _physical_squared_norm(functions, plan):
    _check_norm_plan(plan)
    if any(not callable(function) for function in functions):
        raise TypeError("norm callbacks must be callable")
    arithmetic = plan.arithmetic
    terms = []
    for x, weight in zip(plan.physical_nodes, plan.physical_weights):
        for function in functions:
            value = _finite(arithmetic, function(x), "physical norm callback")
            terms.append(_finite(arithmetic, weight*value*value, "physical squared norm term"))
    return _sum(arithmetic, terms)


def physical_l2_norm(function, plan):
    """Approximate ``sqrt(integral |function(x)|**2 dx)`` on the half-line."""
    squared = _physical_squared_norm((function,), plan)
    return plan.arithmetic.sqrt(squared)


def physical_h1_norm(function, derivative, plan):
    """Approximate the full physical H1 norm, including value and derivative."""
    squared = _physical_squared_norm((function, derivative), plan)
    return plan.arithmetic.sqrt(squared)


def physical_l2_error(function, reference, plan):
    """Approximate physical L2 error with scalar reference evaluations."""
    if not callable(function) or not callable(reference):
        raise TypeError("error callbacks must be callable")
    _check_norm_plan(plan)
    arithmetic = plan.arithmetic

    def difference(x):
        return (_finite(arithmetic, function(x), "physical field")
                -_finite(arithmetic, reference(x), "physical reference"))

    return physical_l2_norm(difference, plan)


def physical_h1_error(function, derivative, reference, reference_dx, plan):
    """Approximate physical H1 error for explicitly supplied derivatives."""
    if any(not callable(callback) for callback in (function, derivative, reference, reference_dx)):
        raise TypeError("error callbacks must be callable")
    _check_norm_plan(plan)
    arithmetic = plan.arithmetic

    def difference(x):
        return (_finite(arithmetic, function(x), "physical field")
                -_finite(arithmetic, reference(x), "physical reference"))

    def derivative_difference(x):
        return (_finite(arithmetic, derivative(x), "physical derivative")
                -_finite(arithmetic, reference_dx(x), "physical reference derivative"))

    return physical_h1_norm(difference, derivative_difference, plan)


@dataclass(frozen=True, init=False, eq=False)
class GalerkinSolution:
    """Immutable coefficients for two independently approximated mixed fields.

    Scalar and zero-dimensional inputs return native arithmetic scalars.
    Arrays retain their shape and use float64 or object dtype.  Evaluation
    never changes global mpmath precision.  Coefficient properties return
    copies, while the stored coefficient tuple cannot be mutated.

    ``plan`` and convergence information describe how this result was obtained;
    constructing a field object alone does not certify a nonlinear solution.
    """

    basis: GalerkinBasis
    physical: BoundFluxProblem = field(repr=False)
    plan: Any = field(default=None, repr=False)
    method: str = "newton"
    converged: bool = False
    nonlinear_converged: bool = False
    quadrature_converged: bool = False
    newton_result: Any = field(default=None, repr=False)
    quadrature_history: Tuple[Any, ...] = field(default=(), repr=False)
    projection_info: Tuple[Any, ...] = field(default=(), repr=False)
    _entries: Tuple[Any, ...] = field(default=(), repr=False)

    solver_type = "galerkin"

    def __init__(self, basis, physical, coefficients, *, plan=None, method="newton", converged=False,
                 nonlinear_converged=False, quadrature_converged=False,
                 newton_result=None, quadrature_history=(), projection_info=()):
        if not isinstance(basis, GalerkinBasis):
            raise TypeError("basis must be a GalerkinBasis")
        basis.check_precision()
        if not isinstance(physical, BoundFluxProblem):
            raise TypeError("physical must be a bound physical problem")
        physical.check_precision()
        if physical.arithmetic is not basis.arithmetic:
            raise ValueError("solution and physical problem must share one arithmetic owner")
        if physical.domain != (basis.arithmetic.scalar(0), basis.arithmetic.inf):
            raise ValueError("mixed Galerkin fields require the physical half-line")
        if physical.delta <= 0 or physical.boundary_conditions[0][0] is None:
            raise ValueError("mixed Galerkin fields require delta > 0 and a prescribed wall value")
        if converged and not (nonlinear_converged and quadrature_converged):
            raise ValueError("Galerkin convergence requires nonlinear and quadrature convergence")
        if method == "projection" and any((converged, nonlinear_converged, quadrature_converged)):
            raise ValueError("projected fields must be solved before claiming convergence")
        a, b = basis.split_coefficients(coefficients)
        for name, value in (("basis", basis), ("physical", physical), ("plan", plan),
                            ("method", method),
                            ("converged", bool(converged)),
                            ("nonlinear_converged", bool(nonlinear_converged)),
                            ("quadrature_converged", bool(quadrature_converged)),
                            ("newton_result", newton_result),
                            ("quadrature_history", tuple(quadrature_history)),
                            ("projection_info", tuple(tuple(item) for item in projection_info)),
                            ("_entries", a+b)):
            object.__setattr__(self, name, value)
        u_polynomial = _distance_polynomial(self.arithmetic, a)
        zero = self.arithmetic.scalar(0)
        wall = physical.boundary_conditions[0][0]
        u_expanded = [zero]
        for degree in range(len(u_polynomial)+1):
            u_expanded.append(_sum(self.arithmetic, (
                wall/2 if degree == 0 else zero,
                2*u_polynomial[degree] if degree < len(u_polynomial) else zero,
                -u_polynomial[degree-1] if degree else zero,
            )))
        object.__setattr__(self, "_u_polynomial", tuple(u_expanded))
        object.__setattr__(self, "_w_polynomial", (zero,)+_distance_polynomial(self.arithmetic, b))
        if plan is not None:
            self._check_field_plan(plan)

    @property
    def arithmetic(self):
        return self.basis.arithmetic

    @property
    def N(self):
        return self.basis.N

    @property
    def L(self):
        return self.basis.mapping.L

    @property
    def unknown_count(self):
        return self.basis.unknown_count

    @property
    def domain(self):
        return self.physical.domain

    def check_precision(self):
        self.basis.check_precision()
        self.physical.check_precision()
        if self.plan is not None:
            self.plan.check_precision()

    @property
    def coefficients(self):
        self.check_precision()
        return self.arithmetic.vector(self._entries)

    @property
    def u_coefficients(self):
        self.check_precision()
        return self.arithmetic.vector(self._entries[:self.basis.n_u])

    @property
    def w_coefficients(self):
        self.check_precision()
        return self.arithmetic.vector(self._entries[self.basis.n_u:])

    def _scalar_fields(self, x):
        self.check_precision()
        arithmetic = self.arithmetic
        x = self.basis.mapping.physical_coordinate(x)
        zero = arithmetic.scalar(0)
        if x == arithmetic.inf:
            return zero, zero, zero, zero
        if x >= self.L:
            ratio = self.L/x
            distance = 2*(ratio/(1+ratio))
        else:
            distance = arithmetic.scalar(2)/(1+x/self.L)
        if distance*self.N*self.N < arithmetic.scalar("0.25"):
            u, u_derivative = _horner_with_derivative(arithmetic, self._u_polynomial, distance)
            w, w_derivative = _horner_with_derivative(arithmetic, self._w_polynomial, distance)
            chain = -(distance*(distance/2))/self.L
            return u, w, u_derivative*chain, w_derivative*chain
        y = arithmetic.scalar(1)-distance
        _, polynomials, derivatives = self.basis._polynomials(y)
        a, b = self._entries[:self.basis.n_u], self._entries[self.basis.n_u:]
        wall = self.physical.boundary_conditions[0][0]
        v_factor = distance*(2-distance)
        u = _sum(arithmetic, (wall*(distance/2),) + tuple(
            coefficient*v_factor*value for coefficient, value in zip(a, polynomials)))
        w = _sum(arithmetic, tuple(coefficient*distance*value
                                  for coefficient, value in zip(b, polynomials)))
        chain = (distance*(distance/2))/self.L
        u_dx = _sum(arithmetic, (-wall/2*chain,) + tuple(
            coefficient*(-2*y*value+v_factor*derivative)*chain
            for coefficient, value, derivative in zip(a, polynomials, derivatives)))
        w_dx = _sum(arithmetic, tuple(
            coefficient*(-value+distance*derivative)*chain
            for coefficient, value, derivative in zip(b, polynomials, derivatives)))
        return u, w, u_dx, w_dx

    def _evaluate(self, x, component):
        self.check_precision()
        inputs = np.asarray(x, dtype=object)
        output = np.empty(inputs.shape, dtype=object if self.arithmetic.use_mpmath else np.float64)
        for index in np.ndindex(inputs.shape):
            coordinate = self.basis.mapping.physical_coordinate(inputs[index])
            values = self._scalar_fields(coordinate)
            if component == "flux":
                value = (self.arithmetic.scalar(0) if coordinate == self.arithmetic.inf else
                         self.physical.epsilon(coordinate)*(
                             values[2]+self.physical.delta**2*values[3]))
            else:
                value = values[component]
            output[index] = _finite(self.arithmetic, value, "physical field evaluation")
        return output[()] if inputs.ndim == 0 else output

    def u(self, x):
        return self._evaluate(x, 0)

    def w(self, x):
        return self._evaluate(x, 1)

    def u_dx(self, x):
        return self._evaluate(x, 2)

    def w_dx(self, x):
        return self._evaluate(x, 3)

    def flux(self, x):
        """Evaluate epsilon*(u' + delta²*w') using the independent w field."""
        return self._evaluate(x, "flux")

    def _check_field_plan(self, plan):
        self.check_precision()
        _check_norm_plan(plan)
        if plan.arithmetic is not self.arithmetic or plan.mapping.L != self.L:
            raise ValueError("field diagnostics require the same arithmetic owner and mapping length")
        boundaries = {endpoint for panel in plan.panels for endpoint in panel}
        if any(self.basis.mapping.to_reference(point) not in boundaries
               for point in self.physical.epsilon.interfaces):
            raise ValueError("field diagnostics must split every epsilon interface")

    def _field_callbacks(self, field, require_derivative=False):
        if field == "u":
            return self.u, self.u_dx
        if field == "w":
            return self.w, self.w_dx
        if field == "flux" and not require_derivative:
            return self.flux, None
        raise ValueError("field must be u or w; flux is available for L2 diagnostics only")

    def l2_norm(self, field="u", *, plan):
        self._check_field_plan(plan)
        function, _ = self._field_callbacks(field)
        return physical_l2_norm(function, plan)

    def h1_norm(self, field="u", *, plan):
        self._check_field_plan(plan)
        function, derivative = self._field_callbacks(field, require_derivative=True)
        return physical_h1_norm(function, derivative, plan)

    def l2_error(self, reference, field="u", *, plan):
        self._check_field_plan(plan)
        function, _ = self._field_callbacks(field)
        return physical_l2_error(function, reference, plan)

    def h1_error(self, reference, reference_dx, field="u", *, plan):
        self._check_field_plan(plan)
        function, derivative = self._field_callbacks(field, require_derivative=True)
        return physical_h1_error(function, derivative, reference, reference_dx, plan)

    def save(self, path):
        """Save typed coefficients and metadata through the versioned format."""
        from overscreening_breakdown.solution_io import save_solution
        return save_solution(self, path)
