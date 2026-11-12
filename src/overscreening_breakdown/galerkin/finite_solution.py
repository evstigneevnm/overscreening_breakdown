"""Independent mixed fields on a finite physical interval.

The affine finite basis has N-1 homogeneous u coefficients and N+1 w
coefficients. Its interval length is geometry, not a rational-map parameter.
All evaluations and physical norms retain the solution's numerical owner.
"""

from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np

from overscreening_breakdown.galerkin.solution import (
    _check_norm_plan, _finite, _sum,
    physical_h1_error, physical_h1_norm, physical_l2_error, physical_l2_norm,
)
from overscreening_breakdown.problems.flux import BoundFluxProblem


@dataclass(frozen=True, init=False, eq=False)
class FiniteGalerkinSolution:
    """Native finite-interval fields with a two-end linear Dirichlet lift."""

    basis: Any
    physical: BoundFluxProblem = field(repr=False)
    plan: Any = field(default=None, repr=False)
    method: str = "newton"
    converged: bool = False
    nonlinear_converged: bool = False
    quadrature_converged: bool = False
    newton_result: Any = field(default=None, repr=False)
    initialization_result: Any = field(default=None, repr=False)
    quadrature_history: Tuple[Any, ...] = field(default=(), repr=False)
    projection_info: Tuple[Any, ...] = field(default=(), repr=False)
    _entries: Tuple[Any, ...] = field(default=(), repr=False)

    solver_type = "galerkin"
    scipy_result = None

    def __init__(self, basis, physical, coefficients, *, plan=None, method="newton", converged=False,
                 nonlinear_converged=False, quadrature_converged=False,
                 newton_result=None, quadrature_history=(), projection_info=(),
                 initialization_result=None):
        from overscreening_breakdown.galerkin.finite_basis import FiniteGalerkinBasis
        if not isinstance(basis, FiniteGalerkinBasis):
            raise TypeError("basis must be a FiniteGalerkinBasis")
        basis.check_precision()
        if not isinstance(physical, BoundFluxProblem):
            raise TypeError("physical must be a bound physical problem")
        physical.check_precision()
        if physical.arithmetic is not basis.arithmetic:
            raise ValueError("solution and physical problem must share one arithmetic owner")
        if physical.domain != basis.mapping.domain:
            raise ValueError("finite fields and physical problem must share the same domain")
        left, right = physical.boundary_conditions
        if (physical.delta <= 0 or left[0] is None or left[3] is None
                or right[0] is None or right[1] is None
                or any(left[index] is not None for index in (1, 2, 4))
                or any(right[index] is not None for index in (2, 3, 4))):
            raise ValueError("finite mixed fields require delta > 0, left u/u''' and right u/u' data")
        if converged and not (nonlinear_converged and quadrature_converged):
            raise ValueError("Galerkin convergence requires nonlinear and quadrature convergence")
        if method == "projection" and any((converged, nonlinear_converged, quadrature_converged)):
            raise ValueError("projected fields must be solved before claiming convergence")
        a, b = basis.split_coefficients(coefficients)
        for name, value in (("basis", basis), ("physical", physical), ("plan", plan), ("method", method),
                            ("converged", bool(converged)), ("nonlinear_converged", bool(nonlinear_converged)),
                            ("quadrature_converged", bool(quadrature_converged)),
                            ("newton_result", newton_result), ("initialization_result", initialization_result),
                            ("quadrature_history", tuple(quadrature_history)),
                            ("projection_info", tuple(tuple(item) for item in projection_info)),
                            ("_entries", a+b)):
            object.__setattr__(self, name, value)
        if plan is not None:
            self._check_field_plan(plan)

    @property
    def arithmetic(self):
        return self.basis.arithmetic

    @property
    def mapping(self):
        return self.basis.mapping

    @property
    def length(self):
        return self.mapping.length

    @property
    def N(self):
        return self.basis.N

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
        y = self.mapping.to_reference(self.mapping.physical_coordinate(x))
        a, b = self._entries[:self.basis.n_u], self._entries[self.basis.n_u:]
        lift = self.basis.lift(self.physical.boundary_conditions[0][0],
                               self.physical.boundary_conditions[1][0])
        u = _sum(self.arithmetic, (lift.value(y),)+tuple(c*v for c, v in zip(a, self.basis.v(y))))
        w = _sum(self.arithmetic, (c*v for c, v in zip(b, self.basis.q(y))))
        u_dx = _sum(self.arithmetic, (lift.dx(y),)+tuple(c*v for c, v in zip(a, self.basis.v_dx(y))))
        w_dx = _sum(self.arithmetic, (c*v for c, v in zip(b, self.basis.q_dx(y))))
        return u, w, u_dx, w_dx

    def _evaluate(self, x, component):
        self.check_precision()
        inputs = np.asarray(x, dtype=object)
        output = np.empty(inputs.shape, dtype=object if self.arithmetic.use_mpmath else np.float64)
        for index in np.ndindex(inputs.shape):
            coordinate = self.mapping.physical_coordinate(inputs[index])
            fields = self._scalar_fields(coordinate)
            value = (self.physical.epsilon(coordinate)*(fields[2]+self.physical.delta**2*fields[3])
                     if component == "flux" else fields[component])
            output[index] = _finite(self.arithmetic, value, "finite physical field evaluation")
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
        """Evaluate epsilon*(u' + delta²*w') from the independent mixed fields."""
        return self._evaluate(x, "flux")

    def _check_field_plan(self, plan):
        self.check_precision()
        _check_norm_plan(plan)
        if (plan.arithmetic is not self.arithmetic
                or getattr(plan.mapping, "mapping_type", None) != "affine_interval"
                or plan.mapping.domain != self.domain):
            raise ValueError("finite field diagnostics require the same arithmetic owner and finite domain")
        boundaries = {endpoint for panel in plan.panels for endpoint in panel}
        if any(self.mapping.to_reference(point) not in boundaries
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
        from overscreening_breakdown.solution_io import save_solution
        return save_solution(self, path)
