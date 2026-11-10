"""Fixed-quadrature mixed assembly for the physical half-line flux problem.

The independent fields are ``u = lift + V*a`` and ``w = Q*b = -u''`` in
the continuous problem. The second weak equation is divided by the declared
right wall trace ``epsilon_b``. All integrals use the physical measure dx.
Only the source load and its analytic derivative depend on the trial state.

This module supplies a discrete system to the shared Newton driver; solver
dispatch, physical error norms and mixed solution persistence are separate.
"""

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any

import numpy as np

from galerkin import GalerkinBasis
from nonlinear_solver import LinearSolveError, NonlinearSystem
from physical_problem import BoundFluxProblem
from quadrature import QuadraturePlan


@dataclass(frozen=True)
class AssemblyRefinementCheck:
    """Fixed-state quadrature diagnostic in algebraic coefficient norms.

    Residuals use the Euclidean norm and Jacobians the Frobenius norm.
    Neither is a physical field error or a rigorous quadrature error bound.
    Both comparisons must pass; no nonlinear solve occurs in this check.
    """

    residual_difference: Any
    residual_target: Any
    jacobian_difference: Any
    jacobian_target: Any
    converged: bool
    refined_assembly: Any = field(repr=False, compare=False)


@dataclass(frozen=True, init=False, eq=False)
class GalerkinAssembly:
    """Assemble a physical problem on one compatible basis and fixed plan.

    ``problem`` is an unbound description with ``bind(arithmetic)``, such as
    FluxProblem or overscreening_breakdown. The assembly captures a bound
    snapshot; already-bound problems are rejected because their parameters
    may have been rounded at a previous precision.

    Require delta>0, domain [0,infinity), wall data u(0) and u'''(0), and
    no other wall constraints. An explicit zero value at infinity is allowed.
    The common positive quadrature must have order>=N+1 per panel and split
    every declared epsilon interface. Extra panel splits are allowed.

    The coefficient order is (a,b), with N-1 and N entries respectively.
    The residual convention is DR*step=-R. Public matrices and vectors are
    fresh native-arithmetic copies of the cached constant data.
    """

    basis: GalerkinBasis
    plan: Any = field(repr=False)
    physical: BoundFluxProblem = field(repr=False)

    def __init__(self, problem, basis, plan):
        if not isinstance(basis, GalerkinBasis):
            raise TypeError("basis must be a GalerkinBasis")
        basis.check_precision()
        if isinstance(problem, BoundFluxProblem):
            raise TypeError("supply an unbound physical problem description")
        bind = getattr(problem, "bind", None)
        if not callable(bind):
            raise TypeError("problem must provide bind(arithmetic)")
        physical = bind(basis.arithmetic)
        if not isinstance(physical, BoundFluxProblem):
            raise TypeError("problem.bind must return a BoundFluxProblem")
        self._initialize(physical, basis, plan)

    @property
    def arithmetic(self):
        return self.basis.arithmetic

    @property
    def unknown_count(self):
        return self.basis.unknown_count

    def _finite(self, value, name, error_type=OverflowError):
        value = self.arithmetic.scalar(value)
        if not self.arithmetic.isfinite(value):
            raise error_type("{} must be finite in the selected arithmetic".format(name))
        return value

    def _sum(self, values):
        values = tuple(values)
        if any(not self.arithmetic.isfinite(value) for value in values):
            raise OverflowError("a mixed assembly term is not finite")
        value = (self.arithmetic.context.fsum(values) if self.arithmetic.use_mpmath
                 else math.fsum(values))
        return self._finite(value, "mixed assembly sum")

    def _initialize(self, physical, basis, plan):
        physical.check_precision()
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "physical", physical)
        if physical.arithmetic is not basis.arithmetic:
            raise ValueError("physical problem and basis must share one arithmetic owner")
        if not isinstance(plan, QuadraturePlan):
            raise TypeError("plan must be a QuadraturePlan from QuadratureFactory")
        object.__setattr__(self, "table", basis.tabulate(plan))
        self._validate_plan()
        if physical.domain != (self.arithmetic.scalar(0), self.arithmetic.inf):
            raise ValueError("mixed Galerkin assembly supports the half-line [0, infinity) only")
        if physical.delta <= 0:
            raise ValueError("mixed Galerkin assembly requires delta > 0")
        left, right = physical.boundary_conditions
        if (left[0] is None or left[3] is None
                or any(left[index] is not None for index in (1, 2, 4))):
            raise ValueError("mixed Galerkin wall data must prescribe u(0) and u'''(0) only")
        if right[0] not in (None, 0) or any(value is not None for value in right[1:]):
            raise ValueError("only an optional zero far-field value is supported")
        panel_boundaries = {endpoint for panel in plan.panels for endpoint in panel}
        for point in physical.epsilon.interfaces:
            if basis.mapping.to_reference(point) not in panel_boundaries:
                raise ValueError("quadrature must split every declared epsilon interface")
        object.__setattr__(self, "lift", basis.lift(left[0]))
        object.__setattr__(self, "epsilon_b", physical.epsilon.wall_value)
        delta_squared = self._finite(physical.delta**2, "delta squared", ValueError)
        if delta_squared <= 0:
            raise ValueError("delta squared must remain strictly positive in the selected arithmetic")
        object.__setattr__(self, "delta_squared", delta_squared)
        ratios = tuple(
            self._finite(physical.epsilon(x)/self.epsilon_b, "epsilon/epsilon_b", ValueError)
            for x in plan.physical_nodes
        )
        if any(value <= 0 for value in ratios):
            raise ValueError("epsilon/epsilon_b must remain strictly positive in the selected arithmetic")
        object.__setattr__(self, "_epsilon_ratios", ratios)
        object.__setattr__(self, "_lift_values", tuple(self.lift.value(y) for y in plan.nodes))
        object.__setattr__(self, "_lift_derivatives", tuple(self.lift.dx(y) for y in plan.nodes))
        self._assemble_constant_terms(left[3])

    def _validate_plan(self):
        # Plans are public dataclasses. Check their mathematical contract as
        # well as their owner so malformed/replaced sequences cannot make zip
        # silently drop terms or destroy positive common-rule assembly.
        plan = self.plan
        if (any(not isinstance(values, tuple) for values in
                (plan.nodes, plan.physical_nodes, plan.weights, plan.physical_weights, plan.panels))
                or any(not isinstance(panel, tuple) or len(panel) != 2 for panel in plan.panels)):
            raise ValueError("quadrature nodes, weights and panels must be immutable tuples")
        if (isinstance(plan.order, bool) or not isinstance(plan.order, Integral)
                or plan.order < self.basis.N+1):
            raise ValueError("mixed assembly requires quadrature order >= N+1 per panel")
        count = len(plan.nodes)
        if (not count or not plan.panels or count != plan.order*len(plan.panels)
                or any(len(values) != count for values in
                       (plan.physical_nodes, plan.weights, plan.physical_weights))):
            raise ValueError("quadrature node and weight lengths must match its panel order")
        one = self.arithmetic.scalar(1)
        previous_right = -one
        for left, right in plan.panels:
            if left != previous_right or not -one <= left < right <= one:
                raise ValueError("quadrature panels must partition [-1, 1]")
            previous_right = right
        if previous_right != one:
            raise ValueError("quadrature panels must partition [-1, 1]")
        interfaces = tuple((point, self.basis.mapping.to_reference(point))
                           for point in self.physical.epsilon.interfaces)
        previous_y, previous_x = -one, self.arithmetic.scalar(0)
        for index, (y, x, weight, physical_weight) in enumerate(zip(
                plan.nodes, plan.physical_nodes, plan.weights, plan.physical_weights)):
            left, right = plan.panels[index//plan.order]
            for value in (y, x, weight, physical_weight):
                self._finite(value, "quadrature data", ValueError)
            if (not left < y < right or not previous_y < y
                    or not previous_x < x or weight <= 0 or physical_weight <= 0):
                raise ValueError("quadrature requires increasing interior nodes and positive weights")
            if (x != self.basis.mapping.to_physical(y)
                    or physical_weight != weight*self.basis.mapping.jacobian(y)):
                raise ValueError("quadrature physical data must match its mapping and weights")
            for point, reference in interfaces:
                if not ((y < reference and x < point) or (y > reference and x > point)):
                    raise ValueError("quadrature nodes must remain inside physical interface panels")
            previous_y, previous_x = y, x

    def check_precision(self):
        """Reject stale basis, geometry and cached assembly after a bit change."""
        self.table.check_precision()
        self.physical.check_precision()

    @classmethod
    def _from_snapshot(cls, physical, basis, plan):
        """Internal reuse of a stamped bound snapshot without rebinding callbacks."""
        physical.check_precision()
        result = object.__new__(cls)
        result._initialize(physical, basis, plan)
        return result

    def _assemble_constant_terms(self, q_b):
        table, weights = self.table, self.plan.physical_weights
        m, n = self.basis.n_u, self.basis.n_w
        rows, lift_entries = [], []
        for i in range(m):
            rows.append(tuple(
                self._sum(weight*vx[i]*vx[j] for weight, vx in zip(weights, table.v_dx))
                for j in range(m)
            ) + tuple(
                -self._sum(weight*v[i]*q[j]
                           for weight, v, q in zip(weights, table.v, table.q))
                for j in range(n)
            ))
            lift_entries.append(self._sum(
                weight*derivative*vx[i] for weight, derivative, vx in
                zip(weights, self._lift_derivatives, table.v_dx)
            ))
        for i in range(n):
            rows.append(tuple(
                self._sum(weight*(epsilon-1)*qx[i]*vx[j]
                           for weight, epsilon, qx, vx in
                           zip(weights, self._epsilon_ratios, table.q_dx, table.v_dx))
                for j in range(m)
            ) + tuple(
                self._sum(weight*(self.delta_squared*epsilon*qx[i]*qx[j] + q[i]*q[j])
                           for weight, epsilon, qx, q in
                           zip(weights, self._epsilon_ratios, table.q_dx, table.q))
                for j in range(n)
            ))
            lift_entries.append(self._sum(
                weight*(epsilon-1)*derivative*qx[i]
                for weight, epsilon, derivative, qx in
                zip(weights, self._epsilon_ratios, self._lift_derivatives, table.q_dx)
            ))
        zero = self.arithmetic.scalar(0)
        wall_entries = (zero,)*m + tuple(
            self._finite(self.delta_squared*q_b*trace, "natural boundary load", ValueError)
            for trace in self.basis.q(-1)
        )
        object.__setattr__(self, "_linear_rows", tuple(rows))
        object.__setattr__(self, "_lift_entries", tuple(lift_entries))
        object.__setattr__(self, "_boundary_entries", wall_entries)
        object.__setattr__(self, "_affine_entries", tuple(
            self._finite(lift-wall, "affine load", ValueError)
            for lift, wall in zip(lift_entries, wall_entries)
        ))

    @property
    def linear_matrix(self):
        self.check_precision()
        return self.arithmetic.matrix(self._linear_rows)

    @property
    def lift_vector(self):
        self.check_precision()
        return self.arithmetic.vector(self._lift_entries)

    @property
    def boundary_vector(self):
        self.check_precision()
        return self.arithmetic.vector(self._boundary_entries)

    @property
    def affine_vector(self):
        self.check_precision()
        return self.arithmetic.vector(self._affine_entries)

    def _state(self, coefficients):
        self.check_precision()
        a, b = self.basis.split_coefficients(coefficients)
        return a, b, a+b

    def _linear_values(self, state):
        return tuple(self._sum((affine,) + tuple(
            entry*coefficient for entry, coefficient in zip(row, state)
        )) for row, affine in zip(self._linear_rows, self._affine_entries))

    def linear_residual(self, coefficients):
        """Evaluate constant blocks plus lift minus boundary, without f."""
        _, _, state = self._state(coefficients)
        return self.arithmetic.vector(self._linear_values(state))

    def _parameter(self, q):
        q = self._finite(q, "homotopy parameter", ValueError)
        if q < 0:
            raise ValueError("homotopy parameter must be nonnegative")
        return q

    def _source_values(self, a, q, derivative=False):
        callback = self.physical.physical_rhs_u if derivative else self.physical.physical_rhs
        values = []
        for x, lift, v in zip(self.plan.physical_nodes, self._lift_values, self.table.v):
            u = self._sum((lift,) + tuple(entry*coefficient for entry, coefficient in zip(v, a)))
            damping = self.arithmetic.exp(-q*x)
            values.append(self._finite(damping*callback(x, u)/self.epsilon_b,
                                       "normalized physical source"))
        return tuple(values)

    def residual(self, coefficients, q=0):
        """Return R(a,b;q), damping only physical f by exp(-q*x)."""
        a, _, state = self._state(coefficients)
        q = self._parameter(q)
        result = list(self._linear_values(state))
        source = self._source_values(a, q)
        for i in range(self.basis.n_w):
            load = self._sum(weight*value*test[i] for weight, value, test in
                             zip(self.plan.physical_weights, source, self.table.q))
            row = self.basis.n_u+i
            result[row] = self._finite(result[row]+load, "mixed residual")
        return self.arithmetic.vector(result)

    def jacobian(self, coefficients, q=0):
        """Return the analytic derivative DR of the residual, with its sign."""
        a, _, _ = self._state(coefficients)
        q = self._parameter(q)
        derivative = self._source_values(a, q, derivative=True)
        rows = [list(row) for row in self._linear_rows]
        for i in range(self.basis.n_w):
            for j in range(self.basis.n_u):
                load = self._sum(weight*value*test[i]*trial[j]
                                 for weight, value, test, trial in
                                 zip(self.plan.physical_weights, derivative,
                                     self.table.q, self.table.v))
                row = self.basis.n_u+i
                rows[row][j] = self._finite(rows[row][j]+load, "mixed Jacobian")
        return self.arithmetic.matrix(rows)

    def residual_norm(self, residual):
        """Algebraic Euclidean norm of a residual vector (not a field norm)."""
        self.check_precision()
        return self.arithmetic.norm(residual)

    def nonlinear_system(self, initial_guess=None):
        """Build shared Newton callbacks; default state is (u=lift,w=0).

        An explicit initial guess is a mixed coefficient vector. This low-level
        interface does not project the problem's scalar initial_guess callback
        or infer w by differentiating that field.
        """
        self.check_precision()
        if initial_guess is None:
            initial = self.arithmetic.zeros(self.unknown_count)
        else:
            _, _, state = self._state(initial_guess)
            initial = self.arithmetic.vector(state)

        def solve_linear(matrix, rhs):
            self.check_precision()
            try:
                return self.arithmetic.solve(matrix, rhs)
            except (np.linalg.LinAlgError, ZeroDivisionError) as error:
                raise LinearSolveError(str(error)) from error

        return NonlinearSystem(
            residual=self.residual, jacobian=self.jacobian,
            solve_linear=solve_linear, residual_norm=self.residual_norm,
            initial_guess=lambda: initial.copy(), scalar=self.arithmetic.scalar,
            isfinite=self.arithmetic.isfinite,
        )

    def with_plan(self, plan):
        """Build a new fixed system, retaining the same physical snapshot."""
        self.check_precision()
        return type(self)._from_snapshot(self.physical, self.basis, plan)

    def refined(self, order_factor=2, subdivisions_factor=1):
        """Reassemble on a refined plan; existing coefficients remain valid."""
        self.check_precision()
        return self.with_plan(self.plan.refined(order_factor, subdivisions_factor))

    def check_refinement(self, coefficients, *, atol, rtol, q=0,
                         order_factor=2, subdivisions_factor=1):
        """Compare R and DR on two plans at the same fixed state and q."""
        _, _, state = self._state(coefficients)
        q = self._parameter(q)
        atol = self._finite(atol, "atol", ValueError)
        rtol = self._finite(rtol, "rtol", ValueError)
        if atol < 0 or rtol < 0:
            raise ValueError("refinement tolerances must be nonnegative")
        state = self.arithmetic.vector(state)
        refined = self.refined(order_factor, subdivisions_factor)
        residual, fine_residual = self.residual(state, q), refined.residual(state, q)
        jacobian, fine_jacobian = self.jacobian(state, q), refined.jacobian(state, q)
        residual_difference = self._finite(self.arithmetic.norm(fine_residual-residual),
                                           "residual refinement difference")
        jacobian_difference = self._finite(self.arithmetic.norm(fine_jacobian-jacobian),
                                           "Jacobian refinement difference")
        residual_target = self._finite(atol+rtol*self.arithmetic.norm(fine_residual),
                                      "residual refinement target")
        jacobian_target = self._finite(atol+rtol*self.arithmetic.norm(fine_jacobian),
                                      "Jacobian refinement target")
        return AssemblyRefinementCheck(
            residual_difference, residual_target, jacobian_difference, jacobian_target,
            bool(residual_difference <= residual_target and jacobian_difference <= jacobian_target),
            refined,
        )
