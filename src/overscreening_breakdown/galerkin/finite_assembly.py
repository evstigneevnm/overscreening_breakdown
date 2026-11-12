"""Finite Petrov mixed assembly with value/third-derivative and value/slope data.

The first weak equation has the right slope load s_R*phi(R). The normalized
second equation has the left load delta²*q_b*psi(0). No value of w(R) is imposed.
Shared state, Newton, merit and quadrature-refinement utilities come from the
half-line assembly; finite trial/test blocks and boundary policy are separate.
"""

from contextlib import nullcontext

from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.galerkin.finite_basis import FiniteGalerkinBasis
from overscreening_breakdown.problems.flux import BoundFluxProblem
from overscreening_breakdown.quadrature import QuadraturePlan


class FiniteGalerkinAssembly(GalerkinAssembly):
    def __init__(self, problem, basis, plan, *, progress=None):
        if not isinstance(basis, FiniteGalerkinBasis):
            raise TypeError("basis must be a FiniteGalerkinBasis")
        basis.check_precision()
        if isinstance(problem, BoundFluxProblem):
            raise TypeError("supply an unbound physical problem description")
        bind = getattr(problem, "bind", None)
        if not callable(bind):
            raise TypeError("problem must provide bind(arithmetic)")
        self._initialize(bind(basis.arithmetic), basis, plan, progress=progress)

    def _initialize_numerical(self, physical, basis, plan):
        if not isinstance(physical, BoundFluxProblem) or not isinstance(basis, FiniteGalerkinBasis):
            raise TypeError("finite assembly requires a bound flux problem and finite basis")
        physical.check_precision()
        if physical.arithmetic is not basis.arithmetic or physical.domain != basis.domain:
            raise ValueError("physical problem and finite basis must share an owner and domain")
        if not isinstance(plan, QuadraturePlan):
            raise TypeError("plan must be a QuadraturePlan from QuadratureFactory")
        for name, value in (("physical", physical), ("basis", basis), ("plan", plan)):
            object.__setattr__(self, name, value)
        if physical.delta <= 0:
            raise ValueError("finite mixed assembly requires delta > 0")
        left, right = physical.boundary_conditions
        if (left[0] is None or left[3] is None or right[0] is None or right[1] is None
                or any(left[index] is not None for index in (1, 2, 4))
                or any(right[index] is not None for index in (2, 3, 4))):
            raise ValueError("finite mixed data must prescribe left u/u''' and right u/u' only")
        scope = (self._progress.phase("basis_tabulation", level=3)
                 if self._progress is not None else nullcontext())
        with scope:
            object.__setattr__(self, "table", basis.tabulate(plan))
        self._validate_plan()
        panel_boundaries = {endpoint for panel in plan.panels for endpoint in panel}
        if any(basis.mapping.to_reference(point) not in panel_boundaries
               for point in physical.epsilon.interfaces):
            raise ValueError("quadrature must split every declared epsilon interface")
        object.__setattr__(self, "lift", basis.lift(left[0], right[0]))
        object.__setattr__(self, "epsilon_b", physical.epsilon.wall_value)
        delta_squared = self._finite(physical.delta**2, "delta squared", ValueError)
        if delta_squared <= 0:
            raise ValueError("delta squared must remain strictly positive in the selected arithmetic")
        object.__setattr__(self, "delta_squared", delta_squared)
        ratios = tuple(self._finite(physical.epsilon(x)/self.epsilon_b, "epsilon/epsilon_b", ValueError)
                       for x in plan.physical_nodes)
        if any(value <= 0 for value in ratios):
            raise ValueError("epsilon/epsilon_b must be strictly positive")
        object.__setattr__(self, "_epsilon_ratios", ratios)
        object.__setattr__(self, "_lift_values", tuple(self.lift.value(y) for y in plan.nodes))
        object.__setattr__(self, "_lift_derivatives", tuple(self.lift.dx(y) for y in plan.nodes))
        scope = (self._progress.phase("constant_blocks", level=3)
                 if self._progress is not None else nullcontext())
        with scope:
            self._assemble_finite_terms(left[3], right[1])

    def _constant_terms(self, epsilon_ratios):
        table, weights, basis = self.table, self.plan.physical_weights, self.basis
        rows, lift_entries = [], []
        for i in range(basis.n_first_tests):
            rows.append(tuple(self._sum(weight*test[i]*trial[j]
                                       for weight, test, trial in zip(weights, table.phi_dx, table.v_dx))
                              for j in range(basis.n_u)) + tuple(
                -self._sum(weight*test[i]*trial[j] for weight, test, trial in zip(weights, table.phi, table.q))
                for j in range(basis.n_w)))
            lift_entries.append(self._sum(weight*test[i]*derivative
                                          for weight, test, derivative in
                                          zip(weights, table.phi_dx, self._lift_derivatives)))
        for i in range(basis.n_second_tests):
            rows.append(tuple(self._sum(weight*(epsilon-1)*test[i]*trial[j]
                                       for weight, epsilon, test, trial in
                                       zip(weights, epsilon_ratios, table.psi_dx, table.v_dx))
                              for j in range(basis.n_u)) + tuple(
                self._sum(weight*(self.delta_squared*epsilon*test_dx[i]*trial_dx[j]+test[i]*trial[j])
                          for weight, epsilon, test_dx, trial_dx, test, trial in
                          zip(weights, epsilon_ratios, table.psi_dx, table.q_dx, table.psi, table.q))
                for j in range(basis.n_w)))
            lift_entries.append(self._sum(weight*(epsilon-1)*test[i]*derivative
                                          for weight, epsilon, test, derivative in
                                          zip(weights, epsilon_ratios, table.psi_dx, self._lift_derivatives)))
        return tuple(rows), tuple(lift_entries)

    def _assemble_finite_terms(self, q_b, slope_R):
        rows, lift_entries = self._constant_terms(self._epsilon_ratios)
        wall_entries = tuple(self._finite(slope_R*trace, "right slope load", ValueError)
                             for trace in self.basis.phi(1)) + tuple(
            self._finite(self.delta_squared*q_b*trace, "left third-derivative load", ValueError)
            for trace in self.basis.psi(-1))
        object.__setattr__(self, "_linear_rows", rows)
        object.__setattr__(self, "_lift_entries", lift_entries)
        object.__setattr__(self, "_boundary_entries", wall_entries)
        object.__setattr__(self, "_affine_entries", tuple(
            self._finite(lift-load, "affine load", ValueError)
            for lift, load in zip(lift_entries, wall_entries)))

    def residual(self, coefficients, q=0):
        first, _, state = self._state(coefficients)
        self._parameter(q)
        result = list(self._linear_values(state))
        source = self._source_values(first)
        start = self.basis.n_first_tests
        for i in range(self.basis.n_second_tests):
            load = self._sum(weight*value*test[i] for weight, value, test in
                             zip(self.plan.physical_weights, source, self.table.psi))
            result[start+i] = self._finite(result[start+i]+load, "finite mixed residual")
        return self.arithmetic.vector(result)

    def jacobian(self, coefficients, q=0):
        first, _, _ = self._state(coefficients)
        self._parameter(q)
        derivative = self._source_values(first, derivative=True)
        rows = [list(row) for row in self._linear_rows]
        start = self.basis.n_first_tests
        for i in range(self.basis.n_second_tests):
            for j in range(self.basis.n_u):
                load = self._sum(weight*value*test[i]*trial[j] for weight, value, test, trial in
                                 zip(self.plan.physical_weights, derivative, self.table.psi, self.table.v))
                rows[start+i][j] = self._finite(rows[start+i][j]+load, "finite mixed Jacobian")
        return self.arithmetic.matrix(rows)

    def reference_operator(self, strength="1"):
        """Use constant wall epsilon and f_ref=epsilon_b*strength*u, retaining all BCs."""
        self.check_precision()
        strength = self._finite(strength, "reference strength", ValueError)
        if strength <= 0:
            raise ValueError("reference strength must be strictly positive")
        one = self.arithmetic.scalar(1)
        rows, lift_entries = self._constant_terms((one,)*len(self.plan.nodes))
        rows = [list(row) for row in rows]
        offset = [lift-load for lift, load in zip(lift_entries, self._boundary_entries)]
        start = self.basis.n_first_tests
        for i in range(self.basis.n_second_tests):
            for j in range(self.basis.n_u):
                load = self._sum(weight*strength*test[i]*trial[j] for weight, test, trial in
                                 zip(self.plan.physical_weights, self.table.psi, self.table.v))
                rows[start+i][j] = self._finite(rows[start+i][j]+load, "finite reference matrix")
            load = self._sum(weight*strength*lift*test[i] for weight, lift, test in
                             zip(self.plan.physical_weights, self._lift_values, self.table.psi))
            offset[start+i] = self._finite(offset[start+i]+load, "finite reference affine load")
        return self.arithmetic.matrix(rows), self.arithmetic.vector(offset)
