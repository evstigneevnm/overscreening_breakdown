"""Large BSK trial fields remain usable by the actual discrete operators."""

import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown import BSKProblem
from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.collocation.adapter import CollocationProblemAdapter
from overscreening_breakdown.collocation.spectral1d import solve_nonlinear_problem
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.galerkin.finite_assembly import FiniteGalerkinAssembly
from overscreening_breakdown.galerkin.finite_basis import FiniteGalerkinBasis
from overscreening_breakdown.quadrature import QuadratureFactory


class BSKOperatorStabilityTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)

    def problem(self, finite, sign):
        wall = sign*1000
        return BSKProblem(
            sigma="3", mu="0.1", gamma="0.5", delta="2",
            domain=(0, 2 if finite else "inf"),
            boundary_conditions=((wall, None, None, 0, None),
                                 (wall, 0, None, None, None) if finite else (None,)*5),
        )

    def assert_operator(self, system, state, use_mpmath):
        residual = system.residual(state)
        jacobian = system.jacobian(state)
        entries = (jacobian[i, j] for i in range(len(state)) for j in range(len(state)))
        self.assertTrue(all(system.isfinite(value) for value in residual))
        self.assertTrue(all(system.isfinite(value) for value in entries))

        # Differentiate the assembled residual, including boundary/lift terms.
        # This does not ask Newton to converge a deliberately difficult state.
        scalar = system.scalar
        values = [scalar((-1)**i)/scalar(i+1) for i in range(len(state))]
        direction = state.copy()
        for index, value in enumerate(values):
            direction[index] = value
        step = scalar("1e-12" if use_mpmath else "1e-3")
        difference = (system.residual(state+step*direction)
                      -system.residual(state-step*direction))/(2*step)
        product = jacobian*direction if use_mpmath else jacobian @ direction
        norm = state.ctx.norm if use_mpmath else np.linalg.norm
        relative = norm(difference-product)/max(scalar(1), norm(product))
        self.assertLess(relative, scalar("1e-20" if use_mpmath else "1e-7"))

    def test_collocation_large_signed_trial_fields_on_both_domains(self):
        for use_mpmath in (False, True):
            for finite in (False, True):
                for sign in (-1, 1):
                    with self.subTest(mpmath=use_mpmath, finite=finite, sign=sign):
                        a = MpmathArithmetic(50) if use_mpmath else NumpyArithmetic()
                        adapter = CollocationProblemAdapter(self.problem(finite, sign).bind(a))
                        solver = solve_nonlinear_problem(
                            6, problem=adapter, L=1,
                            use_mpmath=use_mpmath, prec=50, use_globalization=False,
                        )
                        system = solver.nonlinear_system()
                        state = system.initial_guess()*0
                        # The constant Chebyshev mode reaches +/-1000 at every
                        # physical collocation node. Boundary residuals remain
                        # part of the test, including the half-line far trace.
                        state[0] = system.scalar(sign*1000)
                        points = solver.all_discrete_points_in_domain()
                        values = solver.solution_in_domain(state, points.copy())
                        self.assertTrue(all(abs(value) > 900 for value in values))
                        self.assert_operator(system, state, use_mpmath)

    def test_galerkin_large_signed_trial_fields_on_both_domains(self):
        for use_mpmath in (False, True):
            for finite in (False, True):
                for sign in (-1, 1):
                    with self.subTest(mpmath=use_mpmath, finite=finite, sign=sign):
                        a = MpmathArithmetic(50) if use_mpmath else NumpyArithmetic()
                        problem = self.problem(finite, sign)
                        basis = (FiniteGalerkinBasis(4, a, domain=problem.domain) if finite
                                 else GalerkinBasis(4, a, L=1))
                        plan = QuadratureFactory(a, domain=problem.domain, L=1).plan(8)
                        assembly_type = FiniteGalerkinAssembly if finite else GalerkinAssembly
                        assembly = assembly_type(problem, basis, plan)
                        state = a.zeros(basis.unknown_count)
                        endpoints = (sign*1000, sign*1000) if finite else (sign*1000,)
                        values = [basis.evaluate_fields(state, y, *endpoints)[0]
                                  for y in plan.nodes]
                        self.assertGreater(max(abs(value) for value in values), 900)
                        self.assert_operator(assembly.nonlinear_system(), state, use_mpmath)


if __name__ == "__main__":
    unittest.main()
