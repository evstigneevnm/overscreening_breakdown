"""Failure-injection contracts of the mixed system's shared Newton interface."""

from dataclasses import replace
import unittest

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.continuation import solve_continuation
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.manufactured import rational_case
from overscreening_breakdown.nonlinear_solver import LinearSolveError
from overscreening_breakdown.quadrature import QuadratureFactory


class GalerkinGlobalizationTests(unittest.TestCase):
    def test_continuation_recovers_a_linear_failure_and_preserves_the_wall_lift(self):
        case = rational_case("smooth", nonlinear=True)
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            basis = GalerkinBasis(4, arithmetic)
            assembly = GalerkinAssembly(case.make_problem(), basis,
                                        QuadratureFactory(arithmetic).plan(8))
            system = assembly.nonlinear_system()
            calls = [0]

            def fail_first(matrix, rhs):
                calls[0] += 1
                if calls[0] == 1:
                    raise LinearSolveError("forced first physical step failure")
                return system.solve_linear(matrix, rhs)

            accepted = []
            tolerance = arithmetic.scalar("1e-42" if arithmetic.use_mpmath else "1e-11")
            result = solve_continuation(
                replace(system, solve_linear=fail_first), atol=tolerance, max_iterations=30,
                reference_factory=assembly.reference_operator, use_globalization=True,
                callback=lambda state, record: accepted.append(state),
            )
            self.assertTrue(result.converged, result.reason)
            self.assertEqual(result.homotopy, 0)
            self.assertEqual(result.stages[0].reason, "linear_solve_failed")
            self.assertTrue(any(stage.homotopy > 0 for stage in result.stages))
            self.assertTrue(accepted)
            for state in accepted:
                self.assertEqual(basis.evaluate_fields(state, -1, u_b=1)[0], 1)
            for x in ("0", "0.5", "1", "4"):
                coordinate = arithmetic.scalar(x)
                u, w = basis.evaluate_fields(result.state, basis.mapping.to_reference(coordinate), u_b=1)
                self.assertLess(abs(u-case.u(coordinate, arithmetic)), 1000*tolerance)
                self.assertLess(abs(w-case.w(coordinate, arithmetic)), 1000*tolerance)
