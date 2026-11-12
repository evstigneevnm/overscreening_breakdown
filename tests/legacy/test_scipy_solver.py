"""Regression checks for the public SciPy nonlinear solver integration.

Run with ``python -m pytest tests/legacy/test_scipy_solver.py``.
"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy.optimize import OptimizeResult

from overscreening_breakdown.problems.bsk import (
    overscreening_breakdown,
    overscreening_breakdown_segment,
)
from overscreening_breakdown.collocation.spectral1d import solve_nonlinear_problem
from overscreening_breakdown.nonlinear_solver import LinearSolveError, solve_newton


class ScipySolverTests(unittest.TestCase):
    def setUp(self):
        original_precision = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_precision)
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def solver(self, method="nonlin_solve", segment=False, use_mpmath=False,
               use_globalization=False):
        problem = (
            overscreening_breakdown_segment() if segment else overscreening_breakdown()
        )
        problem.set_parameters(sigma=10, mu=0, gamma=0.5, delta=1, u0=1)
        return solve_nonlinear_problem(
            8,
            problem=problem,
            L=2,
            tolerance=1e-10,
            use_method=method,
            use_globalization=use_globalization,
            use_mpmath=use_mpmath,
            prec=50,
        )

    def check_numerical_solution(self, segment):
        solutions = {}
        for method in ("newton", "root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                solver = self.solver(method=method, segment=segment)
                coefficients = solver.solve_problem()
                solutions[method] = coefficients
                self.assertTrue(solver.is_converged())
                self.assertLess(solver.residual_L2_norm(coefficients), 1e-9)
                self.assertAlmostEqual(
                    np.asarray(
                        solver.obtain_solution_in_domain(coefficients, np.array([0.0]))
                    ).item(),
                    1,
                    places=10,
                )
                third_derivative_at_zero = sum(
                    solver.psi_with_derivative_value_at_right(k, 3) * value
                    for k, value in enumerate(coefficients)
                )
                self.assertAlmostEqual(third_derivative_at_zero, 0, places=10)
                if segment:
                    self.assertAlmostEqual(
                        np.asarray(
                            solver.obtain_solution_in_domain(coefficients, np.array([50.0]))
                        ).item(),
                        0,
                        places=10,
                    )
                    first_derivative_at_end = sum(
                        solver.psi_with_derivative_value_at_left(k, 1) * value
                        for k, value in enumerate(coefficients)
                    )
                    self.assertAlmostEqual(first_derivative_at_end, 0, places=10)
        np.testing.assert_allclose(
            solutions["nonlin_solve"], solutions["newton"], rtol=1e-8, atol=1e-10
        )
        for method in ("root", "root_no_jac"):
            np.testing.assert_allclose(
                solutions["nonlin_solve"], solutions[method], rtol=1e-8, atol=1e-10
            )

    def test_legacy_method_solves_half_line_problem(self):
        self.check_numerical_solution(segment=False)

    def test_legacy_method_solves_finite_segment_problem(self):
        self.check_numerical_solution(segment=True)

    def test_newton_reuses_a_converged_initial_solution_without_iteration(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                solver = self.solver(method="newton", use_mpmath=use_mpmath)
                coefficients = solver.solve_problem()
                restarted = solver.solve_problem(c0=coefficients)
                result = solver.get_newton_result()
                self.assertTrue(solver.is_converged())
                self.assertTrue(result.converged)
                self.assertEqual(result.iterations, 0)
                self.assertEqual(result.homotopy, 0)
                self.assertLess(solver.residual_L2_norm(restarted), 1e-9)

    def test_newton_continuation_preserves_wall_data_and_solves_physical_problem(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                solver = self.solver(
                    method="newton", use_mpmath=use_mpmath,
                    use_globalization=True,
                )
                continuation_starts = []
                failed = False

                def force_one_linear_failure(system, **options):
                    nonlocal failed
                    native_solve = system.solve_linear
                    if options.get("parameter", 0) > 0:
                        continuation_starts.append(system.initial_guess().copy())

                    def solve_linear(matrix, rhs):
                        nonlocal failed
                        if not failed:
                            failed = True
                            raise LinearSolveError("exercise continuation")
                        return native_solve(matrix, rhs)

                    # Force recovery without relying on a fragile difficult
                    # spatial problem; all later systems use the real assembly.
                    return solve_newton(
                        replace(system, solve_linear=solve_linear),
                        **options,
                    )

                with patch("overscreening_breakdown.continuation.solve_newton", side_effect=force_one_linear_failure):
                    coefficients = solver.solve_problem()
                result = solver.get_newton_result()
                self.assertTrue(solver.is_converged())
                self.assertEqual(result.homotopy, 0)
                self.assertEqual(result.stages[-1].homotopy, 0)
                self.assertTrue(result.stages[-1].converged)
                self.assertTrue(any(stage.homotopy > 0 for stage in result.stages))
                self.assertTrue(result.reference_result.converged)
                self.assertTrue(continuation_starts)
                self.assertLess(solver.residual_L2_norm(coefficients), 1e-9)
                for restart in (result.reference_result.state, *continuation_starts):
                    wall_value = np.asarray(solver.obtain_solution_in_domain(
                        restart, np.array([0.0])
                    )).item()
                    self.assertAlmostEqual(wall_value, 1, places=12)
                    third_derivative = sum(
                        solver.psi_with_derivative_value_at_right(k, 3) * value
                        for k, value in enumerate(restart)
                    )
                    self.assertAlmostEqual(third_derivative, 0, places=10)

    def test_newton_diagnostics_clear_when_switching_to_scipy(self):
        solver = self.solver(method="newton")
        coefficients = solver.solve_problem()
        self.assertIsNotNone(solver.get_newton_result())
        solver.set_method("root")
        with patch("overscreening_breakdown.collocation.spectral1d.optimize.root", return_value=OptimizeResult(
            x=coefficients, success=True,
        )):
            solver.solve_problem()
        self.assertIsNone(solver.get_newton_result())

    def test_success_returns_scipy_solution_and_sets_convergence(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                solver = self.solver("newton")
                expected = solver.solve_problem().copy()
                solver.set_method(method)
                result = OptimizeResult(x=expected, success=True, status=1,
                                        message="reported convergence", nfev=7, njev=2)
                with patch("overscreening_breakdown.collocation.spectral1d.optimize.root", return_value=result):
                    actual = solver.solve_problem()
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(solver.is_converged())
                diagnostics = solver.get_scipy_result()
                self.assertTrue(diagnostics.converged and diagnostics.scipy_success)
                self.assertEqual(diagnostics.status, 1)
                self.assertEqual((diagnostics.nfev, diagnostics.njev), (7, 2))
                self.assertLessEqual(diagnostics.residual_norm, diagnostics.target_norm)

    def test_scipy_stagnation_can_still_return_a_verified_root(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                solver = self.solver("newton")
                expected = solver.solve_problem().copy()
                solver.set_method(method)
                result = OptimizeResult(x=expected, success=False, status=5,
                                        message="iteration is not making good progress",
                                        fun=np.full(expected.shape, np.nan), nfev=23)
                with patch("overscreening_breakdown.collocation.spectral1d.optimize.root", return_value=result):
                    actual = solver.solve_problem()
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(solver.is_converged())
                diagnostics = solver.get_scipy_result()
                self.assertFalse(diagnostics.scipy_success)
                self.assertEqual(diagnostics.status, 5)
                self.assertEqual(diagnostics.reason, "converged")
                self.assertLessEqual(diagnostics.residual_norm, diagnostics.target_norm)

    def test_scipy_success_and_cached_zero_residual_do_not_certify_boundary_data(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                solver = self.solver(method)
                # With mu=0, u=0 satisfies the differential equation, but not
                # the prescribed wall value u(0)=1. The boundary rows must be
                # included in the freshly assembled acceptance residual.
                coefficients = np.zeros(10)
                result = OptimizeResult(x=coefficients, success=True, status=1,
                                        fun=np.zeros(10), message="reported success")
                with patch("overscreening_breakdown.collocation.spectral1d.optimize.root", return_value=result):
                    actual = solver.solve_problem()
                np.testing.assert_array_equal(actual, coefficients)
                self.assertFalse(solver.is_converged())
                diagnostics = solver.get_scipy_result()
                self.assertTrue(diagnostics.scipy_success)
                self.assertEqual(diagnostics.reason, "residual_too_large")
                self.assertGreater(diagnostics.residual_norm, diagnostics.target_norm)
                self.assertGreaterEqual(diagnostics.residual_max, 1)

    def test_failure_returns_last_iterate_and_clears_previous_success(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                solver = self.solver("newton")
                verified = solver.solve_problem().copy()
                solver.set_method(method)
                last_iterate = np.linspace(0, 0.5, 10)
                results = [
                    OptimizeResult(x=verified, success=True),
                    OptimizeResult(x=last_iterate, success=False),
                ]
                with patch("overscreening_breakdown.collocation.spectral1d.optimize.root", side_effect=results):
                    solver.solve_problem()
                    self.assertTrue(solver.is_converged())
                    actual = solver.solve_problem()
                np.testing.assert_array_equal(actual, last_iterate)
                self.assertFalse(solver.is_converged())
                self.assertFalse(solver.get_scipy_result().converged)

    def test_unexpected_exception_propagates_and_clears_previous_success(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                solver = self.solver("newton")
                verified = solver.solve_problem().copy()
                solver.set_method(method)
                results = [
                    OptimizeResult(x=verified, success=True),
                    RuntimeError("unexpected solver error"),
                ]
                with patch("overscreening_breakdown.collocation.spectral1d.optimize.root", side_effect=results):
                    solver.solve_problem()
                    self.assertTrue(solver.is_converged())
                    with self.assertRaisesRegex(RuntimeError, "unexpected solver error"):
                        solver.solve_problem()
                self.assertFalse(solver.is_converged())
                self.assertIsNone(solver.get_scipy_result())

    def test_scipy_diagnostics_clear_when_switching_back_to_newton(self):
        solver = self.solver("root_no_jac")
        solver.solve_problem()
        self.assertIsNotNone(solver.get_scipy_result())
        solver.set_method("newton")
        solver.solve_problem()
        self.assertTrue(solver.is_converged())
        self.assertIsNone(solver.get_scipy_result())
        self.assertIsNotNone(solver.get_newton_result())

    def test_scipy_methods_reject_mpmath(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                solver = self.solver(method=method, use_mpmath=True)
                with self.assertRaisesRegex(ValueError, "only 'newton'.*mpmath"):
                    solver.solve_problem()
                self.assertFalse(solver.is_converged())


if __name__ == "__main__":
    unittest.main()
