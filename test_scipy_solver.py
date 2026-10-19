"""Regression checks for the public SciPy nonlinear solver integration.

Run with ``python -m unittest test_scipy_solver``.
"""

from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy.optimize import OptimizeResult

from overscreening_breakdown import (
    overscreening_breakdown,
    overscreening_breakdown_segment,
)
from spectral1d import solve_nonlinear_problem


class ScipySolverTests(unittest.TestCase):
    def setUp(self):
        original_precision = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_precision)
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def solver(self, method="nonlin_solve", segment=False, use_mpmath=False):
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
            use_globalization=False,
            use_mpmath=use_mpmath,
            prec=50,
        )

    def check_numerical_solution(self, segment):
        solutions = {}
        for method in ("newton", "root", "nonlin_solve"):
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
        np.testing.assert_allclose(
            solutions["nonlin_solve"], solutions["root"], rtol=1e-8, atol=1e-10
        )

    def test_legacy_method_solves_half_line_problem(self):
        self.check_numerical_solution(segment=False)

    def test_legacy_method_solves_finite_segment_problem(self):
        self.check_numerical_solution(segment=True)

    def test_success_returns_scipy_solution_and_sets_convergence(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                solver = self.solver(method)
                expected = np.linspace(0, 0.5, 10)
                result = OptimizeResult(x=expected, success=True)
                with patch("spectral1d.optimize.root", return_value=result):
                    actual = solver.solve_problem()
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(solver.is_converged())

    def test_failure_returns_last_iterate_and_clears_previous_success(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                solver = self.solver(method)
                last_iterate = np.linspace(0, 0.5, 10)
                results = [
                    OptimizeResult(x=np.zeros(10), success=True),
                    OptimizeResult(x=last_iterate, success=False),
                ]
                with patch("spectral1d.optimize.root", side_effect=results):
                    solver.solve_problem()
                    self.assertTrue(solver.is_converged())
                    actual = solver.solve_problem()
                np.testing.assert_array_equal(actual, last_iterate)
                self.assertFalse(solver.is_converged())

    def test_unexpected_exception_propagates_and_clears_previous_success(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                solver = self.solver(method)
                results = [
                    OptimizeResult(x=np.zeros(10), success=True),
                    RuntimeError("unexpected solver error"),
                ]
                with patch("spectral1d.optimize.root", side_effect=results):
                    solver.solve_problem()
                    self.assertTrue(solver.is_converged())
                    with self.assertRaisesRegex(RuntimeError, "unexpected solver error"):
                        solver.solve_problem()
                self.assertFalse(solver.is_converged())

    def test_legacy_method_rejects_mpmath(self):
        solver = self.solver(use_mpmath=True)
        with self.assertRaisesRegex(ValueError, "only 'newton'.*mpmath"):
            solver.solve_problem()
        self.assertFalse(solver.is_converged())


if __name__ == "__main__":
    unittest.main()
