"""Initialization is bounded refinement, independent of solver convergence."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic
from overscreening_breakdown.initialization import initialize_fsolve, validate_initializer
from overscreening_breakdown.nonlinear_solver import NonlinearSystem, solve_newton


class InitializationTests(unittest.TestCase):
    def system(self):
        return NonlinearSystem(
            residual=lambda z, q: np.array([z[0]**2-2]),
            jacobian=lambda z, q: np.array([[2*z[0]]]),
            solve_linear=np.linalg.solve, residual_norm=np.linalg.norm,
            initial_guess=lambda: np.array([1.0]), scalar=np.float64,
            isfinite=np.isfinite,
        )

    def test_real_fsolve_improves_the_guess_without_mutating_it(self):
        original = np.array([1.0])
        result = initialize_fsolve(self.system(), original)
        self.assertTrue(result.accepted)
        self.assertLess(result.final_norm, result.initial_norm)
        self.assertLess(abs(result.state[0]-np.sqrt(2)), 0.01)
        self.assertEqual(original[0], 1)
        self.assertIsInstance(result.state[0], np.float64)
        self.assertEqual(result.evaluations, result.function_evaluations+result.jacobian_evaluations)

    def test_unsuccessful_scipy_status_can_supply_a_verified_improvement(self):
        def stopped(fun, initial, **options):
            candidate = np.array([1.4])
            fun(candidate)
            return candidate, {}, 2, "stopped before SciPy convergence"
        with patch("scipy.optimize.fsolve", side_effect=stopped):
            result = initialize_fsolve(self.system(), np.array([1.0]))
        self.assertTrue(result.accepted)
        self.assertFalse(result.scipy_success)
        self.assertEqual(result.scipy_status, 2)
        self.assertLess(result.final_norm, result.initial_norm)

    def test_scipy_success_does_not_certify_main_solver_convergence(self):
        def approximate(fun, initial, **options):
            candidate = np.array([1.2])
            fun(candidate)
            return candidate, {}, 1, "step termination"
        system = self.system()
        with patch("scipy.optimize.fsolve", side_effect=approximate):
            initial = initialize_fsolve(system, np.array([1.0]))
        self.assertTrue(initial.scipy_success and initial.accepted)
        result = solve_newton(replace(system, initial_guess=lambda: initial.state.copy()),
                              max_iterations=0, atol="1e-12")
        self.assertFalse(result.converged)
        self.assertGreater(result.residual_norm, 0.1)

    def test_non_improving_or_nonfinite_final_candidates_are_rejected(self):
        for value in (3.0, np.inf):
            def bad(fun, initial, **options):
                return np.array([value]), {}, 1, "reported success"
            with self.subTest(value=value), patch("scipy.optimize.fsolve", side_effect=bad):
                result = initialize_fsolve(self.system(), np.array([1.0]))
                self.assertFalse(result.accepted)
                np.testing.assert_array_equal(result.state, [1])

    def test_best_verified_candidate_survives_a_later_numerical_failure(self):
        def bad_after_progress(fun, initial, **options):
            fun(np.array([1.4]))
            fun(np.array([np.inf]))
        with patch("scipy.optimize.fsolve", side_effect=bad_after_progress):
            result = initialize_fsolve(self.system(), np.array([1.0]))
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "numerical_failure")
        self.assertEqual(result.state[0], 1.4)

    def test_combined_residual_and_jacobian_budget_is_hard(self):
        def expensive(fun, initial, fprime, **options):
            fun(np.array([1.4]))
            fprime(initial)
            while True:
                fun(initial)
        with patch("scipy.optimize.fsolve", side_effect=expensive):
            result = initialize_fsolve(self.system(), np.array([1.0]), max_evaluations=4)
        self.assertEqual(result.evaluations, 4)
        self.assertEqual(result.reason, "evaluation_limit")
        self.assertTrue(result.accepted)
        with patch("scipy.optimize.fsolve") as optimizer:
            result = initialize_fsolve(self.system(), np.array([1.0]), max_evaluations=1)
            optimizer.assert_not_called()
        self.assertFalse(result.accepted)
        self.assertEqual(result.evaluations, 1)

    def test_exact_and_nonfinite_initial_residuals_do_not_call_scipy(self):
        for residual, expected in ((lambda z, q: np.array([0.0]), "already_solved"),
                                   (lambda z, q: np.array([np.inf]), "numerical_failure")):
            with self.subTest(reason=expected), patch("scipy.optimize.fsolve") as optimizer:
                result = initialize_fsolve(replace(self.system(), residual=residual), np.array([1.0]))
                optimizer.assert_not_called()
                self.assertFalse(result.accepted)
                self.assertEqual(result.reason, expected)

    def test_callback_defects_and_malformed_arrays_propagate(self):
        def broken(z, q):
            raise RuntimeError("callback defect")
        with self.assertRaisesRegex(RuntimeError, "callback defect"):
            initialize_fsolve(replace(self.system(), residual=broken), np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "residual"):
            initialize_fsolve(replace(self.system(), residual=lambda z, q: np.array([1., 2.])),
                              np.array([1.0]))
        def call_jac(fun, initial, fprime, **options):
            fprime(initial)
        with patch("scipy.optimize.fsolve", side_effect=call_jac):
            with self.assertRaisesRegex(RuntimeError, "callback defect"):
                initialize_fsolve(replace(self.system(), jacobian=broken), np.array([1.0]))

    def test_backend_and_option_validation_prevents_float_narrowing(self):
        for options in ({"use_mpmath": True}, {"solver": "galerkin"},
                        {"tolerance": "nan"}, {"tolerance": 0}, {"tolerance": True},
                        {"max_evaluations": 0}, {"max_evaluations": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_initializer("fsolve", **options)
        with self.assertRaises(ValueError):
            validate_initializer("adjoint")
        with self.assertRaises(TypeError):
            initialize_fsolve(self.system(), MpmathArithmetic(120).vector([1]))


if __name__ == "__main__":
    unittest.main()
