"""SciPy status is diagnostic; an independently checked residual certifies roots."""

from dataclasses import FrozenInstanceError
import unittest

import numpy as np
from scipy.optimize import OptimizeResult

from overscreening_breakdown.arithmetic import MpmathArithmetic
from overscreening_breakdown.scipy_solver import verify_scipy_result


class ScipyVerificationTests(unittest.TestCase):
    def verify(self, result, residual, **options):
        return verify_scipy_result(result, residual=residual,
                                   target_norm=options.pop("target_norm", "1e-10"),
                                   method=options.pop("method", "root_no_jac"), **options)

    def test_a_verified_root_is_accepted_despite_unsuccessful_step_termination(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                scipy = OptimizeResult(x=np.array([2.0]), success=False, status=5,
                                       message="The iteration is not making good progress.",
                                       nfev=17, njev=3, fun=np.array([np.nan]))
                result = self.verify(scipy, lambda x: x*x-4, method=method)
                self.assertTrue(result.converged)
                self.assertEqual(result.reason, "converged")
                self.assertFalse(result.scipy_success)
                self.assertEqual(result.status, 5)
                self.assertIn("not making good progress", result.message)
                self.assertEqual((result.nfev, result.njev), (17, 3))
                self.assertEqual(result.residual_norm, 0)
                self.assertEqual(result.residual_max, 0)

    def test_success_and_cached_zero_residual_do_not_certify_a_nonroot(self):
        scipy = OptimizeResult(x=np.array([1.0]), success=True, status=1,
                               fun=np.array([0.0]))
        result = self.verify(scipy, lambda x: x*x-4)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "residual_too_large")
        self.assertTrue(result.scipy_success)
        self.assertEqual(result.residual_norm, 3)
        self.assertEqual(result.residual_max, 3)

    def test_full_residual_includes_boundary_rows_and_retains_maximum(self):
        scipy = OptimizeResult(x=np.zeros(4), success=True)
        values = np.array([0.0, 0.0, 0.0, 2.0])
        result = self.verify(scipy, lambda x: values, target_norm="0.5")
        self.assertFalse(result.converged)
        self.assertEqual(result.residual_norm, 1)
        self.assertEqual(result.residual_max, 2)
        self.assertIsNone(result.status)
        self.assertIsNone(result.nfev)
        self.assertIsNone(result.njev)

    def test_threshold_is_inclusive_without_extra_tolerance(self):
        for value, accepted in ((np.float64(2), True), (np.nextafter(2.0, np.inf), False)):
            with self.subTest(value=value):
                result = self.verify(OptimizeResult(x=np.zeros(2), success=not accepted),
                                     lambda x: np.full(2, value), target_norm=2)
                self.assertEqual(result.converged, accepted)
        exact = self.verify(OptimizeResult(x=np.array([1.0])), lambda x: x-1, target_norm=0)
        self.assertTrue(exact.converged)

    def test_nonfinite_coefficients_are_rejected_without_evaluating_callback(self):
        def forbidden(x):
            raise AssertionError("A nonfinite coefficient vector must not reach the residual")
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                result = self.verify(OptimizeResult(x=np.array([value]), success=True), forbidden)
                self.assertFalse(result.converged)
                self.assertEqual(result.reason, "nonfinite_state")
                self.assertEqual(result.residual_norm, np.inf)

    def test_nonfinite_residuals_and_overflow_fail_while_programming_errors_propagate(self):
        scipy = OptimizeResult(x=np.array([1.0]), success=True)
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                result = self.verify(scipy, lambda x: np.array([value]))
                self.assertFalse(result.converged)
                self.assertEqual(result.reason, "nonfinite_residual")
        def overflow(x):
            raise OverflowError("outside numeric range")
        result = self.verify(scipy, overflow)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "residual_overflow")
        def broken(x):
            raise RuntimeError("callback defect")
        with self.assertRaisesRegex(RuntimeError, "callback defect"):
            self.verify(scipy, broken)

    def test_wrong_shapes_and_dtypes_are_errors_without_precision_narrowing(self):
        wrong_states = ([1.0], np.array([1]), np.array([1], dtype=np.float32),
                        np.array([1+0j]), MpmathArithmetic(120).vector([1]))
        for state in wrong_states:
            with self.subTest(state_type=type(state)), self.assertRaises(TypeError):
                self.verify(OptimizeResult(x=state), lambda x: x-1)
        for state in (np.array(1.0), np.array([[1.0]]), np.array([], dtype=float)):
            with self.subTest(shape=state.shape), self.assertRaises(ValueError):
                self.verify(OptimizeResult(x=state), lambda x: x-1)
        for values in (np.array([0.0, 0.0]), np.array([[0.0]])):
            with self.subTest(residual_shape=values.shape), self.assertRaises(ValueError):
                self.verify(OptimizeResult(x=np.array([1.0])), lambda x: values)
        with self.assertRaises(TypeError):
            self.verify(OptimizeResult(x=np.array([1.0])), lambda x: np.array([0]))

    def test_scaled_rms_retains_tiny_values_and_handles_large_finite_vectors(self):
        for value in (np.float64("1e-300"), np.float64("1e308")):
            with self.subTest(value=value), np.errstate(over="raise", invalid="raise", under="raise"):
                values = np.array([value, -value])
                result = self.verify(OptimizeResult(x=np.zeros(2)), lambda x: values,
                                     target_norm=value)
                self.assertTrue(result.converged)
                self.assertEqual(result.residual_norm, value)
                self.assertEqual(result.residual_max, value)
        values = np.array([np.float64("1e308"), np.float64("1e-308")])
        with np.errstate(over="raise", invalid="raise", under="raise"):
            result = self.verify(OptimizeResult(x=np.zeros(2)), lambda x: values,
                                 target_norm="1e308")
        self.assertTrue(np.isfinite(result.residual_norm))
        self.assertGreater(result.residual_norm, np.float64("7e307"))
        self.assertLess(result.residual_norm, np.float64("8e307"))

    def test_native_immutable_diagnostics_do_not_alias_or_mutate_solver_data(self):
        scipy = OptimizeResult(x=np.array([1.0, 2.0]), success=True)
        values = np.array([3.0, 4.0])
        calls = []
        def residual(state):
            calls.append(state.copy())
            state[:] = 999
            return values
        result = self.verify(scipy, residual, target_norm=4)
        np.testing.assert_array_equal(scipy.x, [1, 2])
        np.testing.assert_array_equal(values, [3, 4])
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(result.residual_norm, np.sqrt(12.5))
        self.assertEqual(result.homotopy, 0)
        for value in (result.residual_norm, result.residual_max, result.target_norm, result.homotopy):
            self.assertIsInstance(value, np.float64)
        with self.assertRaises(FrozenInstanceError):
            result.converged = False

    def test_method_and_residual_threshold_are_validated(self):
        scipy = OptimizeResult(x=np.array([1.0]))
        for target in (-1, "nan", "inf", True, np.bool_(False), [1], 1j,
                       MpmathArithmetic(120).scalar("1e-80")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.verify(scipy, lambda x: x-1, target_norm=target)
        with self.assertRaises(ValueError):
            self.verify(scipy, lambda x: x-1, method="newton")


if __name__ == "__main__":
    unittest.main()
