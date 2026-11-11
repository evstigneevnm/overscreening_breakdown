"""Physical nonlinear manufactured cases covering every method and precision."""

import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown import solve
from overscreening_breakdown.manufactured import nonlinear_case
from tests.manufactured._checks import boundary_errors, collocation_diagnostics, native_values


MPMATH_PRECISIONS = (15, 30, 50, 100, 120)
NUMPY_METHODS = ("newton", "root", "root_no_jac", "nonlin_solve")
SAMPLE_POINTS = (0, 0.1, 0.5, 1, 2, 5, 10)


class NonlinearManufacturedLightTests(unittest.TestCase):
    def configurations(self):
        for method in NUMPY_METHODS:
            yield method, False, 15
        for precision in MPMATH_PRECISIONS:
            yield "newton", True, precision

    def check_case(self, number, size, method, use_mpmath, precision):
        case = nonlinear_case(number)
        previous_bits = mp.mp.prec
        tolerance = "1e-{}".format(max(10, precision-10)) if use_mpmath else "1e-11"
        result = solve(
            case.make_problem(), size, solver="collocation", L=2,
            method=method, use_mpmath=use_mpmath, precision=precision,
            tolerance=tolerance, use_globalization=False,
        )
        diagnostic = (result.scipy_result if result.newton_result is None
                      else result.newton_result.reason)
        self.assertTrue(result.converged, diagnostic)
        self.assertEqual(result.unknown_count, size+sum(
            value is not None for row in result.physical.boundary_conditions for value in row))
        arithmetic = result.arithmetic
        bound = arithmetic.scalar("1e-{}".format(max(8, precision-12))
                                  if use_mpmath else "1e-8")
        if use_mpmath:
            self.assertTrue(all(isinstance(value, type(arithmetic.scalar(0)))
                                for value in result.coefficients))
        else:
            self.assertEqual(result.coefficients.dtype, np.dtype("float64"))
        if result.newton_result is not None:
            self.assertLess(result.newton_result.residual_norm, bound)
            self.assertEqual(result.newton_result.homotopy, 0)
        else:
            self.assertIsNotNone(result.scipy_result)
            self.assertLessEqual(result.scipy_result.residual_norm,
                                 result.scipy_result.target_norm)
            self.assertEqual(result.scipy_result.homotopy, 0)
        _, normalized_defect, error = collocation_diagnostics(result, case)
        self.assertLess(normalized_defect, bound)
        self.assertLess(max(boundary_errors(result)), bound)
        samples = np.asarray(SAMPLE_POINTS, dtype=object if use_mpmath else float)
        actual = result.u(samples)
        expected = native_values(case.u, samples, arithmetic)
        self.assertLess(max(abs(value) for value in actual-expected),
                        5e-4 if size == 12 else 1e-5)
        self.assertTrue(np.isfinite(error))
        self.assertLess(error, 2e-4 if size == 12 else 1e-5)
        self.assertEqual(mp.mp.prec, previous_bits)
        return error, np.asarray(actual, dtype=float)

    def test_second_order_accuracy_refines_for_every_method_and_precision(self):
        reference = {}
        for method, use_mpmath, precision in self.configurations():
            with self.subTest(method=method, mpmath=use_mpmath, dps=precision):
                errors = {}
                for size in (12, 24):
                    errors[size], samples = self.check_case(1, size, method, use_mpmath, precision)
                    if size not in reference:
                        reference[size] = samples
                    else:
                        np.testing.assert_allclose(samples, reference[size], rtol=1e-8, atol=1e-9)
                self.assertLess(errors[24], errors[12]/10)

    def test_fourth_order_accuracy_for_every_method_and_precision(self):
        reference = None
        for method, use_mpmath, precision in self.configurations():
            with self.subTest(method=method, mpmath=use_mpmath, dps=precision):
                _, samples = self.check_case(2, 24, method, use_mpmath, precision)
                if reference is None:
                    reference = samples
                else:
                    np.testing.assert_allclose(samples, reference, rtol=1e-8, atol=1e-9)
