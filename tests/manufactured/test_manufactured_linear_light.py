"""Light linear manufactured cases through the packaged common solver."""

import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown import solve
from overscreening_breakdown.manufactured import linear_case
from tests.manufactured._checks import boundary_errors, collocation_diagnostics, native_values


MPMATH_PRECISIONS = (15, 30, 50, 100, 120)
RESOLUTIONS = (12, 24)
SAMPLE_POINTS = (0, 0.125, 0.5, 1, 2, 4, 8)


class LinearManufacturedLightTests(unittest.TestCase):
    def check_problem(self, number, fine_l2_bound, sample_bounds):
        case = linear_case(number)
        reference_samples = {}
        previous_bits = mp.mp.prec
        for precision in (None,) + MPMATH_PRECISIONS:
            errors = []
            for resolution, sample_bound in zip(RESOLUTIONS, sample_bounds):
                with self.subTest(case=case.identifier, N=resolution, dps=precision):
                    tolerance = ("1e-{}".format(max(9, precision-8))
                                 if precision is not None else "1e-10")
                    result = solve(
                        case.make_problem(), resolution, solver="collocation", L=2,
                        use_mpmath=precision is not None, precision=precision or 15,
                        tolerance=tolerance, use_globalization=False,
                    )
                    self.assertTrue(result.converged, result.newton_result.reason)
                    self.assertEqual(result.newton_result.homotopy, 0)
                    arithmetic = result.arithmetic
                    coefficients = result.coefficients
                    if precision is None:
                        self.assertEqual(coefficients.dtype, np.dtype("float64"))
                    else:
                        self.assertEqual(arithmetic.dps, precision)
                        self.assertTrue(all(isinstance(value, type(arithmetic.scalar(0)))
                                            for value in coefficients))
                    relative, _, error = collocation_diagnostics(result, case)
                    bound = arithmetic.scalar("1e-10" if precision is None else tolerance)
                    self.assertLess(relative, bound)
                    self.assertLess(max(boundary_errors(result)), arithmetic.scalar("1e-9"))
                    samples = np.asarray(SAMPLE_POINTS, dtype=object if precision else float)
                    actual = result.u(samples)
                    expected = native_values(case.u, samples, arithmetic)
                    self.assertLess(max(abs(value) for value in actual-expected), sample_bound)
                    self.assertTrue(np.isfinite(error))
                    errors.append(error)
                    values = np.asarray(actual, dtype=float)
                    if precision is None:
                        reference_samples[resolution] = values
                    else:
                        np.testing.assert_allclose(values, reference_samples[resolution],
                                                   rtol=1e-8, atol=1e-10)
                    self.assertEqual(mp.mp.prec, previous_bits)
            if len(errors) == 2:
                self.assertLess(errors[-1], fine_l2_bound)
                self.assertLess(errors[-1], errors[0]/10)

    def test_fourth_order_exponential(self):
        self.check_problem(2, fine_l2_bound=2e-6, sample_bounds=(2e-3, 2e-6))

    def test_second_order_gaussian(self):
        self.check_problem(3, fine_l2_bound=1e-4, sample_bounds=(2e-2, 2e-4))

    def test_fourth_order_rational(self):
        self.check_problem(5, fine_l2_bound=1e-11, sample_bounds=(1e-7, 1e-11))
