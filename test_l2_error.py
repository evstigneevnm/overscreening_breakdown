"""Regression checks for scalar quadrature and relative L2 errors.

Run with ``python -m unittest test_l2_error``.
"""

import unittest
import warnings

import mpmath as mp
import numpy as np

from spectral1d import collocation_discretization


class L2ErrorTests(unittest.TestCase):
    def setUp(self):
        original_precision = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_precision)
        warning_context = warnings.catch_warnings()
        warning_context.__enter__()
        self.addCleanup(warning_context.__exit__, None, None, None)
        warnings.filterwarnings(
            "error",
            message="Conversion of an array with ndim > 0 to a scalar is deprecated.*",
            category=DeprecationWarning,
        )

    def discretization(self, segment=False, use_mpmath=False):
        return collocation_discretization(
            4,
            domain=[0, 2] if segment else [0, np.inf],
            L=2,
            use_mpmath=use_mpmath,
            prec=80,
        )

    def coefficients(self, values, use_mpmath):
        return mp.matrix(values) if use_mpmath else np.array(values, dtype=float)

    def check_constant_errors(self, array_reference):
        for segment in (False, True):
            for use_mpmath in (False, True):
                for coefficient, expected in ((1, 0), (0, 1), (0.5, 0.5)):
                    with self.subTest(
                        segment=segment,
                        use_mpmath=use_mpmath,
                        coefficient=coefficient,
                    ):
                        discretization = self.discretization(segment, use_mpmath)
                        coefficients = self.coefficients([coefficient], use_mpmath)
                        one = mp.mpf(1) if use_mpmath else 1.0

                        def reference(x):
                            return np.array([one]) if array_reference else one

                        actual = discretization.L2_error(coefficients, reference)

                        self.assertEqual(np.ndim(actual), 0)
                        self.assertAlmostEqual(actual, expected, places=13)

    def test_known_errors_with_scalar_reference(self):
        self.check_constant_errors(array_reference=False)

    def test_known_errors_with_one_element_array_reference(self):
        self.check_constant_errors(array_reference=True)

    def test_nonconstant_reference_uses_the_domain_mapping(self):
        # In basis coordinates the reference is 2 + cos(t). The constant
        # approximation 2 has relative error sqrt((pi/2)/(9*pi/2)) = 1/3.
        for segment in (False, True):
            for use_mpmath in (False, True):
                with self.subTest(segment=segment, use_mpmath=use_mpmath):
                    discretization = self.discretization(segment, use_mpmath)
                    coefficients = self.coefficients([2], use_mpmath)

                    def reference(x):
                        cosine = x - 1 if segment else (x - 2) / (x + 2)
                        return 2 + cosine

                    actual = discretization.L2_error(coefficients, reference)

                    self.assertAlmostEqual(actual, 1 / 3, places=13)

    def test_mpmath_residual_is_computed_before_float_conversion(self):
        for segment in (False, True):
            with self.subTest(segment=segment):
                discretization = self.discretization(segment, use_mpmath=True)
                tiny_difference = mp.mpf("1e-40")
                coefficients = mp.matrix([1 + tiny_difference])

                actual = discretization.L2_error(coefficients, lambda x: mp.mpf(1))

                self.assertAlmostEqual(actual / 1e-40, 1.0, places=13)

    def test_basis_solution_remains_vectorized(self):
        for segment in (False, True):
            for use_mpmath in (False, True):
                with self.subTest(segment=segment, use_mpmath=use_mpmath):
                    discretization = self.discretization(segment, use_mpmath)
                    coefficients = self.coefficients([1, 0.5], use_mpmath)
                    points = np.array([0, np.pi / 2, np.pi])

                    actual = discretization.solution_in_basis(coefficients, points)

                    self.assertIsInstance(actual, np.ndarray)
                    self.assertEqual(actual.shape, points.shape)
                    np.testing.assert_allclose(
                        np.asarray(actual, dtype=float), [1.5, 1.0, 0.5], atol=1e-15
                    )


if __name__ == "__main__":
    unittest.main()
