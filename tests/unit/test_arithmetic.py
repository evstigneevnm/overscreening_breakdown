"""Numerical behavior and precision isolation of solver-owned arithmetic."""

from decimal import Decimal
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic


class ArithmeticTests(unittest.TestCase):
    def arithmetics(self):
        return (NumpyArithmetic(), MpmathArithmetic(50))

    def test_elementary_functions_use_selected_precision(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend):
                x = arithmetic.scalar("0.375")
                one = arithmetic.scalar(1)
                tolerance = arithmetic.scalar("1e-14" if not arithmetic.use_mpmath
                                              else "1e-48")
                self.assertLess(abs(arithmetic.log(arithmetic.exp(x)) - x), tolerance)
                self.assertLess(abs(arithmetic.sinh(x) * 2
                                    - (arithmetic.exp(x) - arithmetic.exp(-x))),
                                tolerance)
                self.assertLess(abs(arithmetic.cosh(x) ** 2
                                    - arithmetic.sinh(x) ** 2 - one), tolerance)
                self.assertLess(abs(arithmetic.tanh(x)
                                    - arithmetic.sinh(x) / arithmetic.cosh(x)),
                                tolerance)
                self.assertLess(abs(arithmetic.sqrt(x) ** 2 - x), tolerance)
                self.assertLess(abs(arithmetic.sin(x) ** 2
                                    + arithmetic.cos(x) ** 2 - one), tolerance)
                self.assertLess(abs(arithmetic.sin(arithmetic.pi)), tolerance)

    def test_scalar_parameters_preserve_decimal_intent(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend):
                self.assertEqual(arithmetic.scalar(Decimal("0.125")),
                                 arithmetic.scalar("0.125"))
                self.assertEqual(arithmetic.scalar(np.int64(17)),
                                 arithmetic.scalar(17))
                self.assertEqual(arithmetic.scalar(np.float32("0.125")),
                                 arithmetic.scalar("0.125"))
        high = MpmathArithmetic(120)
        decimal = "1." + "0" * 79 + "1"
        correction = high.scalar(Decimal(decimal)) - 1
        self.assertLess(abs(correction - high.scalar("1e-80")),
                        high.scalar("1e-118"))
        self.assertEqual(NumpyArithmetic().scalar(decimal), 1)

    def test_owned_precisions_are_independent_of_global_context(self):
        previous_dps = mp.mp.dps
        low = MpmathArithmetic(30)
        high = MpmathArithmetic(120)
        self.assertIsNot(low.context, high.context)
        self.assertIsNot(low.context, mp.mp)
        low_value = low.sqrt(low.scalar(2))
        high_value = high.sqrt(high.scalar(2))
        self.assertLess(abs(high_value ** 2 - 2), high.scalar("1e-118"))
        self.assertGreater(abs(high.scalar(low_value) ** 2 - 2),
                           high.scalar("1e-40"))
        self.assertEqual(low.dps, 30)
        self.assertEqual(high.dps, 120)
        self.assertEqual(mp.mp.dps, previous_dps)
        # A later change to the global context cannot alter either owner.
        try:
            mp.mp.dps = previous_dps + 7
            self.assertEqual(high.sqrt(high.scalar(2)), high_value)
            self.assertEqual(low.sqrt(low.scalar(2)), low_value)
        finally:
            mp.mp.dps = previous_dps

    def test_foreign_context_inputs_are_rounded_without_float64_conversion(self):
        low = MpmathArithmetic(30)
        high = MpmathArithmetic(120)
        x = high.scalar(1) + high.scalar("1e-80")
        self.assertEqual(low.scalar(x), 1)
        for value in (high.scalar(x), high.vector([x])[0], high.matrix([[x]])[0, 0]):
            self.assertLess(abs((value - 1) - high.scalar("1e-80")),
                            high.scalar("1e-118"))
        self.assertEqual(low.vector([x])[0], 1)
        self.assertEqual(low.matrix(high.matrix([[x]]))[0, 0], 1)
        self.assertEqual(NumpyArithmetic().scalar(x), 1)

    def test_vectors_matrices_norms_and_linear_solutions(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend):
                matrix = arithmetic.matrix([["3", "1"], ["1", "2"]])
                rhs = arithmetic.vector(["5", "5"])
                result = arithmetic.solve(matrix, rhs)
                self.assertLess(arithmetic.norm(result - arithmetic.vector([1, 2])),
                                arithmetic.scalar("1e-14"))
                self.assertEqual(arithmetic.norm(arithmetic.vector([3, 4])), 5)
                self.assertEqual(arithmetic.norm(arithmetic.zeros(2)), 0)
                self.assertEqual(arithmetic.norm(arithmetic.zeros(2, 3)), 0)
                self.assertEqual(arithmetic.norm(arithmetic.matrix([[3, 0], [0, 4]])), 5)
                self.assertEqual(len(arithmetic.vector(iter([1, 2]))), 2)
                self.assertEqual(len(arithmetic.vector([])), 0)

    def test_linear_solve_resolves_sub_double_correction(self):
        arithmetic = MpmathArithmetic(120)
        correction = arithmetic.scalar("1e-80")
        matrix = arithmetic.matrix([[1, 1], [1, 2]])
        rhs = arithmetic.vector([2 + correction, 3 + 2 * correction])
        result = arithmetic.solve(matrix, rhs)
        self.assertLess(abs(result[0] - 1), arithmetic.scalar("1e-118"))
        self.assertLess(abs((result[1] - 1) - correction),
                        arithmetic.scalar("1e-118"))

    def test_finiteness_is_a_scalar_boolean(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend):
                for value in (arithmetic.scalar(0), arithmetic.pi):
                    self.assertIs(arithmetic.isfinite(value), True)
                for value in (arithmetic.inf, -arithmetic.inf, arithmetic.scalar("nan")):
                    self.assertIs(arithmetic.isfinite(value), False)

    def test_precision_must_be_a_positive_integer(self):
        for value in (0, -1, "50", 50.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MpmathArithmetic(value)
        self.assertEqual(MpmathArithmetic(np.int64(30)).dps, 30)

    def test_scalar_conversion_rejects_vectors_and_complex_values(self):
        for arithmetic in self.arithmetics():
            for value in ([1], np.array([1]), mp.matrix([1]), 1j,
                          np.complex128(1), mp.mpc(1), np.array(1 + 2j)):
                with self.subTest(backend=arithmetic.backend, value=value):
                    with self.assertRaises(TypeError):
                        arithmetic.scalar(value)
            for value in (1j, np.complex128(1), mp.mpc(1), np.array(1 + 2j)):
                with self.subTest(backend=arithmetic.backend, vector_entry=value):
                    with self.assertRaises(TypeError):
                        arithmetic.vector([value])
                    with self.assertRaises(TypeError):
                        arithmetic.matrix([[value]])


if __name__ == "__main__":
    unittest.main()
