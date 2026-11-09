"""Physical half-line mapping and native arithmetic precision checks."""

from dataclasses import FrozenInstanceError
import unittest

import mpmath as mp
import numpy as np

from arithmetic import MpmathArithmetic, NumpyArithmetic
from rational_mapping import RationalHalfLineMap, arithmetic_signature


class RationalMappingTests(unittest.TestCase):
    def arithmetics(self):
        return (NumpyArithmetic(),) + tuple(
            MpmathArithmetic(dps) for dps in (15, 30, 50, 100, 120)
        )

    def test_endpoints_orientation_and_inverse_in_each_precision(self):
        for arithmetic in self.arithmetics():
            mapping = RationalHalfLineMap(arithmetic, L="2.5")
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                self.assertEqual(mapping.to_reference(0), -1)
                self.assertEqual(mapping.to_reference(arithmetic.inf), 1)
                self.assertEqual(mapping.to_physical(-1), 0)
                self.assertEqual(mapping.to_physical(1), arithmetic.inf)
                self.assertEqual(mapping.to_reference(mapping.L), 0)
                tolerance = arithmetic.scalar("1e-{}".format(max(12, int(arithmetic.dps)-3)))
                for text in ("0.125", "1", "3", "10"):
                    x = arithmetic.scalar(text)
                    self.assertLess(abs(mapping.to_physical(mapping.to_reference(x))-x),
                                    tolerance*max(arithmetic.scalar(1), x))

    def test_physical_jacobians_and_chain_rule(self):
        for arithmetic in self.arithmetics():
            mapping = RationalHalfLineMap(arithmetic, L=3)
            tolerance = arithmetic.scalar("1e-{}".format(max(12, int(arithmetic.dps)-3)))
            for text in ("-1", "-0.4", "0", "0.7"):
                y = arithmetic.scalar(text)
                jacobian = mapping.jacobian(y)
                self.assertGreater(jacobian, 0)
                self.assertLess(abs(jacobian*mapping.inverse_jacobian(y)-1), tolerance)
                if arithmetic.use_mpmath:
                    derivative = arithmetic.context.diff(
                        lambda z: mapping.L*(1+z)/(1-z), y,
                    )
                    self.assertLess(abs(derivative-jacobian), tolerance*jacobian)
            self.assertEqual(mapping.jacobian(1), arithmetic.inf)
            self.assertEqual(mapping.inverse_jacobian(1), 0)

    def test_large_float_arguments_avoid_intermediate_overflow(self):
        arithmetic = NumpyArithmetic()
        length = np.finfo(np.float64).max
        mapping = RationalHalfLineMap(arithmetic, length)
        with np.errstate(over="raise", invalid="raise"):
            self.assertEqual(mapping.to_reference(length), 0)
            self.assertEqual(mapping.to_physical(0), length)
            self.assertEqual(mapping.jacobian(-1), length/2)
            tiny = RationalHalfLineMap(arithmetic, L="1e-310")
            derivative = tiny.inverse_jacobian(arithmetic.scalar("0.9"))
            self.assertTrue(arithmetic.isfinite(derivative))
            self.assertLess(abs(derivative/np.float64("5e307")-1), 1e-13)

    def test_decimal_length_preserves_sub_double_correction(self):
        arithmetic = MpmathArithmetic(120)
        length = "1."+"0"*79+"1"
        mapping = RationalHalfLineMap(arithmetic, length)
        self.assertGreater(mapping.L, 1)
        self.assertEqual(mapping.raw_L, length)
        self.assertEqual(mapping.to_physical(0), arithmetic.scalar(length))
        self.assertLess(mapping.to_reference(1), 0)

    def test_precision_changes_invalidate_old_maps_and_keep_raw_parameters(self):
        arithmetic = MpmathArithmetic(30)
        mapping = RationalHalfLineMap(arithmetic, "1."+"0"*79+"1")
        signature = mapping.precision_signature
        self.assertEqual(mapping.L, 1)
        arithmetic.context.dps = 120
        self.assertNotEqual(arithmetic_signature(arithmetic), signature)
        for operation in (mapping.check_precision, lambda: mapping.to_physical(0),
                          lambda: mapping.to_reference(1), lambda: mapping.jacobian(0)):
            with self.assertRaisesRegex(RuntimeError, "precision changed"):
                operation()
        rebuilt = RationalHalfLineMap(arithmetic, mapping.raw_L)
        self.assertGreater(rebuilt.L, 1)
        # Exact bit precision matters even when decimal dps appears unchanged.
        arithmetic.context.prec += 1
        with self.assertRaises(RuntimeError):
            rebuilt.check_precision()

    def test_separate_contexts_do_not_change_global_precision(self):
        precision = mp.mp.prec
        low = RationalHalfLineMap(MpmathArithmetic(30), "0.1")
        high = RationalHalfLineMap(MpmathArithmetic(120), "0.1")
        value = high.to_physical("0.2")
        low.arithmetic.context.dps = 15
        self.assertEqual(high.to_physical("0.2"), value)
        self.assertEqual(mp.mp.prec, precision)

    def test_invalid_lengths_coordinates_and_complex_values_are_rejected(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            for value in (0, -1, "nan", "inf"):
                with self.assertRaises(ValueError):
                    RationalHalfLineMap(arithmetic, value)
            mapping = RationalHalfLineMap(arithmetic)
            for value in ("nan", "inf", "-inf", "-1.1", "1.1"):
                with self.assertRaises(ValueError):
                    mapping.to_physical(value)
            for value in ("nan", "-inf", -1):
                with self.assertRaises(ValueError):
                    mapping.to_reference(value)
            for value in ([0], 1j):
                with self.assertRaises(TypeError):
                    mapping.to_reference(value)
                with self.assertRaises(TypeError):
                    mapping.to_physical(value)

    def test_mapping_metadata_is_frozen(self):
        mapping = RationalHalfLineMap(NumpyArithmetic())
        with self.assertRaises(FrozenInstanceError):
            mapping.L = 2


if __name__ == "__main__":
    unittest.main()
