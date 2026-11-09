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

    def test_angle_endpoints_center_and_roundtrips_in_each_precision(self):
        for arithmetic in self.arithmetics():
            mapping = RationalHalfLineMap(arithmetic, L="2.5")
            pi = arithmetic.scalar(arithmetic.pi)
            tolerance = arithmetic.scalar("1e-{}".format(max(12, int(arithmetic.dps)-3)))
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                self.assertEqual(mapping.to_angle(0), pi)
                self.assertEqual(mapping.to_angle(arithmetic.inf), 0)
                self.assertEqual(mapping.to_angle(mapping.L), pi/2)
                self.assertEqual(mapping.from_angle(0), arithmetic.inf)
                self.assertEqual(mapping.from_angle(pi), 0)
                self.assertEqual(mapping.from_angle(pi/2), mapping.L)
                for text in ("0.03125", "0.25", "1", "4", "100"):
                    x = arithmetic.scalar(text)
                    angle = mapping.to_angle(x)
                    self.assertIsInstance(angle, type(arithmetic.scalar(0)))
                    self.assertGreater(angle, 0)
                    self.assertLess(angle, pi)
                    self.assertLess(abs(mapping.from_angle(angle)-x), tolerance*x)

    def test_angle_conversions_retain_points_when_reference_rounds_to_endpoints(self):
        for arithmetic in self.arithmetics():
            mapping = RationalHalfLineMap(arithmetic, L="2.5")
            pi = arithmetic.scalar(arithmetic.pi)
            eps = (arithmetic.context.eps if arithmetic.use_mpmath else
                   arithmetic.scalar(np.finfo(np.float64).eps))
            tolerance = 1000*arithmetic.sqrt(eps)
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                near_wall = mapping.L*arithmetic.scalar(
                    "1e-{}".format(int(arithmetic.dps)+2)
                )
                self.assertEqual(mapping.to_reference(near_wall), -1)
                wall_angle = mapping.to_angle(near_wall)
                self.assertLess(wall_angle, pi)
                self.assertGreater(mapping.from_angle(wall_angle), 0)
                self.assertLess(abs(mapping.from_angle(wall_angle)/near_wall-1), tolerance)
                near_infinity = arithmetic.scalar(
                    "1e-{}".format(max(10, int(arithmetic.dps)*7//10))
                )
                far_x = mapping.from_angle(near_infinity)
                self.assertEqual(mapping.to_reference(far_x), 1)
                self.assertGreater(mapping.to_angle(far_x), 0)
                self.assertLess(abs(mapping.to_angle(far_x)/near_infinity-1), tolerance)

    def test_angle_conversions_avoid_extreme_float_intermediates(self):
        arithmetic = NumpyArithmetic()
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            tiny_length = RationalHalfLineMap(arithmetic, L="1e-300")
            # Squaring cot(t/2) first would overflow despite this finite result.
            t = arithmetic.scalar("1e-200")
            physical = tiny_length.from_angle(t)
            self.assertLess(abs(physical/arithmetic.scalar("4e100")-1), 1e-14)
            self.assertLess(abs(tiny_length.to_angle(physical)/t-1), 1e-14)
            extreme = RationalHalfLineMap(arithmetic, L="1e-308")
            x = arithmetic.scalar("1e308")
            angle = extreme.to_angle(x)
            self.assertGreater(angle, 0)
            self.assertLess(abs(extreme.from_angle(angle)/x-1), 1e-14)
            huge = RationalHalfLineMap(arithmetic, L=np.finfo(np.float64).max)
            self.assertEqual(huge.from_angle(arithmetic.scalar(arithmetic.pi)/2), huge.L)
        with np.errstate(over="ignore"):
            with self.assertRaises(OverflowError):
                RationalHalfLineMap(arithmetic).from_angle("1e-200")
        # Smallest positive float survives input validation, but its half-angle
        # cannot be represented and must not be treated as the exact endpoint.
        with self.assertRaises(OverflowError):
            RationalHalfLineMap(arithmetic).from_angle(np.nextafter(0.0, 1.0))

    def test_high_precision_angles_preserve_sub_double_parameter_correction(self):
        arithmetic = MpmathArithmetic(120)
        correction = arithmetic.scalar("1e-80")
        mapping = RationalHalfLineMap(arithmetic, L="1."+"0"*79+"1")
        midpoint = arithmetic.scalar(arithmetic.pi)/2
        angle = mapping.to_angle(1)
        self.assertGreater(angle, midpoint)
        self.assertLess(abs((angle-midpoint)-correction/2), arithmetic.scalar("1e-118"))
        self.assertLess(abs(mapping.from_angle(angle)-1), arithmetic.scalar("1e-118"))

    def test_atan2_is_native_and_quadrant_aware(self):
        for arithmetic in self.arithmetics():
            pi = arithmetic.scalar(arithmetic.pi)
            one, zero = arithmetic.scalar(1), arithmetic.scalar(0)
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                self.assertEqual(arithmetic.atan2(one, zero), pi/2)
                self.assertEqual(arithmetic.atan2(-one, zero), -pi/2)
                self.assertEqual(arithmetic.atan2(zero, -one), pi)
                self.assertEqual(arithmetic.atan2(one, one), pi/4)
                self.assertIsInstance(arithmetic.atan2(one, one), type(zero))

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
                          lambda: mapping.to_reference(1), lambda: mapping.jacobian(0),
                          lambda: mapping.to_angle(1), lambda: mapping.from_angle(1)):
            with self.assertRaisesRegex(RuntimeError, "precision changed"):
                operation()
        rebuilt = RationalHalfLineMap(arithmetic, mapping.raw_L)
        self.assertGreater(rebuilt.L, 1)
        # Exact bit precision matters even when decimal dps appears unchanged.
        arithmetic.context.prec += 1
        for operation in (rebuilt.check_precision, lambda: rebuilt.to_angle(1),
                          lambda: rebuilt.from_angle(1)):
            with self.assertRaises(RuntimeError):
                operation()

    def test_separate_contexts_do_not_change_global_precision(self):
        precision = mp.mp.prec
        low = RationalHalfLineMap(MpmathArithmetic(30), "0.1")
        high = RationalHalfLineMap(MpmathArithmetic(120), "0.1")
        value = high.to_physical("0.2")
        angle = high.to_angle("0.2")
        angle_value = high.from_angle("0.2")
        low.arithmetic.context.dps = 15
        self.assertEqual(high.to_physical("0.2"), value)
        self.assertEqual(high.to_angle("0.2"), angle)
        self.assertEqual(high.from_angle("0.2"), angle_value)
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
                with self.assertRaises(ValueError):
                    mapping.to_angle(value)
            for value in ("nan", "inf", "-inf", -1, 4):
                with self.assertRaises(ValueError):
                    mapping.from_angle(value)
            for value in ([0], 1j):
                with self.assertRaises(TypeError):
                    mapping.to_reference(value)
                with self.assertRaises(TypeError):
                    mapping.to_physical(value)
                with self.assertRaises(TypeError):
                    mapping.to_angle(value)
                with self.assertRaises(TypeError):
                    mapping.from_angle(value)

    def test_mapping_metadata_is_frozen(self):
        mapping = RationalHalfLineMap(NumpyArithmetic())
        with self.assertRaises(FrozenInstanceError):
            mapping.L = 2


if __name__ == "__main__":
    unittest.main()
