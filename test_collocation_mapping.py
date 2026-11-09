"""Shared half-line geometry behind legacy collocation mapping methods."""

import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np

from rational_mapping import RationalHalfLineMap
from spectral1d import basis_functions


class CollocationMappingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)

    def configurations(self):
        yield False, 15
        for precision in (15, 30, 50, 100, 120):
            yield True, precision

    def basis(self, use_mpmath=False, precision=50, L=2):
        return basis_functions(8, L=L, use_mpmath=use_mpmath, prec=precision)

    def scalar(self, use_mpmath):
        return mp.mpf if use_mpmath else np.float64

    def tolerance(self, use_mpmath, precision):
        scalar = self.scalar(use_mpmath)
        return scalar("2e-{}".format(max(12, precision-4) if use_mpmath else 12))

    def assert_native(self, value, use_mpmath):
        self.assertIsInstance(value, mp.mpf if use_mpmath else np.float64)

    def test_scalar_and_zero_dimensional_inputs_return_native_scalars(self):
        for use_mpmath, precision in self.configurations():
            with self.subTest(mpmath=use_mpmath, dps=precision):
                basis = self.basis(use_mpmath, precision)
                scalar = self.scalar(use_mpmath)
                pi = scalar(mp.pi if use_mpmath else np.pi)
                for value in (scalar(2), np.array(scalar(2))):
                    result = basis.from_infinity_map_to_t(value)
                    self.assert_native(result, use_mpmath)
                    self.assertEqual(result, pi/2)
                for value in (pi/2, np.array(pi/2)):
                    result = basis.from_t_map_to_infinity(value)
                    self.assert_native(result, use_mpmath)
                    self.assertEqual(result, 2)

    def test_nd_arrays_lists_and_repeated_endpoints_preserve_shape_and_inputs(self):
        for use_mpmath, precision in self.configurations():
            with self.subTest(mpmath=use_mpmath, dps=precision):
                basis = self.basis(use_mpmath, precision)
                scalar = self.scalar(use_mpmath)
                dtype = object if use_mpmath else np.float64
                pi = scalar(mp.pi if use_mpmath else np.pi)
                points = np.array([[scalar(0), scalar(2), scalar("inf")],
                                   [scalar("inf"), scalar(0), scalar(2)]], dtype=dtype)
                before = points.copy()
                expected = np.array([[pi, pi/2, scalar(0)],
                                     [scalar(0), pi, pi/2]], dtype=dtype)
                for inputs in (points, points.tolist(), points[:, ::-1][:, ::-1]):
                    angles = basis.from_infinity_map_to_t(inputs)
                    self.assertEqual(angles.shape, points.shape)
                    self.assertEqual(angles.dtype, np.dtype(dtype))
                    np.testing.assert_array_equal(angles, expected)
                    for value in angles.flat:
                        self.assert_native(value, use_mpmath)
                    angle_copy = angles.copy()
                    roundtrip = basis.from_t_map_to_infinity(angles)
                    self.assertEqual(roundtrip.shape, points.shape)
                    self.assertEqual(roundtrip.dtype, np.dtype(dtype))
                    np.testing.assert_array_equal(roundtrip, before)
                    np.testing.assert_array_equal(angles, angle_copy)
                np.testing.assert_array_equal(points, before)

    def test_empty_arrays_preserve_shape_and_backend_dtype(self):
        for use_mpmath in (False, True):
            basis = self.basis(use_mpmath)
            for inputs in ([], np.empty((0,)), np.empty((2, 0, 3))):
                for mapping in (basis.from_infinity_map_to_t, basis.from_t_map_to_infinity):
                    with self.subTest(mpmath=use_mpmath, shape=np.shape(inputs),
                                      mapping=mapping.__name__):
                        result = mapping(inputs)
                        self.assertEqual(result.shape, np.shape(inputs))
                        self.assertEqual(result.size, 0)
                        self.assertEqual(result.dtype, np.dtype(object if use_mpmath else np.float64))

    def test_interior_angles_match_independent_half_angle_geometry(self):
        for use_mpmath, precision in self.configurations():
            with self.subTest(mpmath=use_mpmath, dps=precision):
                basis = self.basis(use_mpmath, precision, L=3)
                scalar = self.scalar(use_mpmath)
                reference = mp.mp if use_mpmath else np
                atan = reference.atan if use_mpmath else reference.arctan
                for coordinate in ("0.125", "1", "7", "19.5"):
                    x = scalar(coordinate)
                    expected = 2*atan(reference.sqrt(scalar(3)/x))
                    observed = basis.from_infinity_map_to_t(x)
                    self.assertLess(abs(observed-expected), self.tolerance(use_mpmath, precision))
                    back = basis.from_t_map_to_infinity(observed)
                    self.assertLess(abs(back/x-1), self.tolerance(use_mpmath, precision))

    def test_near_endpoint_angles_preserve_resolvable_physical_scales(self):
        for use_mpmath, precision in self.configurations():
            with self.subTest(mpmath=use_mpmath, dps=precision):
                basis = self.basis(use_mpmath, precision, L=3)
                scalar = self.scalar(use_mpmath)
                reference = mp.mp if use_mpmath else np
                pi = scalar(reference.pi)
                small = scalar("1e-{}".format(max(4, precision//3))) if use_mpmath else scalar("1e-20")
                wall_distance = small if use_mpmath else scalar("1e-7")
                for angle in (small, pi-wall_distance):
                    expected = scalar(3)*(reference.cos(angle/2)/reference.sin(angle/2))**2
                    observed = basis.from_t_map_to_infinity(angle)
                    self.assertGreater(observed, 0)
                    self.assertTrue(reference.isfinite(observed))
                    self.assertLess(abs(observed/expected-1),
                                    self.tolerance(use_mpmath, precision))
                    restored = basis.from_infinity_map_to_t(observed)
                    self.assertLess(abs(restored-angle), self.tolerance(use_mpmath, precision))
                # This tail angle remains representable even after the
                # corresponding rational y coordinate has rounded to one.
                tiny_angle = basis.from_infinity_map_to_t(scalar("1e100"))
                self.assertGreater(tiny_angle, 0)
                self.assertLess(abs(basis.from_t_map_to_infinity(tiny_angle)
                                    / scalar("1e100")-1),
                                self.tolerance(use_mpmath, precision))

    def test_mapping_length_changes_and_decimal_precision_rebind_geometry(self):
        for use_mpmath in (False, True):
            basis = self.basis(use_mpmath, L=2)
            scalar = self.scalar(use_mpmath)
            pi = scalar(mp.pi if use_mpmath else np.pi)
            self.assertEqual(basis.from_t_map_to_infinity(pi/2), 2)
            basis.set_mapping_parameter("3.125")
            self.assertEqual(basis.from_t_map_to_infinity(pi/2), scalar("3.125"))
            self.assertEqual(basis.from_infinity_map_to_t(scalar("3.125")), pi/2)
        decimal = "1."+"0"*79+"1"
        basis = self.basis(True, 30, L=decimal)
        self.assertEqual(basis.from_t_map_to_infinity(mp.pi/2), 1)
        mp.mp.dps = 120
        observed = basis.from_t_map_to_infinity(mp.pi/2)
        self.assertLess(abs((observed-1)-mp.mpf("1e-80")), mp.mpf("1e-118"))

    def test_cached_geometry_reuses_owner_and_refreshes_exact_bit_precision(self):
        basis = self.basis(True, 30)
        with patch("spectral1d.RationalHalfLineMap", wraps=RationalHalfLineMap) as constructor:
            basis.from_infinity_map_to_t(1)
            self.assertEqual(constructor.call_count, 1)
            first_arithmetic = constructor.call_args.args[0]
            self.assertEqual(first_arithmetic.context.prec, mp.mp.prec)
            basis.from_infinity_map_to_t([1, 2])
            basis.from_t_map_to_infinity(mp.pi/3)
            self.assertEqual(constructor.call_count, 1)
            old_dps = mp.mp.dps
            mp.mp.prec += 1
            self.assertEqual(mp.mp.dps, old_dps)
            result = basis.from_infinity_map_to_t(1)
            self.assertEqual(constructor.call_count, 2)
            second_arithmetic = constructor.call_args.args[0]
            self.assertIsNot(first_arithmetic, second_arithmetic)
            self.assertEqual(second_arithmetic.context.prec, mp.mp.prec)
            self.assertLess(abs(result-2*mp.atan(mp.sqrt(2))), mp.mpf("1e-28"))

    def test_switching_backends_refreshes_basis_functions_and_keeps_legacy_reference(self):
        basis = self.basis(False)
        self.assertIs(basis.get_mpmath_ref(), np)
        self.assertFalse(basis.is_using_mpmath())
        basis.from_infinity_map_to_t(1)
        basis.use_mpmath(100)
        self.assertIs(basis.get_mpmath_ref(), mp)
        self.assertTrue(basis.is_using_mpmath())
        self.assertEqual(mp.mp.dps, 100)
        self.assertEqual(basis._domain[1], mp.pi)
        angle = mp.mpf("0.375")
        self.assert_native(basis.from_infinity_map_to_t(mp.mpf(1)), True)
        for method, expected in ((basis.psi, mp.cos(3*angle)),
                                 (basis.dpsi, -3*mp.sin(3*angle)),
                                 (basis.ddpsi, -9*mp.cos(3*angle)),
                                 (basis.dddpsi, 27*mp.sin(3*angle)),
                                 (basis.ddddpsi, 81*mp.cos(3*angle))):
            value = method(3, angle)[0]
            self.assert_native(value, True)
            self.assertLess(abs(value-expected), mp.mpf("1e-95"))
        basis.set_domain([0, 4])
        segment_point = np.array([mp.mpf("1.25")], dtype=object)
        segment_angle = basis.from_segment_map_to_t(segment_point.copy())
        self.assert_native(segment_angle[0], True)
        self.assertLess(abs(segment_angle[0]-mp.acos(mp.mpf("-0.375"))), mp.mpf("1e-95"))
        basis.use_numpy()
        self.assertIs(basis.get_mpmath_ref(), np)
        self.assertFalse(basis.is_using_mpmath())
        self.assertEqual(basis._domain[1], np.pi)
        self.assert_native(basis.from_infinity_map_to_t(1), False)
        self.assertEqual(basis.psi(3, np.float64("0.375")).dtype, np.dtype("float64"))
        segment_angle = basis.from_segment_map_to_t(np.array([1.25]))
        self.assertEqual(segment_angle.dtype, np.dtype("float64"))
        self.assertLess(abs(segment_angle[0]-np.arccos(-0.375)), 1e-14)
        basis.use_mpmath(50)
        self.assert_native(basis.from_infinity_map_to_t(1), True)
        self.assertEqual(basis._domain[1], mp.pi)

    def test_physical_basis_derivatives_keep_their_existing_interpretation(self):
        for use_mpmath, precision in self.configurations():
            with self.subTest(mpmath=use_mpmath, dps=precision):
                basis = self.basis(use_mpmath, precision, L=2)
                scalar = self.scalar(use_mpmath)
                x = scalar("0.625")
                angle = basis.from_infinity_map_to_t(x)
                oracle = mp.mp.clone()
                oracle.dps = precision+20
                point = oracle.mpf("0.625")
                for order, derivative in enumerate((basis.map_derivative1_inf,
                                                     basis.map_derivative2_inf,
                                                     basis.map_derivative3_inf,
                                                     basis.map_derivative4_inf), 1):
                    expected = scalar(oracle.diff(
                        lambda z: oracle.chebyt(3, (z-2)/(z+2)), point, order))
                    observed = np.asarray(derivative(3, angle)).item()
                    self.assertLess(abs(observed-expected),
                                    self.tolerance(use_mpmath, precision)*max(scalar(1), abs(expected)))

    def test_existing_coefficients_represent_the_same_chebyshev_polynomial(self):
        for use_mpmath, precision in self.configurations():
            with self.subTest(mpmath=use_mpmath, dps=precision):
                basis = self.basis(use_mpmath, precision, L=2)
                scalar = self.scalar(use_mpmath)
                coefficients = [scalar(value) for value in ("1", "-0.5", "0.25", "-0.125")]
                modes = np.arange(len(coefficients))
                for x in (scalar(0), scalar("0.375"), scalar(2), scalar(7), scalar("inf")):
                    reference = scalar(1) if x == scalar("inf") else (x-2)/(x+2)
                    expected = np.polynomial.chebyshev.chebval(reference, coefficients)
                    angle = basis.from_infinity_map_to_t(x)
                    observed = sum((coefficient*value for coefficient, value in
                                    zip(coefficients, basis.psi(modes, angle))), scalar(0))
                    self.assertLess(abs(observed-expected), self.tolerance(use_mpmath, precision))

    def test_finite_segment_geometry_and_derivatives_remain_compatible(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                basis = self.basis(use_mpmath)
                basis.set_domain([-4, 7])
                scalar = self.scalar(use_mpmath)
                dtype = object if use_mpmath else np.float64
                points = np.array([scalar(-4), scalar("0.5"), scalar(7)], dtype=dtype)
                angles = basis.from_segment_map_to_t(points.copy())
                restored = basis.from_t_map_to_segment(angles)
                tolerance = self.tolerance(use_mpmath, 50)
                for actual, expected in zip(restored, points):
                    self.assertLess(abs(actual-expected), tolerance)
                middle = angles[1]
                y = (2*points[1]-3)/11
                first = np.asarray(basis.map_derivative1_segment(2, middle)).item()
                second = np.asarray(basis.map_derivative2_segment(2, middle)).item()
                self.assertLess(abs(first-8*y/11), tolerance)
                self.assertLess(abs(second-scalar(16)/121), tolerance)
                outside = np.array([scalar(-9), scalar(12)], dtype=dtype)
                clipped = basis.from_segment_map_to_t(outside)
                self.assertEqual(clipped[0], scalar(mp.pi if use_mpmath else np.pi))
                self.assertEqual(clipped[1], 0)

    def test_invalid_half_line_coordinates_are_rejected(self):
        for use_mpmath in (False, True):
            basis = self.basis(use_mpmath)
            scalar = self.scalar(use_mpmath)
            pi = scalar(mp.pi if use_mpmath else np.pi)
            for invalid in (-1, "nan", "-inf"):
                with self.subTest(mpmath=use_mpmath, x=invalid):
                    with self.assertRaises(ValueError):
                        basis.from_infinity_map_to_t(invalid)
            for invalid in (-1, pi+1, "nan", "inf", "-inf"):
                with self.subTest(mpmath=use_mpmath, t=invalid):
                    with self.assertRaises(ValueError):
                        basis.from_t_map_to_infinity(invalid)
            for mapping in (basis.from_infinity_map_to_t, basis.from_t_map_to_infinity):
                with self.assertRaises(TypeError):
                    mapping(1j)


if __name__ == "__main__":
    unittest.main()
