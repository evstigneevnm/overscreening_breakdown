"""Finite collocation delegates geometry without changing its polynomial basis."""

import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np

from overscreening_breakdown.affine_mapping import AffineIntervalMap
from overscreening_breakdown.collocation.spectral1d import basis_functions


class CollocationAffineMappingTests(unittest.TestCase):
    def setUp(self):
        original = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original)

    def configurations(self):
        yield False, 15
        for digits in (15, 30, 50, 100, 120):
            yield True, digits

    def test_nodes_keep_the_existing_cosine_order_and_coefficient_meaning(self):
        for use_mpmath, digits in self.configurations():
            with self.subTest(mpmath=use_mpmath, digits=digits):
                basis = basis_functions(8, use_mpmath=use_mpmath, prec=digits,
                                        calc_domain=[-4, 7])
                scalar = mp.mpf if use_mpmath else np.float64
                context = mp if use_mpmath else np
                angles = basis.circlemap_in_t(np.arange(8))
                observed = basis.from_t_map_to_segment(angles)
                expected = np.asarray([(context.cos(t)*11+3)/2 for t in angles],
                                      dtype=object if use_mpmath else np.float64)
                np.testing.assert_array_equal(observed, expected)
                self.assertTrue(all(left > right for left, right in zip(observed, observed[1:])))
                point = np.array([scalar(".625")], dtype=object if use_mpmath else np.float64)
                angle = basis.from_segment_map_to_t(point)
                y = (2*point[0]-3)/11
                coefficients = tuple(scalar(value) for value in ("1", "-.5", ".25", "-.125"))
                actual = sum((value*basis.psi(k, angle)[0]
                              for k, value in enumerate(coefficients)), scalar(0))
                expected_polynomial = (coefficients[0]+coefficients[1]*y
                                       +coefficients[2]*(2*y*y-1)
                                       +coefficients[3]*(4*y**3-3*y))
                tolerance = scalar("1e-{}".format(max(12, digits-4)))
                self.assertLess(abs(actual-expected_polynomial), tolerance)

    def test_physical_derivatives_through_four_keep_both_endpoint_signs(self):
        for use_mpmath, digits in self.configurations():
            scalar = mp.mpf if use_mpmath else np.float64
            # The physical solver binds domain endpoints in its native
            # arithmetic before the legacy strong derivative formulas run.
            basis = basis_functions(8, use_mpmath=use_mpmath, prec=digits,
                                    calc_domain=[scalar(-4), scalar(7)])
            oracle = mp.mp.clone()
            oracle.dps = digits+20
            tolerance = scalar("1e-{}".format(max(11, digits-4)))
            modes = (0, 2, 5)
            for mode in modes:
                for raw in ("-4", ".625", "7"):
                    point = oracle.mpf(raw)
                    for order in range(5):
                        expected = scalar(oracle.diff(
                            lambda x: oracle.chebyt(mode, (2*x-3)/11), point, order))
                        if order == 0:
                            angle = basis.from_segment_map_to_t(np.array([scalar(raw)], dtype=object))
                            actual = basis.psi(mode, angle)[0]
                        elif raw in ("-4", "7"):
                            endpoint = "pi" if raw == "-4" else "0"
                            actual = getattr(basis, "map_derivative{}_segment_at_{}".format(order, endpoint))(
                                scalar(mode))
                        else:
                            angle = basis.from_segment_map_to_t(np.array([scalar(raw)], dtype=object))
                            actual = np.asarray(getattr(basis, "map_derivative{}_segment".format(order))(
                                mode, angle)).item()
                        self.assertLess(abs(actual-expected), tolerance*max(scalar(1), abs(expected)))

    def test_legacy_array_clipping_mutates_only_outside_values(self):
        for use_mpmath in (False, True):
            basis = basis_functions(8, use_mpmath=use_mpmath, prec=50, calc_domain=[-4, 7])
            scalar = mp.mpf if use_mpmath else np.float64
            values = np.array([scalar(-9), scalar(".625"), scalar(12)],
                              dtype=object if use_mpmath else np.float64)
            angles = basis.from_segment_map_to_t(values)
            self.assertEqual(tuple(values), (scalar(-4), scalar(".625"), scalar(7)))
            self.assertEqual(angles[0], scalar(mp.pi if use_mpmath else np.pi))
            self.assertEqual(angles[-1], 0)

    def test_mapping_cache_tracks_raw_domain_backend_and_exact_bit_precision(self):
        raw_right = "1."+"0"*79+"1"
        basis = basis_functions(8, use_mpmath=True, prec=30, calc_domain=[0, raw_right])
        with patch("overscreening_breakdown.collocation.spectral1d.AffineIntervalMap",
                   wraps=AffineIntervalMap) as constructor:
            basis.from_t_map_to_segment(mp.pi/2)
            basis.from_segment_map_to_t(np.array([mp.mpf(".25")], dtype=object))
            self.assertEqual(constructor.call_count, 1)
            self.assertEqual(constructor.call_args.args[0].context.prec, mp.mp.prec)
            previous_digits = mp.mp.dps
            mp.mp.prec += 1
            self.assertEqual(mp.mp.dps, previous_digits)
            basis.from_t_map_to_segment(mp.pi/2)
            self.assertEqual(constructor.call_count, 2)
            mp.mp.dps = 120
            midpoint = np.asarray(basis.from_t_map_to_segment(mp.pi/2)).item()
            self.assertGreater(midpoint, mp.mpf(".5"))
            self.assertEqual(constructor.call_count, 3)
            basis.set_domain([0, 4])
            self.assertEqual(np.asarray(basis.from_t_map_to_segment(mp.pi/2)).item(), 2)
            self.assertEqual(constructor.call_count, 4)
            basis.use_numpy()
            self.assertEqual(np.asarray(basis.from_t_map_to_segment(np.pi/2)).item(), 2)
            self.assertEqual(constructor.call_count, 5)
