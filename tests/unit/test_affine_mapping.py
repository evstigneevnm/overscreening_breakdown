"""Finite geometry, Chebyshev orientation, and owned native precision."""

from dataclasses import FrozenInstanceError
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.affine_mapping import AffineIntervalMap
from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic


class AffineMappingTests(unittest.TestCase):
    def arithmetics(self):
        return (NumpyArithmetic(),)+tuple(MpmathArithmetic(dps) for dps in (15, 30, 50, 100, 120))

    def tolerance(self, arithmetic):
        return arithmetic.scalar("1e-{}".format(max(12, arithmetic.dps-4)))

    def test_endpoints_angles_and_physical_jacobians_in_all_precisions(self):
        for a in self.arithmetics():
            for domain in (("0", "50"), ("-4", "7")):
                with self.subTest(backend=a.backend, digits=a.dps, domain=domain):
                    mapping = AffineIntervalMap(a, domain)
                    left, right = (a.scalar(value) for value in domain)
                    pi = a.scalar(a.pi)
                    self.assertEqual(mapping.mapping_type, "affine_interval")
                    self.assertEqual(mapping.domain, (left, right))
                    self.assertEqual(mapping.raw_domain, domain)
                    self.assertEqual(mapping.to_reference(left), -1)
                    self.assertEqual(mapping.to_reference(right), 1)
                    self.assertEqual(mapping.to_physical(-1), left)
                    self.assertEqual(mapping.to_physical(1), right)
                    self.assertEqual(mapping.to_angle(left), pi)
                    self.assertEqual(mapping.to_angle(right), 0)
                    self.assertEqual(mapping.from_angle(0), right)
                    self.assertEqual(mapping.from_angle(pi), left)
                    for raw in ("-1", "-.375", "0", ".625", "1"):
                        y = a.scalar(raw)
                        x = mapping.to_physical(y)
                        self.assertIsInstance(x, type(a.scalar(0)))
                        self.assertLess(abs(mapping.to_reference(x)-y), self.tolerance(a))
                        self.assertLess(abs(mapping.from_angle(mapping.to_angle(x))-x),
                                        self.tolerance(a)*mapping.length)
                        self.assertEqual(mapping.jacobian(y), mapping.length/2)
                        self.assertEqual(mapping.inverse_jacobian(y), 2/mapping.length)

    def test_affine_chain_rule_for_chebyshev_derivatives_zero_through_four(self):
        for a in self.arithmetics():
            mapping = AffineIntervalMap(a, ("-4", "7"))
            oracle = mp.mp.clone()
            oracle.dps = a.dps+20
            # T_4(y)=8y^4-8y^2+1 gives nonzero independent derivatives
            # through order four at both endpoints and at interior points.
            for raw in ("-4", "-.25", "7"):
                x = a.scalar(raw)
                y = mapping.to_reference(x)
                derivatives = (8*y**4-8*y*y+1, 32*y**3-16*y,
                               96*y*y-16, 192*y, a.scalar(192))
                for order, derivative in enumerate(derivatives):
                    expected = a.scalar(oracle.diff(
                        lambda point: oracle.chebyt(4, (2*point-3)/11), oracle.mpf(raw), order))
                    observed = derivative*mapping.inverse_jacobian(y)**order
                    self.assertLess(abs(observed-expected), self.tolerance(a)*max(a.scalar(1), abs(expected)))

    def test_native_parameters_and_independent_contexts_keep_decimal_corrections(self):
        original = mp.mp.prec
        raw_right = "1."+"0"*79+"1"
        low_owner, high_owner = MpmathArithmetic(30), MpmathArithmetic(120)
        raw = ["0", raw_right]
        low = AffineIntervalMap(low_owner, raw)
        high = AffineIntervalMap(high_owner, raw)
        raw[1] = "10"
        self.assertEqual(low.right, 1)
        self.assertGreater(high.right, 1)
        self.assertEqual(high.raw_domain, ("0", raw_right))
        self.assertEqual(mp.mp.prec, original)
        value = high.to_physical(".5")
        low_owner.context.dps = 15
        self.assertEqual(high.to_physical(".5"), value)
        with self.assertRaises(FrozenInstanceError):
            high.length = 2
        high_owner.context.prec += 1
        for operation in (high.check_precision, lambda: high.to_reference(0),
                          lambda: high.to_physical(0), lambda: high.jacobian(0),
                          lambda: high.inverse_jacobian(0), lambda: high.to_angle(0),
                          lambda: high.from_angle(0)):
            with self.assertRaisesRegex(RuntimeError, "precision changed"):
                operation()
        rebuilt = AffineIntervalMap(high_owner, high.raw_domain)
        self.assertGreater(rebuilt.right, 1)

    def test_near_endpoint_angles_and_large_translated_intervals_stay_finite(self):
        for a in self.arithmetics():
            mapping = AffineIntervalMap(a, ("0", "2"))
            x = a.scalar("1e-{}".format(a.dps+2))
            self.assertEqual(mapping.to_reference(x), -1)
            angle = mapping.to_angle(x)
            self.assertLess(angle, a.scalar(a.pi))
            self.assertGreater(mapping.from_angle(angle), 0)
        a = NumpyArithmetic()
        mapping = AffineIntervalMap(a, ("1e308", "1.1e308"))
        with np.errstate(over="raise", invalid="raise"):
            self.assertTrue(a.isfinite(mapping.to_reference("1.05e308")))
            self.assertTrue(a.isfinite(mapping.to_physical(0)))
            self.assertTrue(a.isfinite(mapping.from_angle(a.scalar(a.pi)/2)))

    def test_invalid_geometry_and_coordinates_are_rejected_without_clipping(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            for domain in ((), (0,), (1, 1), (2, 1), ("nan", 1), (0, "inf")):
                with self.subTest(domain=domain), self.assertRaises(ValueError):
                    AffineIntervalMap(a, domain)
            mapping = AffineIntervalMap(a, (0, 2))
            for x in (-1, 3, "nan", "inf", "-inf"):
                with self.assertRaises(ValueError):
                    mapping.to_reference(x)
                with self.assertRaises(ValueError):
                    mapping.to_angle(x)
            for y in (-2, 2, "nan", "inf"):
                with self.assertRaises(ValueError):
                    mapping.to_physical(y)
            for t in (-1, 4, "nan", "inf"):
                with self.assertRaises(ValueError):
                    mapping.from_angle(t)
            with self.assertRaises(TypeError):
                mapping.to_reference(1j)
