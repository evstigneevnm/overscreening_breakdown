"""Independent endpoint, derivative and arithmetic contracts of finite spaces."""

from dataclasses import FrozenInstanceError
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.galerkin.finite_basis import FiniteGalerkinBasis
from overscreening_breakdown.quadrature import QuadratureFactory


class FiniteGalerkinBasisTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for digits in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(digits)

    def tolerance(self, a):
        return a.scalar("1e-11" if not a.use_mpmath else "1e-{}".format(max(10, a.dps-4)))

    def test_dimensions_endpoint_spaces_and_free_curvature(self):
        for a in self.arithmetics():
            for N in (2, 4, 8):
                with self.subTest(backend=a.backend, digits=a.dps, N=N):
                    b = FiniteGalerkinBasis(N, a, domain=(0, "3"))
                    self.assertEqual((b.n_u, b.n_w, b.n_first_tests, b.n_second_tests), (N-1, N+1, N, N))
                    self.assertEqual(b.unknown_count, 2*N)
                    self.assertTrue(all(value == 0 for value in b.v(-1)+b.v(1)+b.phi(-1)+b.psi(1)))
                    self.assertEqual(b.q(1), (a.scalar(1),)*(N+1))
                    self.assertEqual(b.phi(1), (a.scalar(2),)*N)
                    self.assertEqual(b.psi(-1), tuple(a.scalar(2*(-1)**k) for k in range(N)))
                    coefficients = a.zeros(2*N)
                    coefficients[b.n_u] = a.scalar(3)
                    u, w = b.evaluate_fields(coefficients, 1, "2", "5")
                    self.assertEqual((u, w), (a.scalar(5), a.scalar(3)))

    def test_physical_derivatives_match_independent_polynomial_differences(self):
        for a in self.arithmetics():
            with self.subTest(backend=a.backend, digits=a.dps):
                b = FiniteGalerkinBasis(4, a, domain=(0, "3"))
                y, h = a.scalar("0.25"), a.scalar("0.125")
                for name in ("v", "q", "phi", "psi"):
                    callback, dy = getattr(b, name), getattr(b, name+"_dy")(y)
                    samples = [callback(y+shift*h) for shift in (-2, -1, 1, 2)]
                    for index, value in enumerate(dy):
                        difference = (samples[0][index]-8*samples[1][index]
                                      +8*samples[2][index]-samples[3][index])/(12*h)
                        self.assertLess(abs(value-difference), self.tolerance(a))
                        self.assertLess(abs(getattr(b, name+"_dx")(y)[index]-2*value/3), self.tolerance(a))

    def test_two_endpoint_lift_and_native_node_tables(self):
        for a in self.arithmetics():
            with self.subTest(backend=a.backend, digits=a.dps):
                b = FiniteGalerkinBasis(4, a, domain=(0, "2.5"))
                lift = b.lift("1.25", "-0.75")
                self.assertEqual(lift.at(0), a.scalar("1.25"))
                self.assertEqual(lift.at("2.5"), a.scalar("-0.75"))
                self.assertEqual(lift.derivative_at("1.2"), a.scalar("-0.8"))
                plan = QuadratureFactory(a, domain=(0, "2.5"), interfaces=("0.75",)).plan(6)
                table = b.tabulate(plan)
                self.assertEqual(len(table.v), 12)
                for values, width in ((table.v, 3), (table.q, 5), (table.phi, 4), (table.psi, 4)):
                    self.assertTrue(all(len(row) == width for row in values))
                    for row in values:
                        self.assertTrue(all(isinstance(value, type(a.scalar(0))) for value in row))
                self.assertIs(table.mapping, b.mapping)
                table.check_precision()

    def test_precision_isolation_geometry_and_immutability(self):
        original = mp.mp.prec
        a = MpmathArithmetic(50)
        b = FiniteGalerkinBasis(4, a, domain=(0, "2"))
        p = QuadratureFactory(a, domain=(0, "2")).plan(5)
        table = b.tabulate(p)
        self.assertEqual(mp.mp.prec, original)
        for invalid in (QuadratureFactory(MpmathArithmetic(50), domain=(0, "2")).plan(5),
                        QuadratureFactory(a, domain=(0, "3")).plan(5), QuadratureFactory(a).plan(5)):
            with self.assertRaises(ValueError):
                b.tabulate(invalid)
        with self.assertRaises(FrozenInstanceError):
            b.N = 8
        a.context.prec += 1
        for callback in (lambda: b.v(0), lambda: table.check_precision(), lambda: b.lift(0, 1)):
            with self.assertRaises(RuntimeError):
                callback()

    def test_invalid_dimensions_domains_and_coefficients(self):
        a = NumpyArithmetic()
        for N in (True, 1, "4", 2.5, np.bool_(False)):
            with self.subTest(N=N), self.assertRaises(ValueError):
                FiniteGalerkinBasis(N, a, domain=(0, 2))
        with self.assertRaises(ValueError):
            FiniteGalerkinBasis(4, a, domain=(1, 2))
        b = FiniteGalerkinBasis(4, a, domain=(0, 2))
        for coefficients in ([0]*7, [0]*7+[np.inf]):
            with self.assertRaises(ValueError):
                b.split_coefficients(coefficients)
        with self.assertRaises(ValueError):
            b.lift(0, "nan")
