"""Conforming spaces, physical traces and native-precision evaluations."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.galerkin.basis import AffineWallLift, GalerkinBasis
from overscreening_breakdown.quadrature import QuadratureFactory


class GalerkinBasisTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for dps in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(dps)

    def tolerance(self, arithmetic):
        return arithmetic.scalar("2e-12" if not arithmetic.use_mpmath
                                 else "2e-{}".format(max(12, arithmetic.dps-4)))

    def assert_native(self, arithmetic, values):
        for value in values:
            if arithmetic.use_mpmath:
                self.assertTrue(hasattr(value, "_mpf_"))
                self.assertNotIsInstance(value, (float, np.floating))
            else:
                self.assertIsInstance(value, np.float64)

    def test_dimensions_and_polynomial_order_validation(self):
        arithmetic = NumpyArithmetic()
        for N in (2, 4, np.int64(9)):
            basis = GalerkinBasis(N, arithmetic)
            self.assertEqual((basis.n_u, basis.n_w, basis.unknown_count),
                             (N-1, N, 2*N-1))
            self.assertEqual(len(basis.v(0)), N-1)
            self.assertEqual(len(basis.q(0)), N)
        for invalid in (True, False, 1, 0, -1, "4", 4.0):
            with self.subTest(N=invalid), self.assertRaises(ValueError):
                GalerkinBasis(invalid, arithmetic)

    def test_exact_endpoint_traces_and_physical_decay(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                basis = GalerkinBasis(8, arithmetic, L="2.75")
                self.assertEqual(basis.v(-1), (0,)*basis.n_u)
                self.assertEqual(basis.v(1), (0,)*basis.n_u)
                self.assertEqual(basis.q(1), (0,)*basis.n_w)
                self.assertEqual(basis.q(-1), tuple(2*(-1)**k for k in range(8)))
                self.assertEqual(basis.v_dx(1), (0,)*basis.n_u)
                self.assertEqual(basis.q_dx(1), (0,)*basis.n_w)
                # Reference derivatives remain finite even where a commonly
                # used T_k' formula would divide by zero.
                for y in (-1, 1):
                    self.assertTrue(all(arithmetic.isfinite(value)
                                        for value in basis.v_dy(y)+basis.q_dy(y)))
                self.assertEqual(basis.v_dy(-1), tuple(2*(-1)**k for k in range(7)))
                self.assertEqual(basis.v_dy(1), (-2,)*basis.n_u)
                self.assertEqual(basis.q_dy(1), (-1,)*basis.n_w)

    def test_recurrence_matches_independent_trigonometric_definition(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                basis = GalerkinBasis(9, arithmetic)
                theta = arithmetic.scalar("0.7")
                y = arithmetic.cos(theta)
                for k, value in enumerate(basis.q(y)):
                    expected = (1-y)*arithmetic.cos(k*theta)
                    self.assertLess(abs(value-expected), self.tolerance(arithmetic))
                self.assert_native(arithmetic, basis.v(y)+basis.q(y)
                                   +basis.v_dy(y)+basis.q_dy(y)
                                   +basis.v_dx(y)+basis.q_dx(y))

    def test_derivatives_and_chain_rule_against_independent_differentiation(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                basis = GalerkinBasis(5, arithmetic, L="1.75")
                y = arithmetic.scalar("0.375")
                x = basis.mapping.to_physical(y)
                # Independent explicit polynomials avoid checking a recurrence
                # against the same recurrence in the test oracle.
                polynomials = (
                    lambda t: 1+0*t, lambda t: t,
                    lambda t: 2*t*t-1, lambda t: 4*t**3-3*t,
                    lambda t: 8*t**4-8*t*t+1,
                )
                oracle = arithmetic if arithmetic.use_mpmath else MpmathArithmetic(40)
                coordinate = oracle.scalar(str(y))
                length = oracle.scalar("1.75")
                for family, reference, physical, factor in (
                    (polynomials[:4], basis.v_dy(y), basis.v_dx(y), lambda t: 1-t*t),
                    (polynomials, basis.q_dy(y), basis.q_dx(y), lambda t: 1-t),
                ):
                    for k, polynomial in enumerate(family):
                        derivative = oracle.context.diff(
                            lambda t: factor(t)*polynomial(t), coordinate)
                        expected_dy = arithmetic.scalar(derivative)
                        expected_dx = arithmetic.scalar(
                            derivative*(1-coordinate)**2/(2*length))
                        self.assertLess(abs(reference[k]-expected_dy),
                                        self.tolerance(arithmetic))
                        self.assertLess(abs(physical[k]-expected_dx),
                                        self.tolerance(arithmetic))
                self.assertLess(abs(basis.mapping.to_reference(x)-y),
                                self.tolerance(arithmetic))

    def test_lift_has_prescribed_wall_and_infinite_traces(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                basis = GalerkinBasis(4, arithmetic, L="2.5")
                lift = basis.lift("1.125")
                wall = arithmetic.scalar("1.125")
                self.assertIsInstance(lift, AffineWallLift)
                self.assertEqual(lift.value(-1), wall)
                self.assertEqual(lift.value(1), 0)
                self.assertEqual(lift.at(0), wall)
                self.assertEqual(lift.at(arithmetic.inf), 0)
                self.assertEqual(lift.derivative_at(arithmetic.inf), 0)
                for x in (0, arithmetic.scalar("0.25"), arithmetic.scalar(10)):
                    expected = wall*basis.mapping.L/(x+basis.mapping.L)
                    expected_dx = -wall*basis.mapping.L/(x+basis.mapping.L)**2
                    self.assertLess(abs(lift.at(x)-expected), self.tolerance(arithmetic))
                    self.assertLess(abs(lift.derivative_at(x)-expected_dx),
                                    self.tolerance(arithmetic))
                self.assertEqual(lift.dy(1), -wall/2)
                self.assert_native(arithmetic, (lift.value(0), lift.dy(0), lift.dx(0)))

    def test_physical_lift_retains_tail_when_reference_coordinate_rounds_to_one(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                lift = GalerkinBasis(4, arithmetic).lift(1)
                x = arithmetic.scalar("1e100")
                expected = arithmetic.scalar(1)/(1+x)
                expected_dx = -(arithmetic.scalar(1)/(1+x))/(1+x)
                self.assertLess(abs(lift.at(x)/expected-1), self.tolerance(arithmetic))
                self.assertLess(abs(lift.derivative_at(x)/expected_dx-1),
                                self.tolerance(arithmetic))
        lift = GalerkinBasis(4, NumpyArithmetic()).lift("1e308")
        self.assertEqual(lift.value(-1), np.float64("1e308"))
        self.assertEqual(lift.at(0), np.float64("1e308"))

    def test_each_v_function_lies_in_q_space(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                basis = GalerkinBasis(8, arithmetic)
                for coordinate in ("-0.87", "0", "0.39", "0.98"):
                    y = arithmetic.scalar(coordinate)
                    q, v = basis.q(y), basis.v(y)
                    for k, value in enumerate(v):
                        # y*T_0=T_1, and y*T_k=(T_(k-1)+T_(k+1))/2.
                        expected = q[0]+q[1] if k == 0 else q[k]+(q[k-1]+q[k+1])/2
                        self.assertLess(abs(value-expected), self.tolerance(arithmetic))

    def test_physical_mass_and_derivative_grams_have_correct_measure(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                basis = GalerkinBasis(4, arithmetic, L="1.5")
                factory = QuadratureFactory(arithmetic, L="1.5")
                plan = factory.plan(basis.N+1)
                table = basis.tabulate(plan)
                zero, length = arithmetic.scalar(0), basis.mapping.L
                for rows, expected in (
                    (table.v, 16*length/3), (table.q, 4*length),
                    (table.v_dx, 32/(15*length)), (table.q_dx, 4/(3*length)),
                ):
                    observed = sum((weight*row[0]**2 for row, weight in
                                    zip(rows, plan.physical_weights)), zero)
                    self.assertLess(abs(observed-expected), self.tolerance(arithmetic))
                    # Every nonzero polynomial in these conforming spaces has
                    # strictly positive physical mass and derivative energy.
                    coefficients = tuple(arithmetic.scalar(k+1) for k in range(len(rows[0])))
                    energy = sum((weight*sum((a*b for a, b in zip(coefficients, row)), zero)**2
                                  for row, weight in zip(rows, plan.physical_weights)), zero)
                    self.assertGreater(energy, 0)
                # For the highest Q mode the physical derivative Gram
                # integrand has degree 2*N.  N+1 Gauss nodes are sufficient;
                # only N nodes fail, despite integrating the low-mode checks.
                expected = arithmetic.scalar(24908)/(315*length)
                observed = sum((weight*row[3]**2 for row, weight in
                                zip(table.q_dx, plan.physical_weights)), zero)
                self.assertLess(abs(observed-expected), self.tolerance(arithmetic))
                coarse = factory.plan(basis.N)
                coarse_table = basis.tabulate(coarse)
                underintegrated = sum((weight*row[3]**2 for row, weight in
                                       zip(coarse_table.q_dx, coarse.physical_weights)), zero)
                self.assertGreater(abs(underintegrated-expected), arithmetic.scalar("0.001"))

    def test_table_is_immutable_native_and_agrees_with_scalar_evaluation(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                basis = GalerkinBasis(4, arithmetic, L="2.25")
                plan = QuadratureFactory(arithmetic, L="2.25", interfaces=(1,)).plan(3)
                table = basis.tabulate(plan)
                self.assertEqual(table.nodes, plan.nodes)
                self.assertIs(table.mapping, basis.mapping)
                self.assertEqual(table.precision_signature, plan.precision_signature)
                table.check_precision()
                for method, rows in ((basis.v, table.v), (basis.q, table.q),
                                     (basis.v_dx, table.v_dx), (basis.q_dx, table.q_dx)):
                    self.assertIsInstance(rows, tuple)
                    for y, row in zip(plan.nodes, rows):
                        self.assertEqual(row, method(y))
                        self.assertIsInstance(row, tuple)
                        self.assert_native(arithmetic, row)
                with self.assertRaises(FrozenInstanceError):
                    table.v = ()
                with self.assertRaises(FrozenInstanceError):
                    basis.N = 7
                with self.assertRaises(FrozenInstanceError):
                    basis.lift(1).u_b = 2

    def test_full_degree_physical_grams_are_exact_with_N_plus_one_nodes(self):
        def multiply(left, right):
            result = [0]*(len(left)+len(right)-1)
            for i, a in enumerate(left):
                for j, b in enumerate(right):
                    result[i+j] += a*b
            return result

        def moment_integral(coefficients):
            return sum((Fraction(2*value, degree+1)
                        for degree, value in enumerate(coefficients)
                        if degree % 2 == 0), Fraction(0))

        # Independent ordinary polynomials for the Q combination (1,-2,3,-1):
        # p=-2+y+6*y²-4*y³, h=(1-y)*p, h'=3+10*y-30*y²+16*y³.
        p = (-2, 1, 6, -4)
        h_derivative = (3, 10, -30, 16)
        mass_moment = moment_integral(multiply(p, p))
        derivative_moment = moment_integral(multiply(
            (1, -2, 1), multiply(h_derivative, h_derivative)))
        for arithmetic in self.arithmetics():
            basis = GalerkinBasis(4, arithmetic, L="1.5")
            length, zero = basis.mapping.L, arithmetic.scalar(0)
            expected_mass = (2*length*arithmetic.scalar(mass_moment.numerator)
                             / mass_moment.denominator)
            expected_derivative = (arithmetic.scalar(derivative_moment.numerator)
                                   / (2*length*derivative_moment.denominator))
            for interfaces in ((), (1, 3)):
                with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps,
                                  interfaces=interfaces):
                    plan = QuadratureFactory(arithmetic, L="1.5", interfaces=interfaces).plan(5)
                    table = basis.tabulate(plan)
                    for rows, expected in ((table.q, expected_mass),
                                           (table.q_dx, expected_derivative)):
                        observed = sum((weight*sum((coefficient*value for coefficient, value in
                                                   zip((1, -2, 3, -1), row)), zero)**2
                                        for row, weight in zip(rows, plan.physical_weights)), zero)
                        self.assertLess(abs(observed-expected),
                                        self.tolerance(arithmetic)*abs(expected))

    def test_tabulation_rejects_other_contexts_and_mapping_lengths(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            basis = GalerkinBasis(4, arithmetic, L=2)
            other = MpmathArithmetic(50) if arithmetic.use_mpmath else NumpyArithmetic()
            with self.assertRaises(ValueError):
                basis.tabulate(QuadratureFactory(other, L=2).plan(3))
            with self.assertRaises(ValueError):
                basis.tabulate(QuadratureFactory(arithmetic, L=3).plan(3))

    def test_exact_rational_manufactured_pair_at_N4(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                basis = GalerkinBasis(4, arithmetic)
                # u=1/(1+x) is the lift; w=-2/(1+x)^3
                #   =(1-y)*(-3*T_0/8+T_1/2-T_2/8).
                coefficients = arithmetic.vector([0, 0, 0, "-0.375", "0.5", "-0.125", 0])
                for x in (0, arithmetic.scalar("0.125"), 1, arithmetic.scalar("7.25")):
                    y = basis.mapping.to_reference(x)
                    u, w = basis.evaluate_fields(coefficients, y, u_b=1)
                    self.assertLess(abs(u-1/(1+x)), self.tolerance(arithmetic))
                    self.assertLess(abs(w+2/(1+x)**3), self.tolerance(arithmetic))
                self.assertEqual(basis.evaluate_fields(coefficients, 1, u_b=1), (0, 0))

    def test_fields_are_independent_and_coefficient_count_is_checked(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            basis = GalerkinBasis(3, arithmetic)
            zero = arithmetic.vector([0]*basis.unknown_count)
            w_changed = arithmetic.vector([0, 0, 2, 0, 0])
            u0, w0 = basis.evaluate_fields(zero, "0.25", u_b=3)
            u1, w1 = basis.evaluate_fields(w_changed, "0.25", u_b=3)
            self.assertEqual(u0, u1)
            self.assertEqual(w0, 0)
            self.assertEqual(w1, arithmetic.scalar("1.5"))
            a, b = basis.split_coefficients(w_changed)
            self.assertEqual((len(a), len(b)), (basis.n_u, basis.n_w))
            for invalid in ([0]*4, [0]*6, [0, 0, "nan", 0, 0]):
                with self.assertRaises(ValueError):
                    basis.split_coefficients(invalid)

    def test_high_precision_wall_mapping_and_coefficients_do_not_leak_float64(self):
        arithmetic = MpmathArithmetic(120)
        correction = arithmetic.scalar("1e-80")
        decimal = "1."+"0"*79+"1"
        basis = GalerkinBasis(4, arithmetic, L=decimal)
        lift = basis.lift(decimal)
        self.assertLess(abs((lift.value(-1)-1)-correction), arithmetic.scalar("1e-118"))
        self.assertLess(abs((basis.mapping.L-1)-correction), arithmetic.scalar("1e-118"))
        coefficients = [correction, 0, 0, 0, 0, 0, 0]
        u, w = basis.evaluate_fields(coefficients, 0)
        self.assertEqual(u, correction)
        self.assertEqual(w, 0)
        low = GalerkinBasis(4, MpmathArithmetic(30), L=decimal)
        self.assertEqual(low.mapping.L, 1)
        previous = mp.mp.prec
        before = basis.v_dx(arithmetic.scalar("0.123456789"))
        try:
            mp.mp.dps = 17
            self.assertEqual(basis.v_dx(arithmetic.scalar("0.123456789")), before)
            self.assertEqual(arithmetic.dps, 120)
        finally:
            mp.mp.prec = previous

    def test_changed_owned_precision_invalidates_bases_lifts_and_tables(self):
        arithmetic = MpmathArithmetic(30)
        basis = GalerkinBasis(4, arithmetic)
        lift = basis.lift(1)
        plan = QuadratureFactory(arithmetic).plan(3)
        table = basis.tabulate(plan)
        arithmetic.context.dps = 50
        for operation in (basis.check_precision, lambda: basis.v(0), lambda: basis.q(0),
                          lambda: lift.at(1), table.check_precision,
                          lambda: basis.tabulate(plan)):
            with self.assertRaises(RuntimeError):
                operation()
        rebuilt = GalerkinBasis(4, arithmetic)
        rebuilt.tabulate(QuadratureFactory(arithmetic).plan(3)).check_precision()

    def test_invalid_coordinates_wall_values_and_lengths_are_rejected(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            basis = GalerkinBasis(3, arithmetic)
            for value in ("nan", "inf", "-inf", "1.1", "-1.1"):
                for operation in (basis.v, basis.q, basis.v_dy, basis.q_dy,
                                  basis.v_dx, basis.q_dx, basis.lift(1).value):
                    with self.subTest(value=value, operation=operation.__name__):
                        with self.assertRaises(ValueError):
                            operation(value)
            for invalid in ("nan", "inf", "-inf"):
                with self.assertRaises(ValueError):
                    basis.lift(invalid)
            for invalid in (0, -1, "nan", "inf"):
                with self.assertRaises(ValueError):
                    GalerkinBasis(3, arithmetic, L=invalid)
            for coordinate in ([0], 1j):
                with self.assertRaises(TypeError):
                    basis.v(coordinate)


if __name__ == "__main__":
    unittest.main()
