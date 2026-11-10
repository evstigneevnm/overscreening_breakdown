"""Independent identities and arithmetic isolation for shared case definitions."""

from dataclasses import FrozenInstanceError
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.collocation.adapter import CollocationProblemAdapter
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.manufactured import (
    linear_case, nonlinear_case, polynomial_segment_case, rational_case, transmission_case,
)
from overscreening_breakdown.quadrature import QuadratureFactory


class ManufacturedCaseTests(unittest.TestCase):
    def cases(self):
        return ([linear_case(number) for number in range(1, 9)]
                +[nonlinear_case(number) for number in range(1, 8)]
                +[rational_case(name, nonlinear=True) for name in
                  ("constant", "scaled", "smooth", "exponential")]
                +[transmission_case(), polynomial_segment_case()])

    def assert_close(self, actual, expected, arithmetic, digits=None):
        if digits is None:
            digits = max(8, arithmetic.dps-7) if arithmetic.use_mpmath else 10
        target = arithmetic.scalar("1e-{}".format(digits))*max(1, abs(expected))
        self.assertLessEqual(abs(actual-expected), target)

    def test_catalog_is_immutable_and_classifies_equations_and_boundary_families(self):
        both = ("collocation", "galerkin")
        expected_linear = (both, both, ("collocation",), ("collocation",),
                           ("collocation",), ("collocation",), both, ("legacy_collocation",))
        expected_nonlinear = (("collocation",), both, both, both, ("collocation",), both, both)
        self.assertEqual(tuple(linear_case(n).supported_solvers for n in range(1, 9)), expected_linear)
        self.assertEqual(tuple(nonlinear_case(n).supported_solvers for n in range(1, 8)), expected_nonlinear)
        with self.assertRaises(FrozenInstanceError):
            linear_case(1).name = "changed"
        for invalid in (0, -1, True, "1", 1.5, 9):
            with self.subTest(number=invalid), self.assertRaises(ValueError):
                linear_case(invalid)
        with self.assertRaises(ValueError):
            nonlinear_case(8)
        with self.assertRaises(ValueError):
            rational_case("unknown")
        with self.assertRaises(ValueError):
            rational_case(initial="unknown")
        with self.assertRaises(TypeError):
            rational_case(nonlinear=1)
        with self.assertRaisesRegex(ValueError, "finite-delta"):
            linear_case(8).make_problem()
        with self.assertRaisesRegex(ValueError, "historical"):
            rational_case().legacy_problem()

    def test_analytic_fields_match_independent_differentiation_at_all_precisions(self):
        for digits in (15, 30, 50, 100, 120):
            a = MpmathArithmetic(digits)
            for case in self.cases():
                with self.subTest(case=case.identifier, digits=digits):
                    for text in ("0.125", "1.5"):
                        x = a.scalar(text)
                        expected = [a.context.diff(lambda t: case.u(t, a), x, n) for n in range(1, 5)]
                        for n, value in enumerate(expected, 1):
                            self.assert_close(case.derivative(x, a, n), value, a)
                        self.assert_close(case.u_dx(x, a), expected[0], a)
                        self.assert_close(case.w(x, a), -expected[1], a)
                        self.assert_close(case.w_dx(x, a), -expected[2], a)

    def test_physical_sources_jacobians_and_wall_data_obey_independent_identities(self):
        for digits in (15, 30, 50, 100, 120):
            a = MpmathArithmetic(digits)
            for case in self.cases():
                if case.supported_solvers == ("legacy_collocation",):
                    continue
                with self.subTest(case=case.identifier, digits=digits):
                    physical = case.make_problem().bind(a)
                    for text in ("0.125", "1.5"):
                        x = a.scalar(text)
                        u = case.u(x, a)
                        source = a.context.diff(lambda t: case.flux(t, a), x)
                        self.assert_close(physical.physical_rhs(x, u), source, a)
                        trial = a.scalar("0.37")
                        derivative = a.context.diff(lambda value: physical.physical_rhs(x, value), trial)
                        self.assert_close(physical.physical_rhs_u(x, trial), derivative, a)
                    for endpoint, row in zip(physical.domain, physical.boundary_conditions):
                        if endpoint == a.inf:
                            continue
                        for order, value in enumerate(row):
                            if value is not None:
                                self.assert_close(case.derivative(endpoint, a, order), value, a)

    def test_numpy_and_mpmath_callbacks_return_native_real_scalars(self):
        high = MpmathArithmetic(150)
        for a in (NumpyArithmetic(), MpmathArithmetic(15), MpmathArithmetic(120)):
            for case in self.cases():
                with self.subTest(case=case.identifier, backend=a.backend, dps=a.dps):
                    x = a.scalar("0.375")
                    for function in (case.u, case.u_dx, case.w, case.w_dx):
                        value = function(x, a)
                        self.assertTrue(a.isfinite(value))
                        self.assertIsInstance(value, type(a.scalar(0)))
                        self.assert_close(value, a.scalar(function(high.scalar("0.375"), high)), a)
                    if "legacy_collocation" not in case.supported_solvers:
                        physical = case.make_problem().bind(a)
                        for value in (case.flux(x, a), physical.physical_rhs(x, a.scalar("0.37")),
                                      physical.physical_rhs_u(x, a.scalar("0.37"))):
                            self.assertIsInstance(value, type(a.scalar(0)))
        # A rational tail must not evaluate unused Gaussian/exponential profiles.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            a = NumpyArithmetic()
            tail = rational_case()
            x = a.scalar("1e100")
            self.assert_close(tail.u(x, a)*x, a.scalar(1), a)
            self.assertLess(tail.w(x, a), 0)
            self.assertTrue(a.isfinite(tail.flux(x, a)))
            pb_tail = nonlinear_case(1).u(a.scalar(100), a)
            self.assertGreater(pb_tail, 0)
            self.assert_close(pb_tail/(4*a.tanh(a.scalar(1)/4)*a.exp(a.scalar(-100))),
                              a.scalar(1), a)

    def test_context_binding_does_not_round_irrational_parameters_or_reuse_owners(self):
        original = mp.mp.prec
        low, high = MpmathArithmetic(15), MpmathArithmetic(120)
        for case in (linear_case(1), linear_case(6), linear_case(7), nonlinear_case(5), nonlinear_case(7)):
            description = case.make_problem()
            first = description.bind(high)
            x = high.scalar("0.125")
            before = first.physical_rhs(x, high.scalar("0.37"))
            description.bind(low)
            self.assertEqual(first.physical_rhs(x, high.scalar("0.37")), before)
            self.assertIs(first.arithmetic, high)
            self.assertEqual(first.boundary_conditions[0][0], case.u(high.scalar(0), high))
        delta = linear_case(1).make_problem().bind(high).delta
        self.assertLess(abs(delta**2-10), high.scalar("1e-117"))
        self.assertGreater(abs(delta-high.scalar(low.sqrt(low.scalar(10)))), high.scalar("1e-30"))
        wall = nonlinear_case(7).make_problem().bind(high).boundary_conditions[0][0]
        self.assertEqual(wall, high.cos(high.scalar(1)))
        self.assertEqual(mp.mp.prec, original)

    def test_historical_factories_are_fresh_and_physical_normalization_is_explicit(self):
        for case in [linear_case(n) for n in range(1, 9)]+[nonlinear_case(n) for n in range(1, 8)]:
            first, second = case.legacy_problem(), case.legacy_problem()
            self.assertIsNot(first, second)
            first._N_all = [1]
            self.assertNotEqual(first.get_N(), second.get_N())
        a = MpmathArithmetic(80)
        x = a.scalar("0.375")
        physical = linear_case(2).make_problem().bind(a)
        self.assertEqual(physical.delta, a.scalar(1)/50)
        self.assert_close(physical.physical_rhs(x, 0), a.scalar(2499)/2500*a.exp(-x), a)
        physical = nonlinear_case(2).make_problem().bind(a)
        self.assertEqual(physical.delta, 50)
        self.assert_close(physical.physical_rhs(x, a.exp(-x)), -2499*a.exp(-x), a)
        self.assertLess(physical.physical_rhs_u(x, a.scalar("0.37")), 0)

    def test_rational_reaction_initial_guesses_and_exact_mixed_space_are_shared(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50), MpmathArithmetic(120)):
            for name in ("constant", "scaled", "smooth"):
                case = rational_case(name, nonlinear=True)
                basis = GalerkinBasis(4, a, L=1)
                plan = QuadratureFactory(a, L=1).plan(8)
                assembly = GalerkinAssembly(case.make_problem(), basis, plan)
                state = a.vector([0, 0, 0, "-0.375", "0.5", "-0.125", 0])
                self.assertLess(a.norm(assembly.residual(state)), a.scalar("1e-10" if not a.use_mpmath
                                                                        else "1e-{}".format(a.dps-7)))
            default = rational_case(nonlinear=True).make_problem().bind(a)
            changed = rational_case(nonlinear=True, beta="0.1", initial="square").make_problem().bind(a)
            x = a.scalar(1)
            self.assertEqual(changed.base_solution(x), a.scalar("0.25"))
            self.assertEqual(changed.physical_rhs_u(x, a.scalar(2)), a.scalar("2.2"))
            self.assertNotEqual(changed.physical_rhs_u(x, a.scalar(2)), default.physical_rhs_u(x, a.scalar(2)))
            exp = rational_case(initial="exp").make_problem().bind(a)
            self.assertEqual(exp.base_solution(x), a.exp(-x))
            self.assertEqual(rational_case().make_problem().delta, "2")

    def test_transmission_has_C2_fields_and_nonzero_continuous_flux(self):
        for digits in (30, 120):
            a = MpmathArithmetic(digits)
            case = transmission_case()
            problem = case.make_problem().bind(a)
            self.assertEqual(problem.epsilon.interfaces, (a.scalar(1),))
            h = a.scalar("1e-{}".format(digits//2))
            for function in (case.u, case.u_dx, case.w, case.flux):
                self.assertLess(abs(function(1-h, a)-function(1+h, a)), 100*h)
            self.assert_close(case.flux(a.scalar(1), a), a.scalar(5)/4, a)
            self.assertGreater(abs(case.w_dx(1-h, a)-case.w_dx(1+h, a)), a.scalar("0.05"))
            self.assert_close(case.derivative(a.scalar(0), a, 3), -6, a)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            CollocationProblemAdapter(transmission_case().make_problem().bind(NumpyArithmetic()))

    def test_current_solver_compatibility_matches_the_case_classification(self):
        a = NumpyArithmetic()
        for case in self.cases():
            if "legacy_collocation" in case.supported_solvers:
                continue
            description = case.make_problem()
            physical = description.bind(a)
            if "collocation" in case.supported_solvers:
                CollocationProblemAdapter(physical)
            else:
                with self.assertRaises(ValueError):
                    CollocationProblemAdapter(physical)
            basis = GalerkinBasis(4, a, L=1)
            plan = QuadratureFactory(a, L=1, interfaces=physical.epsilon.interfaces).plan(8)
            if "galerkin" in case.supported_solvers:
                GalerkinAssembly(description, basis, plan)
            else:
                with self.assertRaises(ValueError):
                    GalerkinAssembly(description, basis, plan)


if __name__ == "__main__":
    unittest.main()
