"""Native BSK definitions retain the established physical model."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

import mpmath as mp

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown import load_solution, save_solution
from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.problems.bsk import overscreening_breakdown
from overscreening_breakdown.problems.bsk_problem import BSKProblem
from overscreening_breakdown.solvers import solve


class BSKProblemTests(unittest.TestCase):
    def test_description_is_frozen_and_copies_geometry_without_rounding(self):
        domain = [0, "inf"]
        boundaries = [["0.05", None, None, "0", None], [None]*5]
        sigma = Decimal("10.0000000000000000000000000000000000000001")
        problem = BSKProblem(sigma=sigma, delta="2.00000000000000000000001",
                             domain=domain, boundary_conditions=boundaries)
        domain[1] = 2
        boundaries[0][0] = 7
        self.assertEqual(problem.domain, (0, "inf"))
        self.assertEqual(problem.boundary_conditions[0][0], "0.05")
        self.assertIs(problem.sigma, sigma)
        self.assertEqual(problem.delta, "2.00000000000000000000001")
        with self.assertRaises(FrozenInstanceError):
            problem.mu = 1
        with self.assertRaises(TypeError):
            BSKProblem(initial_guess=1)
        with self.assertRaises(ValueError):
            BSKProblem(domain=(0,))
        with self.assertRaises(ValueError):
            BSKProblem(boundary_conditions=((0,), (0,)))

    def test_physical_source_and_jacobian_match_legacy_explicit_parameters(self):
        boundaries = (("0.05", None, None, "0", None), (None,)*5)
        modern = BSKProblem(sigma="3", mu="0.2", gamma="0.5", delta="2",
                            boundary_conditions=boundaries)
        old = overscreening_breakdown()
        old.set_parameters(sigma="3", mu="0.2", gamma="0.5", delta="2",
                           u0="0.05", u0xxx="0")
        for a in (NumpyArithmetic(), MpmathArithmetic(15), MpmathArithmetic(30),
                  MpmathArithmetic(50), MpmathArithmetic(100), MpmathArithmetic(120)):
            first, second = modern.bind(a), old.bind(a)
            with self.subTest(backend=a.backend, digits=a.dps):
                self.assertEqual(first.delta, second.delta)
                self.assertEqual(first.boundary_conditions, second.boundary_conditions)
                for x_text, u_text in (("0", "0.05"), ("0.125", "-0.3"), ("4", "2")):
                    x, u = a.scalar(x_text), a.scalar(u_text)
                    self.assertEqual(first.physical_rhs(x, u), second.physical_rhs(x, u))
                    self.assertEqual(first.physical_rhs_u(x, u), second.physical_rhs_u(x, u))
                    self.assertIsInstance(first.physical_rhs(x, u), type(a.scalar(0)))

    def test_gaussian_normalization_is_preserved_and_derivative_is_independent(self):
        a = MpmathArithmetic(80)
        problem = BSKProblem(sigma="7", mu="0.2", gamma="0.5").bind(a)
        # At u=0,x=0 the original Gaussian has no 1/sigma normalization.
        expected = -a.scalar("0.1")/a.sqrt(2*a.pi)
        self.assertLess(abs(problem.physical_rhs(0, 0)-expected), a.scalar("1e-78"))
        for x, u in ((a.scalar("0.125"), a.scalar("0.37")), (a.scalar(3), a.scalar(-1))):
            derivative = a.context.diff(lambda value: problem.physical_rhs(x, value), u)
            self.assertLess(abs(problem.physical_rhs_u(x, u)-derivative), a.scalar("1e-77"))

    def test_bindings_are_isolated_and_raw_decimal_parameters_survive(self):
        original_bits = mp.mp.prec
        high, low = MpmathArithmetic(120), MpmathArithmetic(15)
        raw = "1."+"0"*79+"1"
        problem = BSKProblem(sigma=raw, mu="0.2", gamma="0.5", delta=raw,
                             boundary_conditions=((raw, None, None, 0, None), (None,)*5))
        first = problem.bind(high)
        x, u = high.scalar("0.125"), high.scalar("0.37")
        value = first.physical_rhs(x, u)
        second = problem.bind(low)
        self.assertIs(first.arithmetic, high)
        self.assertIs(second.arithmetic, low)
        self.assertEqual(first.physical_rhs(x, u), value)
        self.assertEqual(second.delta, 1)
        self.assertGreater(first.delta, 1)
        self.assertGreater(first.boundary_conditions[0][0], 1)
        self.assertEqual(mp.mp.prec, original_bits)

    def test_initial_guess_receives_its_owner_and_default_uses_wall_value(self):
        calls = []
        def guess(x, arithmetic):
            calls.append(arithmetic)
            return arithmetic.scalar("0.05")*arithmetic.exp(-x*x)
        problem = BSKProblem(initial_guess=guess)
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            physical = problem.bind(a)
            self.assertEqual(physical.base_solution(a.scalar(1)), a.scalar("0.05")*a.exp(-a.scalar(1)))
            self.assertIs(calls[-1], a)
            default = BSKProblem(boundary_conditions=(("0.05", None, None, 0, None), (None,)*5)).bind(a)
            self.assertEqual(default.base_solution(a.scalar(1)), a.scalar("0.05")*a.exp(-a.scalar(1)))

    def test_invalid_parameters_and_geometry_fail_at_bind(self):
        a = NumpyArithmetic()
        for options in ({"sigma": 0}, {"sigma": -1}, {"sigma": "inf"}, {"mu": "nan"},
                        {"gamma": -1}, {"gamma": "inf"}, {"delta": -1},
                        {"domain": (1, 2)}, {"domain": (0, -1)},
                        {"boundary_conditions": (("inf", None, None, 0, None), (None,)*5)}):
            with self.subTest(options=options), self.assertRaises((ValueError, OverflowError)):
                BSKProblem(**options).bind(a)

    def test_second_order_and_finite_domain_data_remain_explicit(self):
        a = MpmathArithmetic(30)
        second_order = BSKProblem(delta=0, boundary_conditions=((1, None, None, None, None), (None,)*5))
        physical = second_order.bind(a)
        self.assertEqual(physical.delta, 0)
        self.assertIsNone(physical.boundary_conditions[0][3])
        finite = BSKProblem(delta=0, domain=(0, "2"),
                            boundary_conditions=((1, None, None, None, None), (0, None, None, None, None)))
        self.assertEqual(finite.bind(a).domain, (a.scalar(0), a.scalar(2)))
        # A description retains supplied data; each solver validates its order.
        with self.assertRaises(ValueError):
            solve(BSKProblem(delta=0), 4)

    def test_both_solvers_accept_one_description_in_numpy_and_mpmath(self):
        problem = BSKProblem(sigma="3", mu="0.1", gamma="0.5", delta="1",
                             boundary_conditions=(("0.05", None, None, 0, None), (None,)*5))
        values = {}
        for use_mp in (False, True):
            for method in ("collocation", "galerkin"):
                with self.subTest(solver=method, mpmath=use_mp):
                    # Gaussian integration is checked separately from native
                    # Newton accuracy; this is a bounded interface regression.
                    quadrature = ({"quadrature_atol": "1e-10", "quadrature_rtol": "1e-10"}
                                  if method == "galerkin" else {})
                    result = solve(problem, 8, solver=method, use_mpmath=use_mp, precision=30,
                                   tolerance="1e-20" if use_mp else "1e-10", **quadrature)
                    self.assertTrue(result.converged)
                    self.assertLess(abs(result.u(0)-result.arithmetic.scalar("0.05")),
                                    result.arithmetic.scalar("1e-18" if use_mp else "1e-9"))
                    self.assertTrue(result.arithmetic.isfinite(result.flux(1)))
                    if method == "galerkin":
                        self.assertTrue(result.quadrature_converged)
                    if use_mp:
                        self.assertLessEqual(result.newton_result.residual_norm,
                                             result.arithmetic.scalar("1e-20"))
                    values[method, use_mp] = float(result.u(1))
                    with tempfile.TemporaryDirectory(prefix="overscreening-bsk-", dir="/tmp") as directory:
                        path = Path(directory)/"solution.json"
                        save_solution(result, path)
                        restored = load_solution(path, problem=problem, solver=method)
                        self.assertEqual(tuple(restored.coefficients), tuple(result.coefficients))
                        self.assertEqual(restored.u(1), result.u(1))
                        self.assertEqual(restored.flux(1), result.flux(1))
                        if method == "galerkin":
                            quadrature["quadrature_order"] = result.plan.order
                        restarted = solve(
                            problem, 8, solver=method, initial_guess=path,
                            use_mpmath=use_mp, precision=30,
                            tolerance="1e-20" if use_mp else "1e-10", **quadrature)
                        self.assertTrue(restarted.converged)
        for method in ("collocation", "galerkin"):
            self.assertLess(abs(values[method, False]-values[method, True]), 1e-8)

    def test_nonconstant_epsilon_requires_galerkin(self):
        epsilon = Coefficient(lambda x, a: 1+a.scalar(1)/(1+x),
                              lower_bound=1, upper_bound=2, wall_value=2)
        problem = BSKProblem(sigma="3", mu="0.1", gamma="0.5", delta="1", epsilon=epsilon,
                             boundary_conditions=(("0.05", None, None, 0, None), (None,)*5))
        with self.assertRaisesRegex(ValueError, "epsilon"):
            solve(problem, 8, solver="collocation")
        for use_mp in (False, True):
            result = solve(problem, 8, solver="galerkin", use_mpmath=use_mp, precision=30,
                           tolerance="1e-20" if use_mp else "1e-10",
                           quadrature_atol="1e-10", quadrature_rtol="1e-10")
            self.assertTrue(result.converged)
            self.assertTrue(result.quadrature_converged)


if __name__ == "__main__":
    unittest.main()
