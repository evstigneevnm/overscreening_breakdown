"""Physical BSK source, legacy normalization, and independent bound views."""

from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient, ConstantCoefficient
from overscreening_breakdown import (
    overscreening_breakdown,
    overscreening_breakdown_segment,
)
from overscreening_breakdown.collocation.spectral1d import solve_nonlinear_problem


def independent_source(ctx, x, u, sigma, mu, gamma):
    """Original physical model, without numerical equation normalization."""
    gaussian = ctx.exp(-x ** 2 / (2 * sigma ** 2)) / ctx.sqrt(2 * ctx.pi)
    return ((ctx.sinh(u) - gaussian * mu * ctx.exp(u) / 2)
            / (1 + 2 * gamma * ctx.sinh(u / 2) ** 2))


class PhysicalRhsTests(unittest.TestCase):
    def problem(self, **parameters):
        problem = overscreening_breakdown()
        defaults = dict(sigma=3, mu="0.7", gamma="0.4", delta=2, u0=1)
        defaults.update(parameters)
        problem.set_parameters(**defaults)
        return problem

    def test_bound_physical_source_and_derivative_against_independent_model(self):
        for dps in (30, 120):
            arithmetic = MpmathArithmetic(dps)
            ctx = arithmetic.context
            source = self.problem().bind(arithmetic)
            sigma, mu, gamma = map(arithmetic.scalar, (3, "0.7", "0.4"))
            for x_text, u_text in (("0", "-0.8"), ("0.6", "1.3"), ("4", "-2")):
                with self.subTest(dps=dps, x=x_text, u=u_text):
                    x, u = map(arithmetic.scalar, (x_text, u_text))
                    model = lambda value: independent_source(ctx, x, value, sigma, mu, gamma)
                    tolerance = ctx.mpf(10) ** (-dps + 5)
                    self.assertLess(abs(source.physical_rhs(x, u) - model(u)), tolerance)
                    self.assertLess(abs(source.physical_rhs_u(x, u) - ctx.diff(model, u)),
                                    tolerance)

    def test_large_state_derivative_retains_small_nonzero_value(self):
        for dps in (30, 120):
            arithmetic = MpmathArithmetic(dps)
            ctx = arithmetic.context
            for mu in (0, "0.7"):
                with self.subTest(dps=dps, mu=mu):
                    bound = self.problem(gamma=1, mu=mu).bind(arithmetic)
                    x, u = arithmetic.scalar("0.6"), arithmetic.scalar(100)
                    gaussian = ctx.exp(-x ** 2 / 18) / ctx.sqrt(2 * ctx.pi)
                    expected = (1 - arithmetic.scalar(mu) * gaussian / 2) / ctx.cosh(u) ** 2
                    actual = bound.physical_rhs_u(x, u)
                    self.assertGreater(actual, 0)
                    self.assertLess(abs(actual / expected - 1),
                                    ctx.mpf(10) ** (-dps + 5))

    def test_decimal_parameters_survive_binding_below_float64_resolution(self):
        arithmetic = MpmathArithmetic(120)
        ctx = arithmetic.context
        delta = "1." + "0" * 79 + "1"
        epsilon = "1." + "0" * 79 + "2"
        gamma = "0.5" + "0" * 78 + "1"
        problem = self.problem(delta=delta, gamma=gamma, epsilon=epsilon)
        bound = problem.bind(arithmetic)
        self.assertEqual(bound.delta, ctx.mpf(delta))
        self.assertGreater(bound.delta - 1, 0)
        self.assertEqual(bound.epsilon(2), ctx.mpf(epsilon))
        x, u = ctx.mpf("0.6"), ctx.mpf(2)
        expected = independent_source(ctx, x, u, ctx.mpf(3), ctx.mpf("0.7"),
                                      ctx.mpf(gamma))
        rounded = independent_source(ctx, x, u, ctx.mpf(3), ctx.mpf("0.7"),
                                     ctx.mpf("0.5"))
        actual = bound.physical_rhs(x, u)
        self.assertLess(abs(actual - expected), ctx.mpf("1e-115"))
        self.assertGreater(abs(actual - rounded), ctx.mpf("1e-83"))

    def test_bound_views_snapshot_parameters_and_do_not_mutate_legacy_math(self):
        previous_dps = mp.mp.dps
        problem = self.problem(sigma=3, mu=0.7, gamma=0.4, delta=2)
        x, u = np.array([0.2, 0.7]), np.array([-0.5, 1.2])
        legacy_before = np.asarray(problem.physical_rhs(x, u)).copy()
        low = problem.bind(MpmathArithmetic(30))
        high = problem.bind(MpmathArithmetic(120))
        low_value = low.physical_rhs("0.6", "1.3")
        high_value = high.physical_rhs("0.6", "1.3")
        np.testing.assert_array_equal(problem.physical_rhs(x, u), legacy_before)
        self.assertEqual(np.asarray(problem.physical_rhs(x, u)).dtype.kind, "f")
        try:
            mp.mp.dps = previous_dps + 11
            problem.set_use_mp_math(mp.mp)
            problem.set_parameters(sigma=7, mu=1.5, gamma=0.9, delta=4, u0=8)
            self.assertEqual(low.physical_rhs("0.6", "1.3"), low_value)
            self.assertEqual(high.physical_rhs("0.6", "1.3"), high_value)
            self.assertEqual(low.delta, 2)
            self.assertEqual(high.delta, 2)
            self.assertEqual(low.arithmetic.dps, 30)
            self.assertEqual(high.arithmetic.dps, 120)
        finally:
            mp.mp.dps = previous_dps

    def test_segment_boundary_data_and_domain_are_snapshots(self):
        problem = overscreening_breakdown_segment()
        problem.set_parameters(u0=3, u0x=4, u0xx=5, u0xxx=6, u0xxxx=7)
        bound = problem.bind(MpmathArithmetic(30))
        self.assertEqual(bound.domain, (0, 50))
        self.assertEqual(bound.boundary_conditions,
                         ((3, 4, 5, 6, 7), (0, 0, None, None, None)))
        problem.get_boundary_conditions()[0][0] = 999
        problem.get_domain()[1] = 60
        problem.set_parameters(u0=8)
        self.assertEqual(bound.get_domain(), (0, 50))
        self.assertEqual(bound.get_boundary_conditions()[0][0], 3)
        self.assertEqual(bound.get_boundary_conditions()[1][:2], (0, 0))

    def test_default_and_custom_initial_guesses_are_bound_independently(self):
        problem = self.problem(u0=3)
        arithmetic = MpmathArithmetic(50)
        bound = problem.bind(arithmetic)
        problem.set_parameters(u0=9)
        x = arithmetic.scalar("0.7")
        self.assertEqual(bound.base_solution(x), 3 * arithmetic.exp(-x))
        problem.set_parameters(initial_guess_solution=lambda x: x + 4)
        custom = problem.bind(arithmetic)
        problem.set_parameters(initial_guess_solution=lambda x: x + 100)
        self.assertEqual(custom.base_solution(2), 6)

    def test_legacy_rhs_preserves_homotopy_and_delta_normalization(self):
        x = np.array([0, 0.5, 3.0])
        u = np.array([-0.8, 0.2, 1.3])
        q = 0.4
        for delta in (0, 2):
            with self.subTest(delta=delta):
                problem = self.problem(sigma=3, mu=0.7, gamma=0.4, delta=delta)
                scale = np.exp(-q * x) / (delta ** 2 if delta > 0 else 1)
                np.testing.assert_allclose(problem.right_hand_side(x, u, q),
                                           problem.physical_rhs(x, u) * scale,
                                           rtol=1e-14, atol=1e-15)
                np.testing.assert_allclose(
                    problem.right_hand_side_linearization(x, u, q),
                    problem.physical_rhs_u(x, u) * scale, rtol=1e-14, atol=1e-15,
                )

    def test_source_keeps_existing_gaussian_normalization(self):
        arithmetic = MpmathArithmetic(50)
        ctx = arithmetic.context
        for sigma in (2, 7):
            with self.subTest(sigma=sigma):
                bound = self.problem(sigma=sigma, mu=2, gamma=0).bind(arithmetic)
                x = ctx.mpf("0.8")
                expected = -ctx.exp(-x ** 2 / (2 * sigma ** 2)) / ctx.sqrt(2 * ctx.pi)
                self.assertLess(abs(bound.physical_rhs(x, 0) - expected), ctx.mpf("1e-48"))

    def test_epsilon_belongs_to_flux_and_does_not_rescale_physical_source(self):
        arithmetic = MpmathArithmetic(50)
        unit = self.problem(epsilon=1).bind(arithmetic)
        heterogeneous = self.problem(epsilon=2).bind(arithmetic)
        for method in ("physical_rhs", "physical_rhs_u"):
            self.assertEqual(getattr(unit, method)("0.4", "1.2"),
                             getattr(heterogeneous, method)("0.4", "1.2"))

    def test_delta_is_exposed_without_duplicate_storage(self):
        problem = self.problem(delta=2)
        self.assertEqual(problem.delta, 2)
        problem.set_parameters(delta=7)
        self.assertEqual(problem.delta, 7)
        with self.assertRaises(AttributeError):
            problem.delta = 3

    def test_omitting_epsilon_in_parameter_update_preserves_description(self):
        description = ConstantCoefficient(2)
        problem = self.problem(epsilon=description)
        problem.set_parameters(delta=3)
        self.assertIs(problem.epsilon, description)
        with self.assertRaises(TypeError):
            problem.set_parameters(epsilon=None)

    def test_binding_validates_bsk_parameter_constraints(self):
        for name, value in (("sigma", 0), ("sigma", -1), ("sigma", "nan"),
                            ("sigma", "inf"), ("gamma", -0.1),
                            ("gamma", "nan"), ("gamma", "inf"),
                            ("mu", "nan"), ("mu", "inf")):
            with self.subTest(parameter=name, value=value):
                problem = self.problem(**{name: value})
                with self.assertRaises(ValueError):
                    problem.bind(MpmathArithmetic(30))

    def test_legacy_collocation_rejects_nonidentity_epsilon_before_assembly(self):
        opaque_identity = Coefficient.from_function(
            lambda x: 1, lower_bound=1, upper_bound=1, wall_value=1,
        )
        for epsilon in (2, opaque_identity):
            with self.subTest(epsilon=epsilon):
                problem = self.problem(epsilon=epsilon)
                with patch.object(solve_nonlinear_problem,
                                  "_solve_nonlinear_problem__form_matrices") as assemble:
                    with self.assertRaises(ValueError):
                        solve_nonlinear_problem(4, problem=problem)
                    assemble.assert_not_called()
                for method, args in (("left_hand_side", ()),
                                     ("right_hand_side", (0, 1, 0)),
                                     ("right_hand_side_linearization", (0, 1, 0))):
                    with self.assertRaises(ValueError):
                        getattr(problem, method)(*args)

    def test_legacy_collocation_rejects_epsilon_changed_after_set_problem(self):
        problem = self.problem(sigma=3, mu=0, gamma=0.5, delta=1)
        with redirect_stdout(io.StringIO()):
            solver = solve_nonlinear_problem(4, problem=problem, use_globalization=False)
            problem.set_parameters(epsilon=2)
            with self.assertRaises(ValueError):
                solver.solve_problem()


if __name__ == "__main__":
    unittest.main()
