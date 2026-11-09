"""Scalar physical problems and precision-isolated parameter binding."""

from dataclasses import FrozenInstanceError
import unittest

import mpmath as mp
import numpy as np

from arithmetic import MpmathArithmetic, NumpyArithmetic
from coefficients import Coefficient
from physical_problem import FluxProblem


def cubic_source(x, u, arithmetic):
    return arithmetic.exp(-x) + u**3


def cubic_source_u(x, u, arithmetic):
    return 3*u**2


class FluxProblemTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for precision in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(precision)

    def problem(self, **options):
        arguments = dict(delta=2, f=cubic_source, f_u=cubic_source_u)
        arguments.update(options)
        return FluxProblem(**arguments)

    def test_scalar_source_is_unscaled_in_each_arithmetic(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                bound = self.problem(epsilon=3).bind(arithmetic)
                x, u = arithmetic.scalar("0.5"), arithmetic.scalar("0.25")
                expected = arithmetic.exp(-x) + u**3
                self.assertEqual(bound.physical_rhs(x, u), expected)
                self.assertEqual(bound.physical_rhs_u(x, u), 3*u**2)
                self.assertEqual(bound.delta, 2)
                self.assertEqual(bound.epsilon(x), 3)
                self.assertEqual(bound.get_domain(), bound.domain)
                self.assertEqual(bound.get_boundary_conditions(), bound.boundary_conditions)

    def test_analytic_derivative_checked_independently(self):
        arithmetic = MpmathArithmetic(100)
        bound = self.problem().bind(arithmetic)
        x, u = arithmetic.scalar("0.7"), arithmetic.scalar("0.123456789")
        derivative = arithmetic.context.diff(lambda value: bound.physical_rhs(x, value), u)
        self.assertLess(abs(derivative-bound.physical_rhs_u(x, u)), arithmetic.scalar("1e-95"))

    def test_description_copies_domain_and_boundary_metadata(self):
        domain = [0, "5.25"]
        boundaries = [["1.125", None, None, "-0.25", None], [0, 0, None, None, None]]
        problem = self.problem(domain=domain, boundary_conditions=boundaries)
        domain[1] = 99
        boundaries[0][0] = 99
        boundaries[1].append(99)
        bound = problem.bind(MpmathArithmetic(50))
        self.assertEqual(bound.domain[1], bound.arithmetic.scalar("5.25"))
        self.assertEqual(bound.boundary_conditions[0][0], bound.arithmetic.scalar("1.125"))
        self.assertEqual(bound.boundary_conditions[0][3], bound.arithmetic.scalar("-0.25"))
        self.assertEqual(len(bound.boundary_conditions[1]), 5)
        self.assertIsInstance(problem.boundary_conditions, tuple)
        self.assertIsInstance(bound.boundary_conditions[0], tuple)

    def test_decimal_parameters_and_two_precisions_remain_independent(self):
        delta = "1." + "0"*79 + "1"
        wall = "2." + "0"*79 + "3"
        boundaries = ((wall, None, None, 0, None), (None,)*5)
        problem = self.problem(delta=delta, boundary_conditions=boundaries)
        low = problem.bind(MpmathArithmetic(30))
        high = problem.bind(MpmathArithmetic(120))
        self.assertEqual(problem.delta, delta)
        self.assertEqual(low.delta, 1)
        self.assertGreater(high.delta, 1)
        self.assertLess(abs((high.delta-1)-high.arithmetic.scalar("1e-80")), high.arithmetic.scalar("1e-119"))
        self.assertEqual(low.base_solution(0), 2)
        self.assertGreater(high.base_solution(0), 2)
        before = (low.physical_rhs("0.2", "0.3"), high.physical_rhs("0.2", "0.3"))
        precision = mp.mp.prec
        try:
            mp.mp.dps = 17
            self.problem(delta=5).bind(MpmathArithmetic(15))
            self.assertEqual(low.physical_rhs("0.2", "0.3"), before[0])
            self.assertEqual(high.physical_rhs("0.2", "0.3"), before[1])
            self.assertEqual(low.arithmetic.dps, 30)
            self.assertEqual(high.arithmetic.dps, 120)
        finally:
            mp.mp.prec = precision

    def test_default_and_explicit_contextual_initial_guesses(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                boundaries = ((3, None, None, 0, None), (None,)*5)
                default = self.problem(boundary_conditions=boundaries).bind(arithmetic)
                self.assertEqual(default.base_solution("0.5"), 3*arithmetic.exp(-arithmetic.scalar("0.5")))
                custom = self.problem(
                    initial_guess=lambda x, context: context.sqrt(context.scalar(2))*context.exp(-x)
                ).bind(arithmetic)
                self.assertEqual(custom.base_solution(0), arithmetic.sqrt(arithmetic.scalar(2)))
                empty = self.problem(boundary_conditions=((None,)*5, (None,)*5)).bind(arithmetic)
                self.assertEqual(empty.base_solution(1), 0)

    def test_descriptions_and_bound_views_are_frozen(self):
        problem = self.problem()
        bound = problem.bind(NumpyArithmetic())
        for value, attribute, replacement in ((problem, "delta", 9), (bound, "delta", 9), (bound, "domain", (0, 5))):
            with self.subTest(attribute=attribute, value_type=type(value).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(value, attribute, replacement)

    def test_delta_domain_and_boundary_validation(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            for delta in (-1, "nan", "inf", "-inf"):
                with self.subTest(delta=delta, backend=arithmetic.backend):
                    with self.assertRaises(ValueError):
                        self.problem(delta=delta).bind(arithmetic)
            self.assertEqual(self.problem(delta=0).bind(arithmetic).delta, 0)
            for domain in ((-1, 2), (1, 2), (0, 0), (0, -1), (0, "nan"), (0, "-inf"), ("inf", "inf")):
                with self.subTest(domain=domain, backend=arithmetic.backend):
                    with self.assertRaises(ValueError):
                        self.problem(domain=domain).bind(arithmetic)
            with self.assertRaises(ValueError):
                self.problem(boundary_conditions=(("inf", None, None, 0, None), (None,)*5)).bind(arithmetic)
        for boundaries in ((), ((0,)*5,), ((0,)*4, (0,)*5)):
            with self.assertRaises(ValueError):
                self.problem(boundary_conditions=boundaries)
        with self.assertRaises(ValueError):
            self.problem(domain=(0,))

    def test_callback_configuration_validation(self):
        for options in (dict(f=None), dict(f_u=1), dict(initial_guess=3)):
            with self.subTest(options=options):
                with self.assertRaises(TypeError):
                    self.problem(**options)

    def test_scalar_callback_inputs_and_domain_are_checked(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            bound = self.problem(domain=(0, 2)).bind(arithmetic)
            for x, u in ((-1, 0), (3, 0), ("nan", 0), ("inf", 0), (0, "nan"), (0, "inf")):
                with self.subTest(x=x, u=u, backend=arithmetic.backend):
                    with self.assertRaises(ValueError):
                        bound.physical_rhs(x, u)
                    with self.assertRaises(ValueError):
                        bound.physical_rhs_u(x, u)
            for x, u in (([1], 0), (1, [0]), (np.array([1]), 0), (1+0j, 0), (1, 0j)):
                with self.subTest(x=x, u=u, backend=arithmetic.backend):
                    with self.assertRaises(TypeError):
                        bound.physical_rhs(x, u)
            for x in (-1, 3, "nan", "inf"):
                with self.assertRaises(ValueError):
                    bound.base_solution(x)

    def test_nonfinite_trial_sources_raise_recoverable_overflow(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            for invalid in ("nan", "inf", "-inf"):
                with self.subTest(invalid=invalid, backend=arithmetic.backend):
                    bound = self.problem(
                        f=lambda x, u, context: context.scalar(invalid),
                        f_u=lambda x, u, context: context.scalar(invalid),
                    ).bind(arithmetic)
                    with self.assertRaisesRegex(OverflowError, "physical source value must be finite"):
                        bound.physical_rhs(1, 2)
                    with self.assertRaises(OverflowError):
                        bound.physical_rhs_u(1, 2)
                    initial = self.problem(initial_guess=lambda x, context: context.scalar(invalid)).bind(arithmetic)
                    with self.assertRaises(ValueError):
                        initial.base_solution(1)

    def test_unexpected_callback_errors_propagate(self):
        def broken(*arguments):
            raise RuntimeError("callback defect")
        bound = self.problem(f=broken, f_u=broken, initial_guess=broken).bind(NumpyArithmetic())
        for callback, arguments in ((bound.physical_rhs, (1, 2)), (bound.physical_rhs_u, (1, 2)), (bound.base_solution, (1,))):
            with self.assertRaisesRegex(RuntimeError, "callback defect"):
                callback(*arguments)

    def test_coefficient_interfaces_are_inside_finite_domains(self):
        coefficient = Coefficient(
            lambda x, arithmetic: arithmetic.scalar(1 if x < 1 else 2),
            lower_bound=1, upper_bound=2, wall_value=1, interfaces=(1,)
        )
        bound = self.problem(epsilon=coefficient, domain=(0, 2)).bind(MpmathArithmetic(50))
        self.assertEqual(bound.epsilon("0.5"), 1)
        self.assertEqual(bound.epsilon("1.5"), 2)
        self.assertEqual(bound.epsilon.interfaces, (1,))
        for right in ("0.5", 1):
            with self.assertRaisesRegex(ValueError, "strictly inside"):
                self.problem(epsilon=coefficient, domain=(0, right)).bind(NumpyArithmetic())


if __name__ == "__main__":
    unittest.main()
