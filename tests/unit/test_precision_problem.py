"""Bound physical data require the owner's recorded bit precision."""

from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import mpmath as mp
import numpy as np

from overscreening_breakdown import BSKProblem, FluxProblem
from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient, ConstantCoefficient
from overscreening_breakdown.collocation.adapter import (
    CollocationProblemAdapter, prepare_collocation_problem,
)


def problem(**options):
    arguments = dict(delta=2, f=lambda x, u, a: u+x,
                     f_u=lambda x, u, a: a.scalar(1))
    arguments.update(options)
    return FluxProblem(**arguments)


class PhysicalPrecisionTests(unittest.TestCase):
    def assert_stale(self, operation, *arguments):
        with self.assertRaisesRegex(RuntimeError, r"precision changed.*bits.*bits"):
            operation(*arguments)

    def test_problem_rejects_upward_downward_and_one_bit_changes_before_callbacks(self):
        for digits in (15, 30, 50, 100, 120):
            for change in (-1, 1, 31):
                a = MpmathArithmetic(digits)
                source, derivative, guess = (Mock(return_value=0) for _ in range(3))
                bound = problem(f=source, f_u=derivative, initial_guess=guess).bind(a)
                bits = a.context.prec
                a.context.prec += change
                with self.subTest(digits=digits, change=change):
                    with patch.object(a, "scalar", wraps=a.scalar) as convert:
                        self.assert_stale(bound.physical_rhs, 0, 1)
                        self.assert_stale(bound.physical_rhs_u, 0, 1)
                        self.assert_stale(bound.base_solution, 0)
                        self.assert_stale(bound.get_domain)
                        self.assert_stale(bound.get_boundary_conditions)
                        convert.assert_not_called()
                    for callback in (source, derivative, guess):
                        callback.assert_not_called()
                a.context.prec = bits
                self.assertEqual(bound.physical_rhs(0, 1), 0)

    def test_one_bit_change_is_detected_when_decimal_digits_do_not_change(self):
        a = MpmathArithmetic(50)
        bound = BSKProblem().bind(a)
        digits = a.dps
        a.context.prec += 1
        self.assertEqual(a.dps, digits)
        for callback in (bound.physical_rhs, bound.physical_rhs_u):
            self.assert_stale(callback, 1, 1)
        self.assert_stale(bound.base_solution, 1)

    def test_default_initial_guess_rejects_stale_parameters(self):
        a = MpmathArithmetic(50)
        bound = problem().bind(a)
        a.context.prec -= 1
        self.assert_stale(bound.base_solution, 0)

    def test_coefficient_constants_wall_trace_and_callback_are_all_guarded(self):
        for digits in (15, 50, 120):
            for kind in ("constant", "function"):
                a = MpmathArithmetic(digits)
                callback = Mock(return_value=1)
                description = (ConstantCoefficient("1.25") if kind == "constant" else
                               Coefficient(callback, lower_bound=1, upper_bound=2,
                                           wall_value="1.25", interfaces=("0.125",)))
                coefficient = description.bind(a)
                a.context.prec += 1
                with self.subTest(digits=digits, kind=kind):
                    for x in (0, 1):
                        self.assert_stale(coefficient, x)
                    callback.assert_not_called()

    def test_source_derivative_and_guess_reject_callback_precision_changes(self):
        for name, arguments in (("physical_rhs", (0, 1)), ("physical_rhs_u", (0, 1)),
                                ("base_solution", (0,))):
            a = MpmathArithmetic(50)

            def change_precision(*arguments):
                a.context.prec += 1
                return "nan"  # Precision failure must precede output conversion.

            bound = problem(f=change_precision, f_u=change_precision,
                            initial_guess=change_precision).bind(a)
            with self.subTest(method=name):
                self.assert_stale(getattr(bound, name), *arguments)

    def test_coefficient_rejects_callback_precision_changes_before_conversion(self):
        a = MpmathArithmetic(50)

        def change_precision(x, owner):
            owner.context.prec += 1
            return "nan"

        bound = Coefficient(change_precision, lower_bound=1, upper_bound=2,
                            wall_value=1).bind(a)
        self.assert_stale(bound, 1)

    def test_original_descriptions_rebind_decimal_parameters_at_new_precision(self):
        a = MpmathArithmetic(30)
        raw = "1."+"0"*79+"1"
        description = BSKProblem(delta=raw, sigma=raw, epsilon=raw,
                                 boundary_conditions=((raw, None, None, 0, None), (None,)*5))
        low = description.bind(a)
        self.assertEqual(low.delta, 1)
        self.assertEqual(low.epsilon(1), 1)
        a.context.dps = 120
        self.assert_stale(low.physical_rhs, 0, 0)
        self.assert_stale(low.epsilon, 1)
        high = description.bind(a)
        self.assertEqual(high.delta, a.scalar(raw))
        self.assertGreater(high.delta, 1)
        self.assertEqual(high.epsilon(1), a.scalar(raw))
        self.assertEqual(high.base_solution(0), a.scalar(raw))
        self.assertTrue(a.isfinite(high.physical_rhs(1, 1)))

    def test_problem_and_coefficient_must_share_one_valid_owner(self):
        first, second = MpmathArithmetic(50), MpmathArithmetic(50)
        bound = problem().bind(first)
        with self.assertRaisesRegex(ValueError, "share one arithmetic owner"):
            replace(bound, epsilon=ConstantCoefficient(1).bind(second))
        first.context.prec += 1
        with self.assertRaisesRegex(RuntimeError, "coefficient precision changed"):
            replace(bound)

    def test_binding_detects_precision_change_during_custom_coefficient_binding(self):
        class ChangingCoefficient(ConstantCoefficient):
            def bind(self, arithmetic):
                arithmetic.context.prec += 1
                return super().bind(arithmetic)

        a = MpmathArithmetic(50)
        self.assert_stale(problem(epsilon=ChangingCoefficient()).bind, a)

    def test_owner_mutation_does_not_change_other_contexts_or_global_precision(self):
        old_bits = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", old_bits)
        first, second = MpmathArithmetic(30), MpmathArithmetic(120)
        low, high = problem().bind(first), problem().bind(second)
        original = high.physical_rhs("0.1", "0.2")
        first.context.prec += 1
        self.assert_stale(low.physical_rhs, 0, 1)
        self.assertEqual(mp.mp.prec, old_bits)
        self.assertEqual(second.dps, 120)
        self.assertEqual(high.physical_rhs("0.1", "0.2"), original)
        mp.mp.prec += 19
        self.assertEqual(high.physical_rhs("0.1", "0.2"), original)

    def test_numpy_evaluations_and_empty_adapter_arrays_keep_native_types(self):
        a = NumpyArithmetic()
        bound = problem().bind(a)
        adapter = CollocationProblemAdapter(bound)
        self.assertIsInstance(bound.physical_rhs(1, 2), np.float64)
        self.assertIsInstance(bound.physical_rhs_u(1, 2), np.float64)
        self.assertIsInstance(bound.base_solution(1), np.float64)
        self.assertIsInstance(bound.epsilon(0), np.float64)
        self.assertEqual(adapter.base_solution(np.empty((0, 2))).shape, (0, 2))
        self.assertEqual(adapter.base_solution(np.empty((0, 2))).dtype, np.dtype("float64"))


class AdapterPrecisionTests(unittest.TestCase):
    def test_adapter_guards_validation_normalization_metadata_and_empty_arrays(self):
        a = MpmathArithmetic(50)
        source, derivative, guess = (Mock(return_value=0) for _ in range(3))
        physical = problem(f=source, f_u=derivative, initial_guess=guess).bind(a)
        adapter = CollocationProblemAdapter(physical)
        a.context.prec += 1
        operations = (adapter.check_precision, adapter.validate_collocation_compatibility,
                      adapter.get_domain, adapter.get_boundary_conditions, adapter.left_hand_side,
                      lambda: adapter.base_solution(np.empty((0, 2))),
                      lambda: adapter.right_hand_side([], [], 0),
                      lambda: adapter.right_hand_side_linearization([], [], 0),
                      lambda: CollocationProblemAdapter(physical),
                      lambda: prepare_collocation_problem(adapter, use_mpmath=True, prec=50))
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, "precision changed"):
                    operation()
        for callback in (source, derivative, guess):
            callback.assert_not_called()

    def test_adapter_requested_precision_uses_bits_even_for_valid_new_snapshots(self):
        a = MpmathArithmetic(50)
        a.context.prec += 1
        self.assertEqual(a.dps, 50)
        adapter = CollocationProblemAdapter(problem().bind(a))
        adapter.check_precision()
        with self.assertRaisesRegex(TypeError, "arithmetic does not match"):
            prepare_collocation_problem(adapter, use_mpmath=True, prec=50)

        ordinary = CollocationProblemAdapter(problem().bind(MpmathArithmetic(50)))
        self.assertIs(prepare_collocation_problem(ordinary, use_mpmath=True, prec=50), ordinary)


if __name__ == "__main__":
    unittest.main()
