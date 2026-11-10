"""Declared epsilon contracts, precision preservation, and context isolation."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient, ConstantCoefficient, as_coefficient


class CoefficientTests(unittest.TestCase):
    def arithmetics(self):
        return (NumpyArithmetic(), MpmathArithmetic(dps=30), MpmathArithmetic(dps=120))

    def coefficient(self, function=None, **changes):
        metadata = dict(lower_bound=1, upper_bound=2, wall_value=1)
        metadata.update(changes)
        return Coefficient(
            function if function is not None else lambda x, arithmetic: 1,
            **metadata
        )

    def test_constant_values_and_explicit_identity(self):
        for arithmetic in self.arithmetics():
            for value in (1, "1.000", Decimal("1"), 2, "1.25"):
                with self.subTest(dps=arithmetic.dps, value=value):
                    bound = ConstantCoefficient(value).bind(arithmetic)
                    expected = arithmetic.scalar(value)
                    self.assertEqual(bound(0), expected)
                    self.assertEqual(bound("2.5"), expected)
                    self.assertEqual(bound.lower_bound, expected)
                    self.assertEqual(bound.upper_bound, expected)
                    self.assertEqual(bound.wall_value, expected)
                    self.assertEqual(bound.interfaces, ())
                    self.assertEqual(bound.is_identity, expected == 1)
            self.assertFalse(self.coefficient().bind(arithmetic).is_identity)

    def test_rounding_a_constant_to_one_does_not_make_it_identity(self):
        value = "1." + "0" * 79 + "1"
        description = ConstantCoefficient(value)
        numpy_bound = description.bind(NumpyArithmetic())
        high_bound = description.bind(MpmathArithmetic(dps=120))
        self.assertEqual(numpy_bound(1), 1)
        self.assertGreater(high_bound(1), 1)
        self.assertFalse(numpy_bound.is_identity)
        self.assertFalse(high_bound.is_identity)
        self.assertEqual(description.value, value)

    def test_metadata_is_immutable_and_interfaces_are_defensively_copied(self):
        interfaces = ["1", "2"]
        description = self.coefficient(interfaces=interfaces)
        interfaces.append("3")
        self.assertEqual(description.interfaces, ("1", "2"))
        bound = description.bind(MpmathArithmetic(dps=30))
        for target, attribute, value in (
            (description, "wall_value", 2),
            (description, "interfaces", ()),
            (bound, "wall_value", 2),
            (ConstantCoefficient(), "value", 2),
        ):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(target, attribute, value)

    def test_constant_must_be_finite_and_positive(self):
        for arithmetic in self.arithmetics():
            for value in (0, -1, "-0.5", "nan", "inf", "-inf"):
                with self.subTest(dps=arithmetic.dps, value=value):
                    with self.assertRaises(ValueError):
                        ConstantCoefficient(value).bind(arithmetic)

    def test_declared_bounds_and_wall_trace_are_validated(self):
        invalid = (
            dict(lower_bound=0),
            dict(lower_bound=-1),
            dict(lower_bound=3),
            dict(lower_bound="nan"),
            dict(upper_bound="inf"),
            dict(upper_bound="nan"),
            dict(wall_value="nan"),
            dict(wall_value="inf"),
            dict(wall_value="0.5"),
            dict(wall_value=3),
        )
        for arithmetic in self.arithmetics():
            for metadata in invalid:
                with self.subTest(dps=arithmetic.dps, metadata=metadata):
                    with self.assertRaises(ValueError):
                        self.coefficient(**metadata).bind(arithmetic)

    def test_interfaces_must_be_finite_positive_and_strictly_ordered(self):
        invalid = ((0,), (-1,), ("nan",), ("inf",), (1, 1), (2, 1))
        for arithmetic in self.arithmetics():
            for interfaces in invalid:
                with self.subTest(dps=arithmetic.dps, interfaces=interfaces):
                    with self.assertRaises(ValueError):
                        self.coefficient(interfaces=interfaces).bind(arithmetic)

    def test_unresolved_interfaces_are_rejected_at_the_bound_precision(self):
        interfaces = ["1", "1." + "0" * 79 + "1"]
        description = self.coefficient(interfaces=interfaces)
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(dps=30)):
            with self.subTest(dps=arithmetic.dps):
                with self.assertRaises(ValueError):
                    description.bind(arithmetic)
        high = description.bind(MpmathArithmetic(dps=120))
        self.assertGreater(high.interfaces[1], high.interfaces[0])
        self.assertEqual(description.interfaces, tuple(interfaces))

    def test_each_callback_value_must_satisfy_declared_bounds(self):
        for arithmetic in self.arithmetics():
            for value in ("nan", "inf", 0, "0.5", 3):
                with self.subTest(dps=arithmetic.dps, value=value):
                    bound = self.coefficient(
                        lambda x, context, result=value: context.scalar(result)
                    ).bind(arithmetic)
                    with self.assertRaises(ValueError):
                        bound(1)
            # Both ends of the declared interval are valid.
            bound = self.coefficient(
                lambda x, context: context.scalar(1) if x < 1 else context.scalar(2)
            ).bind(arithmetic)
            self.assertEqual(bound("0.5"), 1)
            self.assertEqual(bound(1), 2)

    def test_invalid_coordinates_do_not_call_the_callback(self):
        for arithmetic in self.arithmetics():
            calls = []
            bound = self.coefficient(
                lambda x, context: calls.append(x) or context.scalar(1)
            ).bind(arithmetic)
            for coordinate in (-1, "-0.1", "nan", "inf", "-inf"):
                with self.subTest(dps=arithmetic.dps, coordinate=coordinate):
                    with self.assertRaises(ValueError):
                        bound(coordinate)
            self.assertEqual(calls, [])

    def test_zero_uses_the_declared_right_trace(self):
        for arithmetic in self.arithmetics():
            calls = []

            def function(x, context):
                calls.append(x)
                return context.scalar(2)

            bound = self.coefficient(function, wall_value="1.25").bind(arithmetic)
            self.assertEqual(bound(0), arithmetic.scalar("1.25"))
            self.assertEqual(calls, [])
            self.assertEqual(bound(1), 2)
            self.assertEqual(len(calls), 1)

    def test_bound_contexts_remain_independent_of_global_mpmath_precision(self):
        original_dps = mp.mp.dps
        self.addCleanup(setattr, mp.mp, "dps", original_dps)
        mp.mp.dps = 15
        seen = []

        def function(x, arithmetic):
            seen.append(arithmetic)
            return arithmetic.scalar(1) + arithmetic.scalar("1e-80") * x / 3

        description = self.coefficient(
            function, wall_value="1.125", interfaces=("0.1",)
        )
        low_arithmetic = MpmathArithmetic(dps=30)
        high_arithmetic = MpmathArithmetic(dps=120)
        low = description.bind(low_arithmetic)
        high = description.bind(high_arithmetic)
        mp.mp.dps = 47
        self.assertEqual(low(3), 1)
        actual = high(3)
        self.assertGreater(actual, 1)
        expected = high_arithmetic.scalar(1) + high_arithmetic.scalar("1e-80")
        self.assertLess(abs(actual - expected), high_arithmetic.scalar("1e-115"))
        self.assertEqual(seen, [low_arithmetic, high_arithmetic])
        self.assertEqual(low_arithmetic.dps, 30)
        self.assertEqual(high_arithmetic.dps, 120)
        self.assertEqual(mp.mp.dps, 47)
        self.assertEqual(high.interfaces[0], high_arithmetic.scalar("0.1"))
        self.assertEqual(description.interfaces, ("0.1",))

    def test_one_argument_callbacks_are_explicitly_adapted(self):
        for arithmetic in self.arithmetics():
            description = Coefficient.from_function(
                lambda x: 1 + x / (1 + x),
                lower_bound=1,
                upper_bound=2,
                wall_value=1,
            )
            bound = description.bind(arithmetic)
            self.assertEqual(bound(1), arithmetic.scalar("1.5"))
            self.assertFalse(bound.is_identity)

    def test_callback_errors_propagate_without_signature_guessing(self):
        for error in (TypeError("callback failure"), RuntimeError("callback failure")):
            for adapted in (False, True):
                with self.subTest(error=type(error), adapted=adapted):
                    def function(*arguments):
                        raise error

                    factory = Coefficient.from_function if adapted else Coefficient
                    bound = factory(
                        function, lower_bound=1, upper_bound=2, wall_value=1
                    ).bind(NumpyArithmetic())
                    with self.assertRaises(type(error)) as caught:
                        bound(1)
                    self.assertIs(caught.exception, error)

    def test_normalization_requires_explicit_callback_metadata(self):
        for value in (1, "1", np.float64(2), Decimal("1.25"), mp.mpf(2)):
            with self.subTest(value=value):
                normalized = as_coefficient(value)
                self.assertIsInstance(normalized, ConstantCoefficient)
                self.assertEqual(normalized.value, value)
        for description in (ConstantCoefficient(), self.coefficient()):
            self.assertIs(as_coefficient(description), description)
        with self.assertRaisesRegex(TypeError, "right wall trace"):
            as_coefficient(lambda x: 1)
        for value in ([], {}, object()):
            with self.subTest(value=type(value)):
                with self.assertRaises(TypeError):
                    as_coefficient(value)
        with self.assertRaises(TypeError):
            Coefficient(1, lower_bound=1, upper_bound=2, wall_value=1)


if __name__ == "__main__":
    unittest.main()
