"""Physical-to-collocation normalization and scalar/array bridge checks."""

from dataclasses import FrozenInstanceError
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.collocation.adapter import CollocationProblemAdapter, prepare_collocation_problem
from overscreening_breakdown.problems.flux import FluxProblem


class CollocationAdapterTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for precision in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(precision)

    def description(self, **options):
        arguments = dict(
            delta=2,
            f=lambda x, u, a: a.exp(-x) + u**3,
            f_u=lambda x, u, a: 3*u**2,
            boundary_conditions=((1, None, None, 0, None), (None,)*5),
        )
        arguments.update(options)
        return FluxProblem(**arguments)

    def adapter(self, arithmetic, **options):
        return CollocationProblemAdapter(self.description(**options).bind(arithmetic))

    def test_normalization_and_homotopy_in_all_precisions(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                adapter = self.adapter(arithmetic)
                x, u, q = map(arithmetic.scalar, ("0.5", "0.25", "0.3"))
                scale = arithmetic.scalar(1)/4
                damping = arithmetic.exp(-q*x)
                self.assertEqual(adapter.left_hand_side(), [None, None, scale, None, -1])
                self.assertEqual(adapter.right_hand_side(x, u, q), scale*damping*(arithmetic.exp(-x)+u**3))
                self.assertEqual(adapter.right_hand_side_linearization(x, u, q), scale*damping*(3*u**2))
                self.assertEqual(adapter.physical.delta, 2)

    def test_second_order_problem_is_not_divided_by_delta(self):
        boundaries = ((1, None, None, None, None), (None,)*5)
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                adapter = self.adapter(arithmetic, delta=0, boundary_conditions=boundaries)
                self.assertEqual(adapter.left_hand_side(), [None, None, 1, None, 0])
                self.assertEqual(adapter.right_hand_side(0, 2), 9)
                self.assertEqual(adapter.right_hand_side_linearization(0, 2), 12)
                self.assertEqual(adapter.right_hand_side(1, 0, 1), arithmetic.exp(-arithmetic.scalar(1))**2)

    def test_scalar_callbacks_are_broadcast_without_dtype_inference(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            calls = []

            def source(x, u, context):
                self.assertEqual(np.ndim(x), 0)
                self.assertEqual(np.ndim(u), 0)
                self.assertIs(context, arithmetic)
                calls.append((x, u))
                return x + 2*u

            adapter = self.adapter(arithmetic, f=source, f_u=lambda x, u, a: a.scalar(2))
            x = np.array([["0.25"], ["0.75"]], dtype=object)
            u = np.array([[1, 2, 3]], dtype=object)
            result = adapter.right_hand_side(x, u)
            derivative = adapter.right_hand_side_linearization(x, u)
            self.assertEqual(result.shape, (2, 3))
            self.assertEqual(derivative.shape, (2, 3))
            self.assertEqual(len(calls), 6)
            self.assertEqual(result.dtype, object if arithmetic.use_mpmath else np.dtype("float64"))
            self.assertEqual(derivative.dtype, result.dtype)
            for row in range(2):
                for column in range(3):
                    self.assertEqual(result[row, column], (arithmetic.scalar(x[row, 0])+2*(column+1))/4)
                    self.assertEqual(derivative[row, column], arithmetic.scalar("0.5"))
                    if arithmetic.use_mpmath:
                        self.assertIsInstance(result[row, column], arithmetic.context.mpf)

    def test_empty_arrays_preserve_shape_and_do_not_evaluate_callbacks(self):
        def forbidden(*arguments):
            raise AssertionError("empty arrays must not evaluate a callback")

        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            adapter = self.adapter(arithmetic, f=forbidden, f_u=forbidden, initial_guess=forbidden)
            empty = np.empty((2, 0))
            for result in (
                adapter.right_hand_side(empty, 0),
                adapter.right_hand_side_linearization(empty, 0),
                adapter.base_solution(empty),
            ):
                self.assertEqual(result.shape, (2, 0))
                self.assertEqual(result.dtype, object if arithmetic.use_mpmath else np.dtype("float64"))

    def test_initial_guess_bridge_returns_native_scalars_and_arrays(self):
        for arithmetic in self.arithmetics():
            adapter = self.adapter(arithmetic, initial_guess=lambda x, a: a.scalar(3)/(1+x))
            scalar = adapter.base_solution("0.5")
            array = adapter.base_solution(np.array([0, "0.5", 1], dtype=object))
            self.assertEqual(scalar, 2)
            self.assertEqual(array.shape, (3,))
            self.assertEqual(array.tolist(), [3, 2, arithmetic.scalar("1.5")])
            self.assertIsInstance(scalar, arithmetic.context.mpf if arithmetic.use_mpmath else np.float64)
            source = adapter.right_hand_side("0.5", "0.25")
            self.assertIsInstance(source, arithmetic.context.mpf if arithmetic.use_mpmath else np.float64)

    def test_homotopy_parameter_validation(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            adapter = self.adapter(arithmetic)
            for invalid in (-1, "nan", "inf", "-inf"):
                with self.subTest(backend=arithmetic.backend, invalid=invalid):
                    for callback in (adapter.right_hand_side, adapter.right_hand_side_linearization):
                        with self.assertRaisesRegex(ValueError, "q must be finite and nonnegative"):
                            callback(1, 2, invalid)
            for invalid in ([1], 1j):
                with self.assertRaises(TypeError):
                    adapter.right_hand_side(1, 2, invalid)

    def test_nonidentity_coefficients_are_rejected_before_rounding(self):
        declared = Coefficient(lambda x, a: a.scalar(1), lower_bound=1, upper_bound=1, wall_value=1)
        for coefficient in (2, "1." + "0"*79 + "1", declared):
            for arithmetic in (NumpyArithmetic(), MpmathArithmetic(120)):
                with self.subTest(coefficient=coefficient, backend=arithmetic.backend):
                    with self.assertRaisesRegex(ValueError, "explicitly constant epsilon=1"):
                        self.adapter(arithmetic, epsilon=coefficient)
        self.adapter(NumpyArithmetic(), epsilon="1.000")

    def test_half_line_boundary_counts_and_far_field_scope(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            for delta, left in ((0, (1, None, None, None, None)), (2, (1, None, None, 0, None))):
                for right in ((None,)*5, (0, None, None, None, None)):
                    self.adapter(arithmetic, delta=delta, boundary_conditions=(left, right))
            for boundaries in (
                ((None,)*5, (None,)*5),
                ((1, 0, 0, None, None), (None,)*5),
            ):
                with self.assertRaisesRegex(ValueError, "exactly 2 wall"):
                    self.adapter(arithmetic, boundary_conditions=boundaries)
            with self.assertRaisesRegex(ValueError, "zero far-field"):
                self.adapter(arithmetic, boundary_conditions=((1, None, None, 0, None), (1, None, None, None, None)))
            with self.assertRaisesRegex(ValueError, "at infinity"):
                self.adapter(arithmetic, boundary_conditions=((1, None, None, 0, None), (None, 0, None, None, None)))

    def test_finite_interval_counts_and_boundary_derivative_orders(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            self.adapter(arithmetic, domain=(0, 2), boundary_conditions=((1, 0, None, None, None), (0, 0, None, None, None)))
            self.adapter(arithmetic, delta=0, domain=(0, 2), boundary_conditions=((1, None, None, None, None), (0, None, None, None, None)))
            with self.assertRaisesRegex(ValueError, "exactly 4 boundary"):
                self.adapter(arithmetic, domain=(0, 2))
            with self.assertRaisesRegex(ValueError, "orders below 2"):
                self.adapter(arithmetic, delta=0)
            with self.assertRaisesRegex(ValueError, "orders below 4"):
                self.adapter(arithmetic, boundary_conditions=((1, None, None, None, 0), (None,)*5))

    def test_getters_copy_metadata_and_adapter_is_frozen(self):
        adapter = self.adapter(MpmathArithmetic(50))
        domain = adapter.get_domain()
        boundaries = adapter.get_boundary_conditions()
        domain[1] = 3
        boundaries[0][0] = 9
        boundaries[1].append(7)
        self.assertEqual(adapter.physical.domain[1], adapter.arithmetic.inf)
        self.assertEqual(adapter.get_boundary_conditions()[0][0], 1)
        self.assertEqual(len(adapter.get_boundary_conditions()[1]), 5)
        with self.assertRaises(FrozenInstanceError):
            adapter.physical = None

    def test_raw_description_is_bound_without_mutation_or_global_precision_change(self):
        description = self.description(delta="2.125")
        precision = mp.mp.prec
        try:
            mp.mp.dps = 17
            adapter = prepare_collocation_problem(description, use_mpmath=True, prec=120)
            self.assertEqual(mp.mp.dps, 17)
            self.assertEqual(description.delta, "2.125")
            self.assertIsInstance(adapter.arithmetic, MpmathArithmetic)
            self.assertEqual(adapter.arithmetic.dps, 120)
            self.assertEqual(adapter.physical.delta, adapter.arithmetic.scalar("2.125"))
        finally:
            mp.mp.prec = precision
        numpy_adapter = prepare_collocation_problem(description)
        self.assertIsInstance(numpy_adapter.arithmetic, NumpyArithmetic)

    def test_binding_rejects_prebound_views_and_incompatible_adapter_reuse(self):
        bound = self.description().bind(MpmathArithmetic(50))
        with self.assertRaisesRegex(TypeError, "unbound FluxProblem"):
            prepare_collocation_problem(bound, use_mpmath=True, prec=50)
        adapter = CollocationProblemAdapter(bound)
        self.assertIs(prepare_collocation_problem(adapter, use_mpmath=True, prec=50), adapter)
        for options in (dict(use_mpmath=False), dict(use_mpmath=True, prec=100)):
            with self.assertRaisesRegex(TypeError, "does not match"):
                prepare_collocation_problem(adapter, **options)
        numpy_adapter = self.adapter(NumpyArithmetic())
        self.assertIs(prepare_collocation_problem(numpy_adapter, prec=120), numpy_adapter)
        legacy = object()
        self.assertIs(prepare_collocation_problem(legacy), legacy)
        with self.assertRaisesRegex(TypeError, "BoundFluxProblem"):
            CollocationProblemAdapter(self.description())

    def test_120_digit_scalar_and_array_evaluation_preserve_small_parameter(self):
        arithmetic = MpmathArithmetic(120)
        adapter = self.adapter(
            arithmetic, delta="1." + "0"*79 + "1",
            f=lambda x, u, a: a.scalar(1), f_u=lambda x, u, a: a.scalar(1),
        )
        precision = mp.mp.prec
        try:
            mp.mp.dps = 17
            scalar = adapter.right_hand_side(1, 0)
            array = adapter.right_hand_side(np.array([1, 2]), 0)
            expected = arithmetic.scalar(1)/adapter.physical.delta**2
            self.assertEqual(scalar, expected)
            self.assertEqual(array.tolist(), [expected, expected])
            self.assertLess(scalar, 1)
            self.assertLess(abs((1-scalar)-arithmetic.scalar("2e-80")), arithmetic.scalar("1e-118"))
            self.assertIsInstance(scalar, arithmetic.context.mpf)
        finally:
            mp.mp.prec = precision

    def test_callback_errors_and_nonfinite_sources_propagate(self):
        def broken(*arguments):
            raise RuntimeError("source callback defect")

        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            adapter = self.adapter(arithmetic, f=broken, f_u=broken, initial_guess=broken)
            for callback, arguments in (
                (adapter.right_hand_side, (1, 2)),
                (adapter.right_hand_side_linearization, (1, 2)),
                (adapter.base_solution, (1,)),
            ):
                with self.assertRaisesRegex(RuntimeError, "source callback defect"):
                    callback(*arguments)
            invalid = self.adapter(arithmetic, f=lambda x, u, a: a.inf)
            with self.assertRaisesRegex(OverflowError, "physical source value must be finite"):
                invalid.right_hand_side(1, 2)


if __name__ == "__main__":
    unittest.main()
