"""Small numerical invariants for the common positive half-line quadrature."""

from dataclasses import FrozenInstanceError
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.quadrature import QuadratureFactory


class GalerkinQuadratureTests(unittest.TestCase):
    def arithmetics(self):
        return (NumpyArithmetic(),) + tuple(
            MpmathArithmetic(dps) for dps in (15, 30, 50, 100, 120)
        )

    def tolerance(self, arithmetic):
        if not arithmetic.use_mpmath:
            return arithmetic.scalar("3e-13")
        return arithmetic.scalar("1e-{}".format(max(10, arithmetic.dps - 5)))

    def test_polynomial_moments_on_composite_panels(self):
        for arithmetic in self.arithmetics():
            factory = QuadratureFactory(arithmetic, interfaces=("3",))
            for order in (1, 2, 4, 7):
                with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps,
                                  order=order):
                    plan = factory.plan(order, subdivisions=2)
                    for degree in range(2 * order):
                        actual = plan.integrate_reference(lambda y: y ** degree)
                        expected = (arithmetic.scalar(0) if degree % 2 else
                                    arithmetic.scalar(2) / (degree + 1))
                        self.assertLess(abs(actual - expected), self.tolerance(arithmetic))

    def test_all_nodes_are_interior_and_all_weights_are_positive(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                plan = QuadratureFactory(
                    arithmetic, L="0.75", interfaces=("0.25", "2.5")
                ).plan(5, subdivisions=2)
                self.assertEqual(len(plan.panels), 6)
                self.assertEqual(len(plan.nodes), 30)
                for panel_index, (left, right) in enumerate(plan.panels):
                    for node in plan.nodes[5 * panel_index:5 * (panel_index + 1)]:
                        self.assertLess(left, node)
                        self.assertLess(node, right)
                self.assertTrue(all(w > 0 for w in plan.weights))
                self.assertTrue(all(w > 0 for w in plan.physical_weights))
                self.assertTrue(all(x > 0 and arithmetic.isfinite(x)
                                    for x in plan.physical_nodes))
                self.assertTrue(all(a < b for a, b in zip(
                    plan.physical_nodes[:-1], plan.physical_nodes[1:]
                )))
                expected_type = type(arithmetic.scalar(0))
                for sequence in (plan.nodes, plan.physical_nodes,
                                 plan.weights, plan.physical_weights):
                    self.assertIsInstance(sequence, tuple)
                    self.assertTrue(all(isinstance(x, expected_type) for x in sequence))

    def test_mapped_rational_integrals_use_physical_measure(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                plan = QuadratureFactory(arithmetic).plan(5)
                for degree in range(8):
                    value = plan.integrate(
                        lambda x: x ** degree / (1 + x) ** (degree + 2)
                    )
                    expected = arithmetic.scalar(1) / (degree + 1)
                    self.assertLess(abs(value - expected), self.tolerance(arithmetic))

    def test_mapping_lengths_preserve_normalized_kernel_integral(self):
        for arithmetic in self.arithmetics():
            for raw_length in ("0.2", "1", "9.75"):
                with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps,
                                  L=raw_length):
                    plan = QuadratureFactory(arithmetic, L=raw_length).plan(3)
                    length = arithmetic.scalar(raw_length)
                    value = plan.integrate(lambda x: length / (x + length) ** 2)
                    self.assertLess(abs(value - 1), self.tolerance(arithmetic))

    def test_interface_split_integrates_jump_without_sampling_it(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                plan = QuadratureFactory(arithmetic, interfaces=(1,)).plan(2)
                visited = []

                def integrand(x):
                    self.assertNotEqual(x, 1)
                    visited.append(x)
                    coefficient = 1 if x < 1 else 2
                    return coefficient / (1 + x) ** 2

                self.assertLess(abs(plan.integrate(integrand)
                                    - arithmetic.scalar("1.5")),
                                self.tolerance(arithmetic))
                self.assertEqual(len(visited), 4)
                self.assertEqual(plan.panels, ((arithmetic.scalar(-1), arithmetic.scalar(0)),
                                              (arithmetic.scalar(0), arithmetic.scalar(1))))

    def test_high_precision_nodes_and_integrals_do_not_use_float64_seeds(self):
        arithmetic = MpmathArithmetic(120)
        plan = QuadratureFactory(arithmetic).plan(2)
        exact_node = 1 / arithmetic.sqrt(arithmetic.scalar(3))
        tolerance = arithmetic.scalar("1e-118")
        self.assertLess(abs(plan.nodes[0] + exact_node), tolerance)
        self.assertLess(abs(plan.nodes[1] - exact_node), tolerance)
        correction = arithmetic.scalar("1e-80")
        value = plan.integrate(lambda x: (1 + correction) / (1 + x) ** 2)
        self.assertGreater(value - 1, 0)
        self.assertLess(abs((value - 1) - correction), tolerance)

    def test_refinement_reports_failed_and_passed_integration_checks(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            with self.subTest(backend=arithmetic.backend):
                plan = QuadratureFactory(arithmetic).plan(2)

                def integrand(x):
                    y = (x - 1) / (x + 1)
                    return 1 / ((1 + x) ** 2 * (2 + y))

                failed = plan.check_refinement(integrand, atol="1e-12", rtol=0)
                self.assertFalse(failed.converged)
                self.assertEqual(failed.refined_plan.order, 4)
                self.assertEqual(failed.difference,
                                 abs(failed.refined_value - failed.coarse_value))
                fine = plan.refined(order_factor=8)
                passed = fine.check_refinement(integrand, atol="1e-12", rtol="1e-13")
                self.assertTrue(passed.converged)
                self.assertEqual(passed.target, arithmetic.scalar("1e-12")
                                 + arithmetic.scalar("1e-13") * abs(passed.refined_value))
                exact = arithmetic.log(arithmetic.scalar(3)) / 2
                self.assertLess(abs(passed.refined_value - exact), arithmetic.scalar("1e-13"))

    def test_reference_subdivision_refines_each_interface_panel(self):
        arithmetic = MpmathArithmetic(30)
        plan = QuadratureFactory(arithmetic, interfaces=(1, 3)).plan(2)
        refined = plan.refined(order_factor=1, subdivisions_factor=3)
        self.assertEqual(refined.order, 2)
        self.assertEqual(refined.subdivisions, 3)
        self.assertEqual(len(refined.panels), 9)
        self.assertEqual(len(refined.nodes), 18)
        for endpoint, _ in plan.panels[1:]:
            self.assertTrue(any(left == endpoint for left, _ in refined.panels))

    def test_factory_reuses_cached_plans_and_bounds_cache_size(self):
        factory = QuadratureFactory(NumpyArithmetic(), max_cached_plans=2)
        first = factory.plan(2)
        second = factory.plan(3)
        self.assertIs(factory.plan(2), first)
        factory.plan(4)
        self.assertIs(factory.plan(2), first)
        self.assertIsNot(factory.plan(3), second)
        self.assertIs(factory.plan(3), factory.plan(3))
        self.assertIsNot(factory.plan(3), factory.plan(3, subdivisions=2))

    def test_precision_change_rebuilds_raw_geometry_and_rejects_old_plan(self):
        arithmetic = MpmathArithmetic(30)
        raw_length = "1." + "0" * 79 + "1"
        factory = QuadratureFactory(arithmetic, L=raw_length, interfaces=("2",))
        old = factory.plan(3)
        self.assertEqual(old.mapping.L, 1)
        arithmetic.context.dps = 120
        new = factory.plan(3)
        self.assertIsNot(new, old)
        self.assertNotEqual(new.precision_signature, old.precision_signature)
        self.assertGreater(new.mapping.L - 1, 0)
        self.assertLess(abs((new.mapping.L - 1) - arithmetic.scalar("1e-80")),
                        arithmetic.scalar("1e-118"))
        self.assertIs(new, factory.plan(3))
        for action in (old.check_precision, lambda: old.integrate(lambda x: 1),
                       lambda: old.integrate_reference(lambda y: 1), old.refined):
            with self.assertRaises((ValueError, RuntimeError)):
                action()

    def test_cache_uses_exact_bit_precision_and_never_crosses_contexts(self):
        first = MpmathArithmetic(30)
        second = MpmathArithmetic(30)
        factory = QuadratureFactory(first)
        old = factory.plan(2)
        other = QuadratureFactory(second).plan(2)
        self.assertIsNot(old, other)
        self.assertIs(old.arithmetic, first)
        self.assertIs(other.arithmetic, second)
        previous_dps = first.dps
        first.context.prec += 1
        self.assertEqual(first.dps, previous_dps)
        self.assertIsNot(old, factory.plan(2))
        self.assertEqual(second.dps, 30)

    def test_global_precision_is_unchanged_and_cannot_change_owned_rule(self):
        previous_dps = mp.mp.dps
        arithmetic = MpmathArithmetic(100)
        factory = QuadratureFactory(arithmetic)
        plan = factory.plan(4)
        self.assertEqual(mp.mp.dps, previous_dps)
        try:
            mp.mp.dps = previous_dps + 11
            self.assertIs(factory.plan(4), plan)
            self.assertEqual(arithmetic.dps, 100)
            value = plan.integrate(lambda x: 1 / (1 + x) ** 2)
            self.assertLess(abs(value - 1), arithmetic.scalar("1e-97"))
        finally:
            mp.mp.dps = previous_dps

    def test_plan_and_raw_interface_snapshot_are_immutable(self):
        interfaces = [1]
        factory = QuadratureFactory(NumpyArithmetic(), interfaces=interfaces)
        interfaces.append(2)
        plan = factory.plan(2)
        self.assertEqual(len(plan.panels), 2)
        with self.assertRaises(FrozenInstanceError):
            plan.order = 7
        with self.assertRaises(TypeError):
            plan.nodes[0] = 0

    def test_invalid_orders_subdivisions_cache_limits_and_refinements(self):
        factory = QuadratureFactory(NumpyArithmetic())
        for value in (0, -1, True, 1.5, "2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    factory.plan(value)
                with self.assertRaises(ValueError):
                    factory.plan(2, subdivisions=value)
                with self.assertRaises(ValueError):
                    QuadratureFactory(NumpyArithmetic(), max_cached_plans=value)
                with self.assertRaises(ValueError):
                    factory.plan(2).refined(order_factor=value)
                with self.assertRaises(ValueError):
                    factory.plan(2).refined(subdivisions_factor=value)
        with self.assertRaises(ValueError):
            factory.plan(2).refined(order_factor=1, subdivisions_factor=1)

    def test_invalid_or_unresolved_physical_interfaces_are_rejected(self):
        invalid = ((0,), (-1,), (1, 1), (2, 1), ("nan",), ("inf",),
                   ("1e-100",), ("1e100",))
        for interfaces in invalid:
            with self.subTest(interfaces=interfaces):
                with self.assertRaises(ValueError):
                    QuadratureFactory(NumpyArithmetic(), interfaces=interfaces)
        with self.assertRaises(ValueError):
            QuadratureFactory(MpmathArithmetic(30), interfaces=(
                "1", "1." + "0" * 79 + "1"
            ))
        # Reference-coordinate interiority alone cannot prevent an inverse map
        # from rounding onto the actual physical interface at a narrow panel.
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(30),
                           MpmathArithmetic(120)):
            adjacent = (arithmetic.scalar(1) + arithmetic.context.eps
                        if arithmetic.use_mpmath else np.nextafter(1.0, np.inf))
            for order in (1, 2):
                with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps,
                                  order=order, adjacent_interfaces=True):
                    with self.assertRaises(ValueError):
                        QuadratureFactory(arithmetic, interfaces=(1, adjacent)).plan(order)

    def test_invalid_integrands_and_refinement_tolerances_are_rejected(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(30)):
            plan = QuadratureFactory(arithmetic).plan(2)
            for value in ("nan", "inf", "-inf"):
                with self.subTest(backend=arithmetic.backend, value=value):
                    with self.assertRaises(ValueError):
                        plan.integrate(lambda x: arithmetic.scalar(value))
            for callback in (lambda x: 1j, lambda x: np.array([1])):
                with self.assertRaises(TypeError):
                    plan.integrate(callback)
            with self.assertRaises(TypeError):
                plan.integrate(1)
            for atol, rtol in ((-1, 0), (0, -1), ("nan", 0), (0, "inf")):
                with self.assertRaises(ValueError):
                    plan.check_refinement(lambda x: 1 / (1 + x) ** 2,
                                          atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
