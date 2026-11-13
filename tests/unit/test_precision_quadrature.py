"""Quadrature and physical diagnostics reject precision changes in callbacks."""

import unittest

import mpmath as mp

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.norms import (
    physical_h1_error, physical_h1_norm, physical_l2_error, physical_l2_norm,
)
from overscreening_breakdown.quadrature import QuadratureFactory


class PrecisionQuadratureTests(unittest.TestCase):
    def factory(self, arithmetic, finite):
        return QuadratureFactory(arithmetic, domain=(0, 2) if finite else None)

    def test_one_bit_change_rejects_old_plans_before_any_integrand_call(self):
        for finite in (False, True):
            for change in (-1, 1):
                with self.subTest(finite=finite, change=change):
                    arithmetic = MpmathArithmetic(50)
                    plan = self.factory(arithmetic, finite).plan(4)
                    digits = arithmetic.dps
                    arithmetic.context.prec += change
                    if change == 1:
                        self.assertEqual(arithmetic.dps, digits)
                    calls = []
                    callback = lambda x: calls.append(x) or 1
                    operations = (
                        lambda: plan.integrate(callback),
                        lambda: plan.integrate_reference(callback),
                        lambda: plan.check_refinement(callback, atol=1, rtol=0),
                        plan.refined,
                    )
                    for operation in operations:
                        with self.assertRaisesRegex(RuntimeError, "precision changed.*bits"):
                            operation()
                    self.assertEqual(calls, [])

    def test_integrand_mutation_is_rejected_before_conversion_or_another_node(self):
        for finite in (False, True):
            for method in ("integrate", "integrate_reference", "check_refinement"):
                with self.subTest(finite=finite, method=method):
                    arithmetic = MpmathArithmetic(50)
                    plan = self.factory(arithmetic, finite).plan(4)
                    bits = arithmetic.context.prec
                    calls = []

                    def callback(x):
                        calls.append(x)
                        arithmetic.context.prec += 1
                        # This cannot be converted: precision must be checked
                        # before the return value reaches native arithmetic.
                        return object()

                    options = dict(atol=1, rtol=0) if method == "check_refinement" else {}
                    with self.assertRaisesRegex(RuntimeError, "precision changed.*bits"):
                        getattr(plan, method)(callback, **options)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(arithmetic.context.prec, bits+1)

    def test_stale_norm_plans_reject_all_fields_and_references_before_callbacks(self):
        for finite in (False, True):
            arithmetic = MpmathArithmetic(50)
            plan = self.factory(arithmetic, finite).plan(4)
            arithmetic.context.prec += 1
            calls = []
            function = lambda x: calls.append(x) or 1
            operations = (
                lambda: physical_l2_norm(function, plan),
                lambda: physical_h1_norm(function, function, plan),
                lambda: physical_l2_error(function, function, plan),
                lambda: physical_h1_error(function, function, function, function, plan),
            )
            for operation in operations:
                with self.subTest(finite=finite, operation=operation):
                    with self.assertRaisesRegex(RuntimeError, "precision changed.*bits"):
                        operation()
            self.assertEqual(calls, [])

    def test_norm_callback_changes_stop_before_later_fields_or_reference_calls(self):
        # H1 error invokes field, reference, derivative, reference derivative
        # at each node. Each individual boundary must reject a changed owner.
        for finite in (False, True):
            for changed in range(4):
                with self.subTest(finite=finite, changed_callback=changed):
                    arithmetic = MpmathArithmetic(50)
                    plan = self.factory(arithmetic, finite).plan(4)
                    calls = []

                    def callback(index):
                        def evaluate(x):
                            calls.append(index)
                            if index == changed:
                                arithmetic.context.prec += 1
                                return object()
                            return arithmetic.scalar(1)
                        return evaluate

                    with self.assertRaisesRegex(RuntimeError, "precision changed.*bits"):
                        physical_h1_error(callback(0), callback(2), callback(1), callback(3), plan)
                    self.assertEqual(calls, list(range(changed+1)))

    def test_norm_value_and_derivative_callbacks_each_check_returned_precision(self):
        for finite in (False, True):
            for derivative in (False, True):
                with self.subTest(finite=finite, derivative=derivative):
                    arithmetic = MpmathArithmetic(50)
                    plan = self.factory(arithmetic, finite).plan(4)
                    calls = []

                    def changed(x):
                        calls.append("changed")
                        arithmetic.context.prec += 1
                        return object()

                    def field(x):
                        calls.append("field")
                        return arithmetic.scalar(1)

                    with self.assertRaisesRegex(RuntimeError, "precision changed.*bits"):
                        if derivative:
                            physical_h1_norm(field, changed, plan)
                        else:
                            physical_l2_norm(changed, plan)
                    self.assertEqual(calls, ["field", "changed"] if derivative else ["changed"])

    def test_scoped_extra_precision_inside_callback_preserves_valid_plan(self):
        for finite in (False, True):
            for digits in (15, 30, 50, 100, 120):
                with self.subTest(finite=finite, digits=digits):
                    arithmetic = MpmathArithmetic(digits)
                    plan = self.factory(arithmetic, finite).plan(4)
                    bits = arithmetic.context.prec

                    def polynomial(y):
                        with arithmetic.context.workprec(bits+30):
                            return y*y

                    result = plan.integrate_reference(polynomial)
                    self.assertIsInstance(result, type(arithmetic.scalar(0)))
                    self.assertEqual(arithmetic.context.prec, bits)
                    tolerance = arithmetic.scalar(10)**(-digits+4)
                    self.assertLess(abs(result-arithmetic.scalar(2)/3), tolerance)

                    def physical_field(x):
                        with arithmetic.context.workprec(bits+30):
                            return arithmetic.scalar(1) if finite else 1/(1+x)

                    norm = physical_l2_norm(physical_field, plan)
                    expected = arithmetic.sqrt(arithmetic.scalar(2)) if finite else arithmetic.scalar(1)
                    self.assertLess(abs(norm-expected), tolerance)
                    self.assertEqual(arithmetic.context.prec, bits)
                    plan.check_precision()

    def test_factory_rebuild_recovers_raw_geometry_and_keeps_old_plan_stale(self):
        tiny = "0."+"0"*79+"1"
        for finite in (False, True):
            with self.subTest(finite=finite):
                arithmetic = MpmathArithmetic(30)
                length = "1."+"0"*79+"1"
                end = "2."+"0"*79+"1"
                interface = "0.5"+"0"*78+"1"
                factory = QuadratureFactory(arithmetic, L=length, interfaces=(interface,),
                                            domain=(0, end) if finite else None)
                old = factory.plan(4)
                arithmetic.context.dps = 120
                fresh = factory.plan(4)
                self.assertIsNot(old, fresh)
                self.assertIs(factory.plan(4), fresh)
                fresh.check_precision()
                recovered = fresh.mapping.right-2 if finite else fresh.mapping.L-1
                self.assertGreater(recovered, 0)
                self.assertLess(abs(recovered-arithmetic.scalar(tiny)), arithmetic.scalar("1e-118"))
                recovered_interface = fresh.mapping.to_physical(fresh.panels[0][1])
                self.assertGreater(recovered_interface-arithmetic.scalar("0.5"), 0)
                with self.assertRaisesRegex(RuntimeError, "precision changed"):
                    old.integrate_reference(lambda y: 1)
                self.assertLess(abs(fresh.integrate_reference(lambda y: y*y)
                                    -arithmetic.scalar(2)/3), arithmetic.scalar("1e-117"))

    def test_context_changes_are_local_and_global_precision_is_untouched(self):
        original_bits = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_bits)
        low, high = MpmathArithmetic(30), MpmathArithmetic(120)
        low_plan, high_plan = QuadratureFactory(low).plan(4), QuadratureFactory(high).plan(4)
        expected = high_plan.integrate_reference(lambda y: y*y)
        mp.mp.prec += 7
        self.assertEqual(high_plan.integrate_reference(lambda y: y*y), expected)
        low.context.prec += 1
        with self.assertRaisesRegex(RuntimeError, "precision changed"):
            low_plan.integrate_reference(lambda y: y*y)
        self.assertEqual(high_plan.integrate_reference(lambda y: y*y), expected)
        self.assertEqual(mp.mp.prec, original_bits+7)

    def test_native_norms_and_errors_retain_analytic_values_in_both_geometries(self):
        arithmetics = (NumpyArithmetic(),)+tuple(MpmathArithmetic(dps) for dps in (15, 30, 50, 100, 120))
        for arithmetic in arithmetics:
            for finite in (False, True):
                with self.subTest(backend=arithmetic.backend, digits=arithmetic.dps, finite=finite):
                    plan = self.factory(arithmetic, finite).plan(4)
                    if finite:
                        function = lambda x: 1+x
                        derivative = lambda x: arithmetic.scalar(1)
                        squared_l2, squared_h1 = arithmetic.scalar(26)/3, arithmetic.scalar(32)/3
                    else:
                        function = lambda x: 1/(1+x)
                        derivative = lambda x: -1/(1+x)**2
                        squared_l2, squared_h1 = arithmetic.scalar(1), arithmetic.scalar(4)/3
                    tolerance = arithmetic.scalar("1e-13") if not arithmetic.use_mpmath else arithmetic.scalar(10)**(-arithmetic.dps+4)
                    l2, h1 = physical_l2_norm(function, plan), physical_h1_norm(function, derivative, plan)
                    self.assertIsInstance(l2, type(arithmetic.scalar(0)))
                    self.assertIsInstance(h1, type(arithmetic.scalar(0)))
                    self.assertLess(abs(l2-arithmetic.sqrt(squared_l2)), tolerance)
                    self.assertLess(abs(h1-arithmetic.sqrt(squared_h1)), tolerance)
                    self.assertEqual(physical_l2_error(function, function, plan), 0)
                    self.assertEqual(physical_h1_error(function, derivative, function, derivative, plan), 0)


if __name__ == "__main__":
    unittest.main()
