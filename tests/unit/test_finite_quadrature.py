"""Physical finite-interval rules share native Gauss machinery with the half-line."""

import unittest

import mpmath as mp

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.quadrature import QuadratureFactory


class FiniteQuadratureTests(unittest.TestCase):
    def arithmetics(self):
        return (NumpyArithmetic(),)+tuple(MpmathArithmetic(dps) for dps in (15, 30, 50, 100, 120))

    def tolerance(self, arithmetic):
        return arithmetic.scalar("1e-{}".format(max(11, arithmetic.dps-4)))

    def test_positive_split_rules_integrate_physical_polynomial_moments(self):
        for a in self.arithmetics():
            for domain in (("0", "5"), ("-4", "7")):
                with self.subTest(backend=a.backend, digits=a.dps, domain=domain):
                    left, right = (a.scalar(value) for value in domain)
                    plan = QuadratureFactory(a, domain=domain, interfaces=("1", "2")).plan(4, 2)
                    self.assertEqual(plan.mapping.mapping_type, "affine_interval")
                    self.assertEqual(plan.domain, (left, right))
                    self.assertEqual(len(plan.panels), 6)
                    self.assertTrue(all(left < x < right for x in plan.physical_nodes))
                    self.assertTrue(all(x not in (1, 2) for x in plan.physical_nodes))
                    self.assertTrue(all(weight > 0 for weight in plan.physical_weights))
                    for degree in range(8):
                        exact = (right**(degree+1)-left**(degree+1))/(degree+1)
                        actual = plan.integrate(lambda x: x**degree)
                        self.assertLess(abs(actual-exact), self.tolerance(a)*max(a.scalar(1), abs(exact)))

    def test_jump_integration_never_samples_interfaces_or_outside_the_domain(self):
        for a in self.arithmetics():
            plan = QuadratureFactory(a, domain=(0, 5), interfaces=(1, 3)).plan(3)

            def callback(x):
                self.assertTrue(0 < x < 5)
                self.assertNotIn(x, (1, 3))
                return a.scalar(1 if x < 1 else 2 if x < 3 else 4)

            self.assertLess(abs(plan.integrate(callback)-13), self.tolerance(a))

    def test_refinement_and_caches_retain_the_finite_geometry(self):
        a = MpmathArithmetic(50)
        factory = QuadratureFactory(a, domain=(0, "2.5"), interfaces=(1,))
        plan = factory.plan(2)
        self.assertIs(factory.plan(2), plan)
        check = plan.check_refinement(lambda x: a.exp(x), atol="1e-20", rtol=0)
        self.assertFalse(check.converged)
        self.assertEqual(check.refined_plan.domain, plan.domain)
        self.assertEqual(check.refined_plan.mapping.raw_domain, (0, "2.5"))
        fine = plan.refined(order_factor=8, subdivisions_factor=2)
        self.assertEqual(fine.subdivisions, 2)
        self.assertLess(abs(fine.integrate(lambda x: 1)-a.scalar("2.5")), a.scalar("1e-47"))
        self.assertIsNot(plan, QuadratureFactory(MpmathArithmetic(50), domain=(0, "2.5")).plan(2))
        self.assertIsNot(plan, QuadratureFactory(a, domain=(0, 3)).plan(2))

    def test_raw_endpoint_rebuilds_after_precision_change_without_float64_leakage(self):
        original = mp.mp.prec
        a = MpmathArithmetic(30)
        raw = "1."+"0"*79+"1"
        domain = ["0", raw]
        factory = QuadratureFactory(a, domain=domain)
        old = factory.plan(2)
        domain[1] = "100"
        self.assertEqual(old.mapping.right, 1)
        self.assertLess(abs(old.integrate(lambda x: 1)-1), a.scalar("1e-28"))
        a.context.dps = 120
        new = factory.plan(2)
        self.assertEqual(new.mapping.raw_domain, ("0", raw))
        self.assertGreater(new.integrate(lambda x: 1), 1)
        self.assertLess(abs((new.integrate(lambda x: 1)-1)-a.scalar("1e-80")), a.scalar("1e-118"))
        with self.assertRaises(RuntimeError):
            old.integrate(lambda x: 1)
        self.assertEqual(mp.mp.prec, original)

    def test_outside_unordered_and_unresolved_interfaces_are_rejected(self):
        for interfaces in ((0,), (5,), (6,), (-1,), (2, 1), (1, 1), ("nan",), ("inf",)):
            with self.subTest(interfaces=interfaces), self.assertRaises(ValueError):
                QuadratureFactory(NumpyArithmetic(), domain=(0, 5), interfaces=interfaces)
        with self.assertRaises(ValueError):
            QuadratureFactory(NumpyArithmetic(), domain=(0, 5), interfaces=("1e-100",))
        for domain in ((0,), (1, 1), (2, 1), ("nan", 3)):
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                QuadratureFactory(NumpyArithmetic(), domain=domain)
        default = QuadratureFactory(NumpyArithmetic()).plan(2)
        explicit = QuadratureFactory(NumpyArithmetic(), domain=(0, "inf")).plan(2)
        self.assertEqual(default.mapping.mapping_type, "rational_half_line")
        self.assertEqual(default.physical_nodes, explicit.physical_nodes)
        self.assertEqual(default.physical_weights, explicit.physical_weights)
