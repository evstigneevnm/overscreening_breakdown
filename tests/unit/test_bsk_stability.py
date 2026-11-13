"""BSK values and Jacobians against the original model at guard precision."""

import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown import BSKProblem
from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.problems.bsk import (
    _physical_rhs,
    _physical_rhs_u,
    overscreening_breakdown,
)


def original_model(arithmetic, *, x, u, sigma, mu, gamma):
    """Differentiate the original quotient, independently of scaled kernels.

    Exact conversions of the owner's rounded parameters make float64 and
    mpmath comparisons refer to the same physical data. Guard digits resolve
    the cancellation of exp(2*abs(u)) terms in the quotient-rule derivative.
    """
    ctx = mp.mp.clone()
    ctx.dps = max(1250, arithmetic.dps + 100)
    x, u, sigma, mu, gamma = (
        ctx.mpf(arithmetic.scalar(value)) for value in (x, u, sigma, mu, gamma)
    )
    gaussian = ctx.exp(-x*x/(2*sigma*sigma))/ctx.sqrt(2*ctx.pi)
    numerator = ctx.sinh(u)-mu*gaussian*ctx.exp(u)/2
    numerator_u = ctx.cosh(u)-mu*gaussian*ctx.exp(u)/2
    denominator = 1+2*gamma*ctx.sinh(u/2)**2
    denominator_u = gamma*ctx.sinh(u)
    return (ctx, numerator/denominator,
            (numerator_u*denominator-numerator*denominator_u)/denominator**2)


class BSKStabilityTests(unittest.TestCase):
    def assert_reference(self, arithmetic, *, x=0, u, sigma=3, mu="0.1", gamma="0.5",
                         relative=None):
        source = BSKProblem(sigma=sigma, mu=mu, gamma=gamma).bind(arithmetic)
        ctx, expected, derivative = original_model(
            arithmetic, x=x, u=u, sigma=sigma, mu=mu, gamma=gamma)
        tolerance = (ctx.mpf(relative) if relative else
                     ctx.mpf("2e-12") if not arithmetic.use_mpmath else
                     ctx.power(10, -arithmetic.dps+3))
        for method, target in (("physical_rhs", expected), ("physical_rhs_u", derivative)):
            with self.subTest(method=method):
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    value = getattr(source, method)(x, u)
                self.assertIsInstance(value, type(arithmetic.scalar(0)))
                self.assertTrue(arithmetic.isfinite(value))
                actual = ctx.mpf(value)
                if not arithmetic.use_mpmath and float(target) == 0:
                    self.assertEqual(value, 0)
                elif target == 0:
                    self.assertEqual(value, 0)
                else:
                    # Relative error detects lost tiny Jacobians; an absolute
                    # tolerance at ordinary scale would accept a false zero.
                    self.assertLessEqual(abs((actual-target)/target), tolerance)

    def test_signed_states_and_parameters_against_original_model(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(15), MpmathArithmetic(30),
                  MpmathArithmetic(50), MpmathArithmetic(100), MpmathArithmetic(120)):
            for gamma in ("0", "0.5", "1", "2"):
                for u in ("-2", "-1e-20", "0", "1e-20", "2"):
                    with self.subTest(backend=a.backend, dps=a.dps, gamma=gamma, u=u):
                        self.assert_reference(a, x="0.125", u=u, gamma=gamma)

    def test_small_state_without_gaussian_preserves_relative_accuracy(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(30), MpmathArithmetic(120)):
            for u in ("-1e-100", "-1e-20", "1e-20", "1e-100"):
                with self.subTest(backend=a.backend, dps=a.dps, u=u):
                    self.assert_reference(a, u=u, mu=0)

    def test_large_signed_states_retain_representable_float64_derivatives(self):
        a = NumpyArithmetic()
        for gamma, states in (("0.5", (-400, 400)), ("1", (-300, 300))):
            for u in states:
                with self.subTest(gamma=gamma, u=u):
                    self.assert_reference(a, u=u, gamma=gamma)
                    derivative = BSKProblem(mu="0.1", gamma=gamma).bind(a).physical_rhs_u(0, u)
                    self.assertGreater(derivative, 0)

    def test_saturation_and_true_float64_underflow_at_extreme_states(self):
        a = NumpyArithmetic()
        source = BSKProblem(mu="0.1", gamma="0.5").bind(a)
        for u in (-1000, 1000):
            with self.subTest(u=u):
                self.assert_reference(a, u=u)
                self.assertEqual(source.physical_rhs_u(0, u), 0)
        self.assertEqual(source.physical_rhs(0, -1000), -2)
        expected_positive_limit = 2*(1-0.1/np.sqrt(2*np.pi))
        self.assertAlmostEqual(source.physical_rhs(0, 1000), expected_positive_limit, places=14)

    def test_large_states_remain_native_at_every_mpmath_precision(self):
        for dps in (15, 30, 50, 100, 120):
            a = MpmathArithmetic(dps)
            for gamma, u in (("0.5", -1000), ("0.5", 1000), ("1", -1000),
                             ("1", 1000), ("0", -700), ("0", 700)):
                with self.subTest(dps=dps, gamma=gamma, u=u):
                    self.assert_reference(a, u=u, gamma=gamma)

    def test_tiny_positive_gamma_does_not_erase_representable_results(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50), MpmathArithmetic(120)):
            for u in (-1000, -750, 750, 1000):
                with self.subTest(backend=a.backend, dps=a.dps, u=u):
                    self.assert_reference(a, u=u, gamma="1e-200", relative="3e-12"
                                          if not a.use_mpmath else None)

    def test_subnormal_positive_gamma_can_have_a_finite_derivative(self):
        a = NumpyArithmetic()
        source = BSKProblem(mu="0.1", gamma="1e-320").bind(a)
        # The source exceeds float64 range while its derivative is ordinary.
        # Evaluate the callbacks independently instead of rejecting both.
        for u in (-1000, 1000):
            with self.subTest(u=u):
                ctx, _, derivative = original_model(
                    a, x=0, u=u, sigma=10, mu="0.1", gamma="1e-320")
                actual = source.physical_rhs_u(0, u)
                self.assertTrue(np.isfinite(actual))
                self.assertLess(abs(ctx.mpf(actual)/derivative-1), ctx.mpf("3e-12"))
                with self.assertRaises(OverflowError):
                    source.physical_rhs(0, u)

    def test_float64_subnormal_derivatives_are_retained(self):
        a = NumpyArithmetic()
        for gamma, u in (("0.5", -744), ("0.5", 744), ("1", -372), ("1", 372)):
            with self.subTest(gamma=gamma, u=u):
                source = BSKProblem(mu="0.1", gamma=gamma).bind(a)
                _, _, derivative = original_model(
                    a, x=0, u=u, sigma=10, mu="0.1", gamma=gamma)
                expected = float(derivative)
                actual = source.physical_rhs_u(0, u)
                self.assertGreater(expected, 0)
                self.assertLess(expected, np.finfo(np.float64).tiny)
                self.assertGreater(actual, 0)
                self.assertLessEqual(abs(actual-expected), np.nextafter(0, 1))

    def test_large_gamma_and_signed_mu_against_original_model(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            for gamma, u in (("1e300", "1e-150"), ("1e300", "-1e-150"),
                             ("2", "400"), ("2", "-400")):
                for mu in ("-0.7", "0", "0.7"):
                    with self.subTest(backend=a.backend, gamma=gamma, u=u, mu=mu):
                        self.assert_reference(a, u=u, gamma=gamma, mu=mu)

    def test_small_derivative_ratio_can_be_amplified_into_float64_range(self):
        # The two factors of the factored derivative must be combined before
        # underflowing t/d; the final positive-state derivatives are finite.
        for u in (-700, -600, 600, 700):
            with self.subTest(u=u):
                self.assert_reference(NumpyArithmetic(), u=u, gamma="1e100", mu="1e208")

    def test_exact_owner_gaussian_cancellation_retains_decaying_field(self):
        # This tests a floating-point identity of the owner's rounded b=1.
        # A higher-precision pi would instead change b, spoiling the exact
        # cancellation and describing different data at these extreme states.
        for a in (NumpyArithmetic(), MpmathArithmetic(15), MpmathArithmetic(30),
                  MpmathArithmetic(50), MpmathArithmetic(100), MpmathArithmetic(120)):
            mu = a.sqrt(2*a.pi)
            self.assertEqual(mu/a.sqrt(2*a.pi), 1)
            tolerance = a.scalar("2e-12") if not a.use_mpmath else a.scalar(10)**(-a.dps+3)
            for u in (100, 400, 1000):
                native = a.scalar(u)
                expected = a.exp(-native)/2
                for callback, sign in ((_physical_rhs, -1), (_physical_rhs_u, 1)):
                    with self.subTest(backend=a.backend, dps=a.dps, u=u,
                                      callback=callback.__name__):
                        actual = callback(a.scalar(0), native, a, a.scalar(3), mu, a.scalar(0))
                        if expected == 0:
                            self.assertEqual(actual, 0)
                        else:
                            self.assertLessEqual(abs(actual/(sign*expected)-1), tolerance)

    def test_exact_owner_zero_state_jacobian_cancellation(self):
        # With the owner's b=2, f=-cosh(u)/(1-gamma+gamma*cosh(u)).
        # This catches loss of the small nonzero derivative near the root,
        # including its sign reversal for gamma>1 and exact zero at gamma=1.
        for a in (NumpyArithmetic(), MpmathArithmetic(15), MpmathArithmetic(30),
                  MpmathArithmetic(50), MpmathArithmetic(100), MpmathArithmetic(120)):
            mu = 2*a.sqrt(2*a.pi)
            self.assertEqual(mu/a.sqrt(2*a.pi), 2)
            tolerance = a.scalar("2e-12") if not a.use_mpmath else a.scalar(10)**(-a.dps+3)
            for gamma_text in ("0", "0.5", "1", "2"):
                gamma = a.scalar(gamma_text)
                for text in ("-1e-100", "-1e-20", "1e-20", "1e-100"):
                    u = a.scalar(text)
                    with self.subTest(backend=a.backend, dps=a.dps, gamma=gamma_text, u=text):
                        value = _physical_rhs(a.scalar(0), u, a, a.scalar(3), mu, gamma)
                        derivative = _physical_rhs_u(a.scalar(0), u, a, a.scalar(3), mu, gamma)
                        denominator = 1-gamma+gamma*a.cosh(u)
                        expected_value = -a.cosh(u)/denominator
                        expected_derivative = -(1-gamma)*a.sinh(u)/denominator**2
                        self.assertLessEqual(abs(value/expected_value-1), tolerance)
                        if expected_derivative == 0:
                            self.assertEqual(derivative, 0)
                        else:
                            self.assertLessEqual(abs(derivative/expected_derivative-1), tolerance)

    def test_gamma_zero_preserves_growth_and_reports_true_overflow(self):
        a = NumpyArithmetic()
        for u in (-700, 700):
            with self.subTest(u=u):
                self.assert_reference(a, u=u, gamma=0)
        source = BSKProblem(mu="0.1", gamma=0).bind(a)
        for u in (-1000, 1000):
            for method in (source.physical_rhs, source.physical_rhs_u):
                with self.subTest(u=u, method=method.__name__):
                    with self.assertRaises(OverflowError):
                        method(0, u)

    def test_extreme_coordinates_and_widths_have_finite_gaussian_tail(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            for x, sigma in (("1e308", "1e-308"), ("1e200", "1"),
                             ("1e-308", "1e308")):
                with self.subTest(backend=a.backend, x=x, sigma=sigma):
                    self.assert_reference(a, x=x, sigma=sigma, u="0.37")

    def test_weighted_gaussian_survives_unweighted_underflow(self):
        # Subnormal or underflowed Gaussians become ordinary after weighting;
        # reconstruct before their earlier rounding destroys accuracy.
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            for x in (a.scalar(np.sqrt(2*740)), a.scalar(np.sqrt(2*744)),
                      a.scalar(np.sqrt(2*745)), a.scalar(40)):
                with self.subTest(backend=a.backend, x=str(x)):
                    self.assert_reference(a, x=x, sigma=1, mu="1e300", u=0)

    def test_zero_mu_has_odd_source_and_even_derivative(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50), MpmathArithmetic(120)):
            for gamma in (0, "0.5", 1, 2):
                source = BSKProblem(mu=0, gamma=gamma).bind(a)
                for u in (a.scalar("1e-20"), a.scalar("0.37"), a.scalar(300)):
                    with self.subTest(backend=a.backend, gamma=gamma, u=str(u)):
                        self.assertEqual(source.physical_rhs(1, -u), -source.physical_rhs(1, u))
                        self.assertEqual(source.physical_rhs_u(1, -u), source.physical_rhs_u(1, u))

    def test_legacy_arrays_keep_broadcasting_and_numpy_dtype(self):
        legacy = overscreening_breakdown()
        legacy.set_parameters(sigma=3, mu="0.1", gamma="0.5")
        x = np.array([[0], [1]])
        u = np.array([-1000, -400, 0, 400, 1000])
        modern = BSKProblem(sigma=3, mu="0.1", gamma="0.5").bind(NumpyArithmetic())
        for name in ("physical_rhs", "physical_rhs_u"):
            with self.subTest(method=name):
                values = np.asarray(getattr(legacy, name)(x, u))
                self.assertEqual(values.shape, (2, 5))
                self.assertEqual(values.dtype, np.dtype("float64"))
                expected = [[getattr(modern, name)(int(xi[0]), int(ui)) for ui in u] for xi in x]
                np.testing.assert_allclose(values, expected, rtol=1e-13, atol=0)

    def test_legacy_mpmath_arrays_keep_native_values(self):
        a = MpmathArithmetic(50)
        legacy = overscreening_breakdown()
        legacy.set_use_mp_math(a.context)
        legacy.set_parameters(sigma=3, mu="0.1", gamma="0.5")
        x = np.array([[a.scalar(0)], [a.scalar(1)]], dtype=object)
        u = np.array([a.scalar(-1000), a.scalar(0), a.scalar(1000)], dtype=object)
        modern = BSKProblem(sigma=3, mu="0.1", gamma="0.5").bind(a)
        for name in ("physical_rhs", "physical_rhs_u"):
            values = np.asarray(getattr(legacy, name)(x, u))
            self.assertEqual(values.shape, (2, 3))
            self.assertEqual(values.dtype, np.dtype("object"))
            for i in range(2):
                for j in range(3):
                    with self.subTest(method=name, i=i, j=j):
                        self.assertIsInstance(values[i, j], type(a.scalar(0)))
                        self.assertEqual(values[i, j], getattr(modern, name)(x[i, 0], u[j]))


if __name__ == "__main__":
    unittest.main()
