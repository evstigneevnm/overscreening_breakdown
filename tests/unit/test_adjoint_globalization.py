"""Native merit recovery, bounded attempts, and final physical convergence."""

from dataclasses import replace
import unittest

import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.nonlinear_globalization import adjoint_step
from overscreening_breakdown.nonlinear_solver import (
    LinearSolveError, NonlinearSystem, solve_newton,
)


class AdjointGlobalizationTests(unittest.TestCase):
    def arithmetics(self):
        return [NumpyArithmetic()]+[MpmathArithmetic(dps) for dps in (15, 30, 50, 100, 120)]

    def tolerance(self, arithmetic):
        return arithmetic.scalar("1e-{}".format(max(9, arithmetic.dps-8)
                                               if arithmetic.use_mpmath else 12))

    def linear_solve(self, arithmetic):
        def solve(matrix, rhs):
            try:
                return arithmetic.solve(matrix, rhs)
            except (np.linalg.LinAlgError, ZeroDivisionError) as error:
                raise LinearSolveError(str(error)) from error
        return solve

    def scalar_system(self, arithmetic, *, initial=0, residual=None, jacobian=None,
                      solve_linear=None, **callbacks):
        return NonlinearSystem(
            residual=(lambda z, q: arithmetic.vector([z[0]-1])) if residual is None else residual,
            jacobian=(lambda z, q: arithmetic.matrix([[1]])) if jacobian is None else jacobian,
            solve_linear=self.linear_solve(arithmetic) if solve_linear is None else solve_linear,
            residual_norm=arithmetic.norm, initial_guess=lambda: arithmetic.vector([initial]),
            scalar=arithmetic.scalar, isfinite=arithmetic.isfinite, **callbacks,
        )

    def fail_linear(self, matrix, rhs):
        raise LinearSolveError("forced unusable Newton system")

    def test_singular_newton_recovers_and_retries_at_every_precision(self):
        for a in self.arithmetics():
            with self.subTest(backend=a.backend, precision=a.dps):
                system = NonlinearSystem(
                    residual=lambda z, q: a.vector([z[0]**2+z[1]-1, z[0]+z[1]**2-1]),
                    jacobian=lambda z, q: a.matrix([[2*z[0], 1], [1, 2*z[1]]]),
                    solve_linear=self.linear_solve(a), residual_norm=a.norm,
                    initial_guess=lambda: a.vector(["0.5", "0.5"]),
                    scalar=a.scalar, isfinite=a.isfinite,
                )
                original = solve_newton(system, atol=self.tolerance(a))
                self.assertFalse(original.converged)
                self.assertEqual(original.reason, "linear_solve_failed")
                callbacks = []
                def observer(state, record):
                    callbacks.append((state.copy(), record))
                    state[0] = a.scalar(999)  # observer cannot mutate accepted state
                result = solve_newton(system, atol=self.tolerance(a),
                                      newton_fallback="adjoint", callback=observer)
                self.assertTrue(result.converged, result.reason)
                self.assertEqual(result.homotopy, 0)
                exact = (a.sqrt(a.scalar(5))-1)/2
                self.assertLess(abs(result.state[0]-exact), 10*self.tolerance(a))
                self.assertLessEqual(result.residual_norm, self.tolerance(a))
                self.assertEqual(result.adjoint_steps, 1)
                self.assertEqual(result.adjoint_evaluations, 3)
                self.assertEqual(result.history[0].kind, "adjoint")
                self.assertEqual(result.history[0].backtracks, 2)
                self.assertEqual(result.history[0].step, a.scalar("0.25"))
                self.assertEqual(result.recovery_reasons, ("linear_solve_failed",))
                self.assertTrue(all(record.kind == "newton" for record in result.history[1:]))
                self.assertEqual(len(callbacks), len(result.history))
                self.assertEqual(result.iterations, sum(stage.iterations for stage in result.stages))
                self.assertTrue(all(isinstance(value, type(a.scalar(0))) for value in result.state))
                recovery = result.history[0]
                for value in (recovery.merit, recovery.gradient_norm, recovery.step):
                    self.assertIsInstance(value, type(a.scalar(0)))

    def test_weighted_merit_gradient_and_armijo_are_checked_independently(self):
        for a in self.arithmetics():
            with self.subTest(backend=a.backend, precision=a.dps):
                def residual(z, q):
                    return a.vector([z[0]**2+2*z[1]-q, 2*z[0]+z[1]**2+q])
                def jacobian(z, q):
                    return a.matrix([[2*z[0], 2], [2, 2*z[1]]])
                def dot(left, right):
                    return sum((x*y for x, y in zip(left, right)), a.scalar(0))
                def gradient(r, j):
                    return a.vector([sum((j[k, i]*r[k] for k in range(2)), a.scalar(0))/2
                                     for i in range(2)])
                system = NonlinearSystem(
                    residual=residual, jacobian=jacobian, solve_linear=self.fail_linear,
                    residual_norm=lambda r: a.norm(r)/a.sqrt(a.scalar(2)),
                    initial_guess=lambda: a.vector(["0.7", "-0.2"]), scalar=a.scalar,
                    isfinite=a.isfinite, merit_gradient=gradient, dot=dot,
                )
                state, q = system.initial_guess(), a.scalar("0.4")
                r, j = residual(state, q), jacobian(state, q)
                g = gradient(r, j)
                probe = a.vector([a.scalar(1)/5, -a.scalar(1)/3])
                def merit(t):
                    return system.residual_norm(residual(state+t*probe, q))**2/2
                if a.use_mpmath:
                    observed = a.context.diff(merit, a.scalar(0))
                    bound = self.tolerance(a)
                else:
                    h = a.scalar("1e-6")
                    observed = (merit(h)-merit(-h))/(2*h)
                    bound = a.scalar("1e-8")
                self.assertLess(abs(observed-dot(g, probe)), bound)
                parameters = []
                def evaluate(candidate, parameter):
                    parameters.append(parameter)
                    values = residual(candidate, parameter)
                    return values, system.residual_norm(values)
                step = adjoint_step(system, state, r, j, q, residual_norm=system.residual_norm(r),
                                    evaluate=evaluate)
                self.assertTrue(step.accepted, step.reason)
                self.assertTrue(all(parameter == q for parameter in parameters))
                self.assertLess(step.residual_norm, system.residual_norm(r))
                self.assertLessEqual(step.merit, step.initial_merit+a.scalar("1e-4")
                                     *step.step*step.directional_derivative)
                self.assertLess(step.directional_derivative, 0)

    def test_nonroot_stationarity_never_reports_convergence(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            system = self.scalar_system(
                a, residual=lambda z, q: a.vector([z[0]**2+1]),
                jacobian=lambda z, q: a.matrix([[2*z[0]]]),
            )
            result = solve_newton(system, atol=self.tolerance(a), newton_fallback="adjoint")
            self.assertFalse(result.converged)
            self.assertEqual(result.reason, "adjoint_stationary")
            self.assertEqual(result.residual_norm, 1)
            self.assertEqual(result.adjoint_steps, 0)
            self.assertEqual(result.adjoint_evaluations, 0)
            self.assertEqual(result.history, ())

    def test_stage_budget_bounds_repeated_recoveries_and_evaluation_counts(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(30)):
            system = self.scalar_system(a, solve_linear=self.fail_linear,
                                        adjoint_direction=lambda g: -g/4)
            result = solve_newton(system, newton_fallback="adjoint", adjoint_max_steps=2,
                                  max_iterations=50)
            self.assertFalse(result.converged)
            self.assertEqual(result.reason, "adjoint_budget_exhausted")
            self.assertEqual(result.iterations, 3)
            self.assertEqual(result.adjoint_steps, 2)
            self.assertEqual(result.adjoint_evaluations, 2)
            self.assertEqual(result.state[0], a.scalar("0.4375"))
            self.assertEqual(len(result.history), 2)
            self.assertTrue(all(record.kind == "adjoint" for record in result.history))

    def test_nonfinite_steps_and_failed_newton_search_can_recover(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(30)):
            for trigger in ("nonfinite_step", "line_search_failed"):
                with self.subTest(backend=a.backend, trigger=trigger):
                    callback = ((lambda matrix, rhs: a.vector([a.scalar("nan")]))
                                if trigger == "nonfinite_step" else lambda matrix, rhs: -rhs)
                    system = self.scalar_system(a, solve_linear=callback)
                    result = solve_newton(system, newton_fallback="adjoint", max_backtracks=4)
                    self.assertTrue(result.converged)
                    self.assertEqual(result.adjoint_steps, 1)
                    self.assertEqual(result.history[0].recovery_reason, trigger)

    def test_inplace_linear_failure_does_not_corrupt_the_merit_jacobian(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(30)):
            def destroy(matrix, rhs):
                matrix[0, 0] = a.scalar("nan")
                return self.fail_linear(matrix, rhs)
            system = self.scalar_system(a, solve_linear=destroy)
            result = solve_newton(system, newton_fallback="adjoint")
            self.assertTrue(result.converged, result.reason)
            self.assertEqual(result.adjoint_steps, 1)
            self.assertEqual(result.state[0], 1)

    def test_nonfinite_trials_are_rejected_without_observer_calls(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            rejected = []
            attempts = [0]
            def residual(z, q):
                if z[0] <= 0:
                    rejected.append(z[0])
                    raise OverflowError("outside the logarithm domain")
                return a.vector([a.log(z[0])+2])
            def solve(matrix, rhs):
                attempts[0] += 1
                if attempts[0] == 1:
                    return self.fail_linear(matrix, rhs)
                return self.linear_solve(a)(matrix, rhs)
            system = self.scalar_system(a, initial=1, residual=residual,
                                        jacobian=lambda z, q: a.matrix([[1/z[0]]]), solve_linear=solve)
            accepted = []
            result = solve_newton(system, newton_fallback="adjoint", atol=self.tolerance(a),
                                  callback=lambda z, record: accepted.append(z[0]))
            self.assertTrue(result.converged, result.reason)
            self.assertTrue(rejected)
            self.assertTrue(all(value > 0 for value in accepted))
            self.assertEqual(result.history[0].kind, "adjoint")
            self.assertEqual(result.history[0].backtracks, 1)
            self.assertEqual(result.adjoint_evaluations, 2)

    def test_unusable_gradients_and_directions_are_numerical_failures(self):
        a = MpmathArithmetic(50)
        variants = (
            ({"merit_gradient": lambda r, j: a.vector([a.inf])}, "adjoint_nonfinite_gradient"),
            ({"adjoint_direction": lambda g: a.vector([a.inf])}, "adjoint_nonfinite_direction"),
            ({"adjoint_direction": lambda g: g.copy()}, "adjoint_not_descent"),
            ({"adjoint_direction": lambda g: a.vector([0])}, "adjoint_stationary"),
        )
        for callbacks, reason in variants:
            with self.subTest(reason=reason):
                system = self.scalar_system(a, solve_linear=self.fail_linear, **callbacks)
                result = solve_newton(system, newton_fallback="adjoint")
                self.assertFalse(result.converged)
                self.assertEqual(result.reason, reason)
                self.assertEqual(result.history, ())
                self.assertEqual(result.adjoint_evaluations, 0)

    def test_nonfinite_residual_or_jacobian_does_not_invoke_fallback(self):
        a = NumpyArithmetic()
        def forbidden(*args):
            raise AssertionError("fallback should not run")
        for part, reason in (("residual", "nonfinite_residual"), ("jacobian", "nonfinite_jacobian")):
            system = self.scalar_system(a, merit_gradient=forbidden)
            system = replace(system, **{part: lambda z, q: a.vector([a.inf])
                                       if part == "residual" else a.matrix([[a.inf]])})
            result = solve_newton(system, newton_fallback="adjoint")
            self.assertFalse(result.converged)
            self.assertEqual(result.reason, reason)
            self.assertEqual(result.adjoint_steps, 0)

    def test_failed_adjoint_line_search_is_bounded_and_unaccepted(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(30)):
            system = self.scalar_system(a, residual=lambda z, q: a.vector([100*z[0]-1]),
                                        jacobian=lambda z, q: a.matrix([[100]]),
                                        solve_linear=self.fail_linear)
            result = solve_newton(system, newton_fallback="adjoint", max_backtracks=1)
            self.assertFalse(result.converged)
            self.assertEqual(result.reason, "adjoint_line_search_failed")
            self.assertEqual(result.adjoint_steps, 0)
            self.assertEqual(result.adjoint_evaluations, 1)
            self.assertEqual(result.history, ())

    def test_exact_initial_states_and_zero_newton_budgets_are_unchanged(self):
        a = MpmathArithmetic(50)
        def forbidden(*args):
            raise AssertionError("no derivative or fallback evaluation expected")
        for initial, converged in ((1, True), (0, False)):
            system = self.scalar_system(a, initial=initial, jacobian=forbidden,
                                        merit_gradient=forbidden)
            result = solve_newton(system, max_iterations=0, newton_fallback="adjoint")
            self.assertEqual(result.converged, converged)
            self.assertEqual(result.iterations, 0)
            self.assertEqual(result.adjoint_steps, 0)
            self.assertEqual(result.adjoint_evaluations, 0)
        system = self.scalar_system(a, solve_linear=self.fail_linear, adjoint_direction=lambda g: -g/4)
        result = solve_newton(system, max_iterations=1, newton_fallback="adjoint")
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "max_iterations")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.adjoint_steps, 1)

    def test_default_off_does_not_evaluate_new_hooks_or_change_history(self):
        a = NumpyArithmetic()
        def forbidden(*args):
            raise AssertionError("disabled fallback hook was called")
        system = self.scalar_system(a, merit_gradient=forbidden, dot=forbidden, adjoint_direction=forbidden)
        result = solve_newton(system)
        self.assertTrue(result.converged)
        self.assertEqual(result.adjoint_steps, 0)
        self.assertEqual(result.adjoint_evaluations, 0)
        self.assertEqual(result.recovery_reasons, ())
        self.assertEqual(result.history[0].kind, "newton")
        self.assertIsNone(result.history[0].merit)
        self.assertIsNone(result.history[0].gradient_norm)

    def test_tiny_120_digit_step_is_retained_without_float64_conversion(self):
        a = MpmathArithmetic(120)
        target = 1+a.scalar("1e-80")
        system = self.scalar_system(a, initial=1, residual=lambda z, q: a.vector([z[0]-target]),
                                    solve_linear=self.fail_linear)
        result = solve_newton(system, atol="1e-110", newton_fallback="adjoint")
        self.assertTrue(result.converged)
        self.assertEqual(result.state[0], target)
        self.assertEqual(result.adjoint_steps, 1)
        self.assertLess(abs(result.history[0].gradient_norm-a.scalar("1e-80")), a.scalar("1e-118"))
        self.assertIsInstance(result.history[0].merit, type(a.scalar(0)))
        # A finite but huge gradient needs positive direction normalization;
        # its squared Euclidean norm would overflow without stable scaling.
        a = NumpyArithmetic()
        system = self.scalar_system(a, residual=lambda z, q: a.vector([a.scalar("1e150")*(z[0]-1)]),
                                    jacobian=lambda z, q: a.matrix([["1e150"]]),
                                    solve_linear=self.fail_linear)
        with np.errstate(over="raise", invalid="raise"):
            result = solve_newton(system, newton_fallback="adjoint")
        self.assertTrue(result.converged)
        self.assertEqual(result.state[0], 1)
        self.assertTrue(a.isfinite(result.history[0].gradient_norm))

    def test_recovery_keeps_the_parameter_fixed_and_preserves_stage_labels(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(30)):
            parameter = a.scalar("0.75")
            parameters, attempts = [], []

            def residual(z, q):
                parameters.append(q)
                return a.vector([z[0]**2+q*z[0]-1])

            def solve(matrix, rhs):
                attempts.append(1)
                if len(attempts) == 1:
                    return self.fail_linear(matrix, rhs)
                return self.linear_solve(a)(matrix, rhs)

            system = self.scalar_system(
                a, residual=residual,
                jacobian=lambda z, q: a.matrix([[2*z[0]+q]]), solve_linear=solve,
            )
            observed = []
            result = solve_newton(
                system, parameter=parameter, stage=3, iteration_offset=10,
                atol=self.tolerance(a), newton_fallback="adjoint",
                callback=lambda z, record: observed.append((z.copy(), record)),
            )
            self.assertTrue(result.converged, result.reason)
            self.assertEqual(result.homotopy, parameter)
            self.assertEqual(result.adjoint_steps, 1)
            self.assertEqual(len(result.stages), 1)
            self.assertEqual(result.history[0].kind, "adjoint")
            self.assertEqual(result.history[0].iteration, 11)
            for state, record in observed:
                actual = a.norm(residual(state, parameter))
                self.assertLess(abs(actual-record.residual_norm), 10*self.tolerance(a))
                self.assertEqual(record.homotopy, parameter)
                self.assertEqual(record.stage, 3)
            self.assertTrue(all(q == parameter for q in parameters))
            self.assertEqual(result.adjoint_steps, result.stages[0].adjoint_steps)
            self.assertEqual(result.adjoint_evaluations,
                             result.stages[0].adjoint_evaluations)

    def test_programming_errors_in_new_callbacks_propagate(self):
        a = NumpyArithmetic()
        def broken(*args):
            raise RuntimeError("callback defect")
        for name in ("merit_gradient", "dot", "adjoint_direction"):
            system = self.scalar_system(a, solve_linear=self.fail_linear, **{name: broken})
            with self.subTest(callback=name), self.assertRaisesRegex(RuntimeError, "callback defect"):
                solve_newton(system, newton_fallback="adjoint")
        for name in ("merit_gradient", "adjoint_direction"):
            system = self.scalar_system(a, solve_linear=self.fail_linear,
                                        **{name: lambda *args: a.vector([1, 2])})
            with self.subTest(callback=name), self.assertRaises(ValueError):
                solve_newton(system, newton_fallback="adjoint")

    def test_numerical_direction_errors_and_overflow_fail_without_false_success(self):
        a = MpmathArithmetic(30)
        def overflow(*args):
            raise OverflowError("numerical overflow")
        for callback, reason in ((overflow, "adjoint_overflow"),
                                 (lambda g: self.fail_linear(None, None), "adjoint_direction_failed")):
            system = self.scalar_system(a, solve_linear=self.fail_linear, adjoint_direction=callback)
            result = solve_newton(system, newton_fallback="adjoint")
            self.assertFalse(result.converged)
            self.assertEqual(result.reason, reason)
            self.assertEqual(result.adjoint_steps, 0)

    def test_fallback_options_are_validated_and_zero_budget_is_explicit(self):
        a = NumpyArithmetic()
        system = self.scalar_system(a, solve_linear=self.fail_linear)
        for options in ({"newton_fallback": "invalid"}, {"newton_fallback": True},
                        {"adjoint_max_steps": -1}, {"adjoint_max_steps": True},
                        {"adjoint_max_steps": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                solve_newton(system, **options)
        result = solve_newton(system, newton_fallback="adjoint", adjoint_max_steps=0)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "adjoint_budget_exhausted")
        self.assertEqual(result.adjoint_evaluations, 0)


if __name__ == "__main__":
    unittest.main()
