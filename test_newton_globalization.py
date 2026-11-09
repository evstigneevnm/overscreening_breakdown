"""Behavioral checks for the shared damped Newton and continuation driver.

Run with ``python -m unittest test_newton_globalization``.  Small systems make
the iteration policy observable independently of either spatial formulation.
"""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import mpmath as mp
import numpy as np

from nonlinear_solver import LinearSolveError, NonlinearSystem, solve_newton


class NewtonGlobalizationTests(unittest.TestCase):
    def system(self, initial, residual, jacobian, use_mpmath=False, dps=30,
               restart=None):
        if use_mpmath:
            ctx = mp.mp.clone()
            ctx.dps = dps
            scalar = ctx.mpf
            vector = lambda values: ctx.matrix([scalar(value) for value in values])
            matrix = ctx.matrix
            native_solve = ctx.lu_solve
            norm = ctx.norm
            isfinite = ctx.isfinite
            linear_errors = (ZeroDivisionError,)
        else:
            ctx = SimpleNamespace(mpf=float, exp=np.exp, log=np.log, sqrt=np.sqrt)
            scalar = float
            vector = lambda values: np.array(values, dtype=float)
            matrix = lambda values: np.array(values, dtype=float)
            native_solve = np.linalg.solve
            norm = np.linalg.norm
            isfinite = np.isfinite
            linear_errors = (np.linalg.LinAlgError,)

        def solve_linear(a, b):
            try:
                return native_solve(a, b)
            except linear_errors as error:
                raise LinearSolveError(str(error)) from error

        initial_state = vector(initial)
        system = NonlinearSystem(
            residual=lambda z, q: vector(residual(z, q, ctx)),
            jacobian=lambda z, q: matrix(jacobian(z, q, ctx)),
            solve_linear=solve_linear,
            residual_norm=norm,
            initial_guess=lambda: initial_state.copy(),
            scalar=scalar,
            isfinite=isfinite,
            restart=restart,
        )
        return system, ctx

    def backends(self):
        return (False, True)

    def tolerance(self, use_mpmath):
        return "1e-22" if use_mpmath else "1e-12"

    def test_exact_initial_solution_needs_no_linearization(self):
        for use_mpmath in self.backends():
            with self.subTest(mpmath=use_mpmath):
                def unexpected_jacobian(z, q, ctx):
                    self.fail("An exact initial root must not need a Jacobian")

                system, _ = self.system(
                    ["2"], lambda z, q, ctx: [z[0] - 2],
                    unexpected_jacobian, use_mpmath,
                )
                result = solve_newton(system, max_iterations=0)
                self.assertTrue(result.converged)
                self.assertEqual(result.homotopy, 0)
                self.assertEqual(result.residual_norm, 0)
                self.assertEqual(result.iterations, 0)
                self.assertEqual(len(result.history), 0)
                self.assertEqual(result.stages[0].initial_norm, 0)

    def test_damping_accepts_only_residual_decreasing_states(self):
        for use_mpmath in self.backends():
            with self.subTest(mpmath=use_mpmath):
                system, ctx = self.system(
                    ["0.1"], lambda z, q, ctx: [z[0] ** 3 - 1],
                    lambda z, q, ctx: [[3 * z[0] ** 2]], use_mpmath,
                )
                snapshots = []
                result = solve_newton(
                    system, atol=self.tolerance(use_mpmath),
                    use_globalization=False,
                    callback=lambda z, step: snapshots.append((z, step)),
                )
                self.assertTrue(result.converged)
                self.assertTrue(any(step.step < 1 for step in result.history))
                old_norm = abs(ctx.mpf("0.1") ** 3 - 1)
                self.assertLess(abs(result.stages[0].initial_norm - old_norm),
                                ctx.mpf(self.tolerance(use_mpmath)))
                self.assertEqual(len(snapshots), len(result.history))
                for state, step in snapshots:
                    actual_norm = system.residual_norm(system.residual(state, 0))
                    self.assertLess(actual_norm, old_norm)
                    self.assertEqual(actual_norm, step.residual_norm)
                    old_norm = actual_norm
                self.assertLess(abs(result.state[0] - 1),
                                ctx.mpf(self.tolerance(use_mpmath)))

    def test_rejected_trials_never_replace_last_accepted_state(self):
        for use_mpmath in self.backends():
            with self.subTest(mpmath=use_mpmath):
                system, ctx = self.system(
                    ["0.1"], lambda z, q, ctx: [z[0] ** 3 - 1],
                    lambda z, q, ctx: [[3 * z[0] ** 2]], use_mpmath,
                )
                result = solve_newton(
                    system, atol=self.tolerance(use_mpmath), min_step="0.5",
                    use_globalization=False,
                )
                self.assertFalse(result.converged)
                self.assertEqual(result.reason, "line_search_failed")
                self.assertEqual(result.state[0], ctx.mpf("0.1"))
                self.assertEqual(len(result.history), 0)
                self.assertEqual(result.residual_norm,
                                 abs(ctx.mpf("0.1") ** 3 - 1))

    def test_nonfinite_trials_are_backtracked_until_a_finite_step_is_found(self):
        evaluations = []

        def residual(z, q, ctx):
            value = ctx.exp(z[0]) - 2
            evaluations.append(value)
            return [value]

        system, ctx = self.system(
            [-10], residual, lambda z, q, ctx: [[ctx.exp(z[0])]],
        )
        with np.errstate(over="ignore"):
            result = solve_newton(system, atol="1e-12", use_globalization=False)
        self.assertTrue(any(not np.isfinite(value) for value in evaluations))
        self.assertTrue(result.converged)
        self.assertLess(abs(result.state[0] - ctx.log(2)), 1e-12)
        self.assertTrue(all(np.isfinite(step.residual_norm)
                            for step in result.history))
        self.assertGreater(result.history[0].backtracks, 0)

    def test_convergence_on_last_allowed_iteration_is_success(self):
        for use_mpmath in self.backends():
            with self.subTest(mpmath=use_mpmath):
                system, _ = self.system(
                    [0], lambda z, q, ctx: [z[0] - 3],
                    lambda z, q, ctx: [[1]], use_mpmath,
                )
                result = solve_newton(system, max_iterations=1)
                self.assertTrue(result.converged)
                self.assertEqual(result.iterations, 1)
                self.assertEqual(result.state[0], 3)

    def test_nonfinite_initial_residual_does_not_attempt_a_newton_step(self):
        for use_mpmath in self.backends():
            with self.subTest(mpmath=use_mpmath):
                def unexpected_jacobian(z, q, ctx):
                    self.fail("A nonfinite initial residual cannot be linearized")

                system, _ = self.system(
                    [2], lambda z, q, ctx: [ctx.mpf("inf")],
                    unexpected_jacobian, use_mpmath,
                )
                result = solve_newton(system, use_globalization=False)
                self.assertFalse(result.converged)
                self.assertEqual(result.reason, "nonfinite_residual")
                self.assertEqual(result.state[0], 2)
                self.assertEqual(result.iterations, 0)

    def test_nonfinite_jacobian_can_trigger_continuation_without_linear_solving(self):
        for use_mpmath in self.backends():
            for use_globalization in (False, True):
                with self.subTest(mpmath=use_mpmath,
                                  globalization=use_globalization):
                    system, _ = self.system(
                        [0, 0], lambda z, q, ctx: [z[0] - 1, z[1] - 1],
                        lambda z, q, ctx: (
                            [[ctx.mpf("nan"), 0], [0, ctx.mpf("nan")]]
                            if q == 0 else [[1, 0], [0, 1]]
                        ), use_mpmath,
                    )
                    solves = []
                    native_solve = system.solve_linear

                    def tracked_solve(a, b):
                        solves.append(a)
                        return native_solve(a, b)

                    system = replace(system, solve_linear=tracked_solve)
                    result = solve_newton(
                        system, use_globalization=use_globalization,
                        homotopy_cutoff="0.1", max_stages=20,
                    )
                    self.assertEqual(result.stages[0].reason,
                                     "nonfinite_jacobian")
                    if use_globalization:
                        self.assertTrue(result.converged)
                        self.assertEqual(result.homotopy, 0)
                        self.assertEqual(result.state[0], 1)
                        self.assertEqual(result.state[1], 1)
                        self.assertEqual(len(solves), 1)
                    else:
                        self.assertFalse(result.converged)
                        self.assertEqual(result.reason, "nonfinite_jacobian")
                        self.assertEqual(result.state[0], 0)
                        self.assertEqual(result.state[1], 0)
                        self.assertEqual(len(solves), 0)

    def test_jacobian_overflow_follows_the_nonfinite_failure_policy(self):
        for use_mpmath in self.backends():
            for use_globalization in (False, True):
                with self.subTest(mpmath=use_mpmath, globalization=use_globalization):
                    def jacobian(z, q, ctx):
                        if q == 0:
                            raise OverflowError("physical derivative overflow")
                        return [[1]]

                    system, _ = self.system(
                        [0], lambda z, q, ctx: [z[0] - 1], jacobian, use_mpmath,
                    )
                    result = solve_newton(
                        system, use_globalization=use_globalization,
                        homotopy_cutoff="0.1", max_stages=20,
                    )
                    self.assertEqual(result.stages[0].reason, "nonfinite_jacobian")
                    self.assertEqual(result.converged, use_globalization)
                    self.assertEqual(result.state[0], 1 if use_globalization else 0)
                    if use_globalization:
                        self.assertEqual(result.homotopy, 0)

    def test_nonfinite_newton_step_is_rejected_without_changing_state(self):
        for use_mpmath in self.backends():
            for value in ("nan", "inf"):
                with self.subTest(mpmath=use_mpmath, value=value):
                    system, ctx = self.system(
                        [0], lambda z, q, ctx: [z[0] - 1],
                        lambda z, q, ctx: [[1]], use_mpmath,
                    )
                    step = system.initial_guess()
                    step[0] = ctx.mpf(value)
                    system = replace(system, solve_linear=lambda a, b: step.copy())
                    result = solve_newton(system, use_globalization=False)
                    self.assertFalse(result.converged)
                    self.assertEqual(result.reason, "nonfinite_step")
                    self.assertEqual(result.state[0], 0)
                    self.assertEqual(len(result.history), 0)

    def test_zero_iteration_budget_returns_nonroot_unchanged(self):
        for use_mpmath in self.backends():
            with self.subTest(mpmath=use_mpmath):
                system, _ = self.system(
                    [0], lambda z, q, ctx: [z[0] - 3],
                    lambda z, q, ctx: [[1]], use_mpmath,
                )
                result = solve_newton(
                    system, max_iterations=0, use_globalization=False,
                )
                self.assertFalse(result.converged)
                self.assertEqual(result.reason, "max_iterations")
                self.assertEqual(result.state[0], 0)
                self.assertEqual(result.iterations, 0)

    def test_relative_target_uses_measured_initial_residual(self):
        for use_mpmath in self.backends():
            with self.subTest(mpmath=use_mpmath):
                system, _ = self.system(
                    [0], lambda z, q, ctx: [1000 * (z[0] - 1)],
                    lambda z, q, ctx: [[1000]], use_mpmath,
                )
                result = solve_newton(system, atol="0", rtol="0.01")
                self.assertTrue(result.converged)
                self.assertEqual(result.stages[0].initial_norm, 1000)
                self.assertEqual(result.stages[0].target_norm, 10)

    def continuation_system(self, use_mpmath=False, restart=None, dimension=1):
        evaluations = []

        def residual(z, q, ctx):
            evaluations.append((q, z.copy()))
            a = ctx.exp(-q)
            value = a * z[dimension - 1] ** 3 + (1 - a) * z[dimension - 1] - a
            return [value] if dimension == 1 else [z[0] - 7, value]

        def jacobian(z, q, ctx):
            a = ctx.exp(-q)
            value = 3 * a * z[dimension - 1] ** 2 + 1 - a
            return [[value]] if dimension == 1 else [[1, 0], [0, value]]

        system, ctx = self.system(
            [0] if dimension == 1 else [7, 0], residual, jacobian,
            use_mpmath, restart=restart,
        )
        return system, ctx, evaluations

    def test_continuation_must_solve_the_final_physical_system(self):
        for use_mpmath in self.backends():
            with self.subTest(mpmath=use_mpmath):
                system, ctx, evaluations = self.continuation_system(use_mpmath)
                tolerance = ctx.mpf(self.tolerance(use_mpmath))
                result = solve_newton(
                    system, atol=self.tolerance(use_mpmath),
                    homotopy_cutoff="0.1", max_stages=30,
                )
                self.assertTrue(result.converged)
                self.assertEqual(result.homotopy, 0)
                self.assertEqual(result.stages[0].reason, "linear_solve_failed")
                self.assertTrue(any(stage.homotopy > 0 for stage in result.stages))
                self.assertTrue(result.stages[-1].converged)
                self.assertEqual(result.stages[-1].homotopy, 0)
                self.assertLess(abs(result.state[0] ** 3 - 1), tolerance)
                last_positive = max(index for index, (q, _) in enumerate(evaluations)
                                    if q > 0)
                self.assertTrue(any(q == 0 for q, _ in evaluations[last_positive + 1:]))
                # A stage's stopping target must use that stage's actual start.
                for stage in result.stages:
                    self.assertGreaterEqual(stage.initial_norm, 0)
                    if stage.converged:
                        self.assertLessEqual(stage.final_norm, stage.target_norm)

    def test_restart_callback_preserves_a_nonzero_essential_component(self):
        for use_mpmath in self.backends():
            with self.subTest(mpmath=use_mpmath):
                restarts = []

                def restart(state, q):
                    restarts.append((state.copy(), q))
                    state[0] = 7
                    state[1] = 0
                    return state

                system, _, evaluations = self.continuation_system(
                    use_mpmath, restart=restart, dimension=2,
                )
                result = solve_newton(
                    system, atol=self.tolerance(use_mpmath),
                    homotopy_cutoff="0.1", max_stages=30,
                )
                self.assertTrue(result.converged)
                self.assertTrue(restarts)
                self.assertTrue(all(state[0] == 7 for _, state in evaluations))
                self.assertEqual(result.state[0], 7)

    def test_stage_limit_cannot_claim_physical_convergence(self):
        system, _, _ = self.continuation_system()
        result = solve_newton(system, atol="1e-12", max_stages=2)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "max_stages")
        self.assertEqual(len(result.stages), 2)
        self.assertGreater(abs(result.state[0] ** 3 - 1), 1e-12)

    def test_callback_receives_a_copy_of_the_accepted_state(self):
        system, _ = self.system(
            [0], lambda z, q, ctx: [z[0] - 3],
            lambda z, q, ctx: [[1]],
        )
        observed = []

        def callback(state, step):
            observed.append(state[0])
            state[0] = -1000

        result = solve_newton(system, callback=callback)
        self.assertTrue(result.converged)
        self.assertEqual(observed, [3])
        self.assertEqual(result.state[0], 3)

    def test_unexpected_system_callback_errors_propagate(self):
        system, _ = self.system(
            [0], lambda z, q, ctx: [z[0] - 3],
            lambda z, q, ctx: [[1]],
        )

        def fail(a, b):
            raise RuntimeError("unexpected user error")

        for callback in ("residual", "jacobian", "solve_linear"):
            with self.subTest(callback=callback):
                failing_system = replace(system, **{callback: fail})
                with self.assertRaisesRegex(RuntimeError, "unexpected user error"):
                    solve_newton(failing_system)

    def test_invalid_options_are_rejected_before_iteration(self):
        invalid_options = (
            {"max_iterations": -1},
            {"max_iterations": True},
            {"max_stages": 0},
            {"max_stages": 1.5},
            {"max_backtracks": 0},
            {"atol": "-1"},
            {"rtol": "-1"},
            {"atol": "nan"},
            {"rtol": "inf"},
            {"atol": "0", "rtol": "0"},
            {"min_step": "0"},
            {"min_step": "1.01"},
            {"homotopy_cutoff": "0"},
            {"globalization_init": "0"},
        )
        for use_mpmath in self.backends():
            system, _ = self.system(
                [0], lambda z, q, ctx: [z[0] - 3],
                lambda z, q, ctx: [[1]], use_mpmath,
            )
            for options in invalid_options:
                with self.subTest(mpmath=use_mpmath, options=options):
                    with self.assertRaises(ValueError):
                        solve_newton(system, **options)

    def test_mpmath_preserves_a_correction_below_float64_resolution(self):
        system, ctx = self.system(
            [1],
            lambda z, q, ctx: [z[0] ** 2 - (1 + ctx.mpf("1e-80"))],
            lambda z, q, ctx: [[2 * z[0]]],
            use_mpmath=True, dps=120,
        )
        result = solve_newton(system, atol="1e-110", use_globalization=False)
        target = 1 + ctx.mpf("1e-80")
        self.assertTrue(result.converged)
        self.assertIsInstance(result.state[0], type(ctx.mpf(0)))
        self.assertNotEqual(result.state[0], 1)
        self.assertLess(abs(result.state[0] ** 2 - target), ctx.mpf("1e-110"))
        self.assertLess(abs(result.state[0] - ctx.sqrt(target)), ctx.mpf("1e-110"))


if __name__ == "__main__":
    unittest.main()
