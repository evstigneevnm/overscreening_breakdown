"""Progress observes native iteration and never changes numerical work."""

from contextlib import redirect_stdout
import io
import unittest

import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.diagnostics import ProgressCallbackError, ProgressReporter
from overscreening_breakdown.initialization import initialize_fsolve
from overscreening_breakdown.nonlinear_solver import LinearSolveError, NonlinearSystem, solve_newton


class NewtonProgressTests(unittest.TestCase):
    def system(self, arithmetic, *, exact=False, fail_first=False):
        counts = dict(residual=0, jacobian=0, linear=0)

        def residual(state, q):
            counts["residual"] += 1
            return arithmetic.vector([state[0]**3-1])

        def jacobian(state, q):
            counts["jacobian"] += 1
            return arithmetic.matrix([[3*state[0]**2]])

        def linear(matrix, rhs):
            counts["linear"] += 1
            if fail_first and counts["linear"] == 1:
                raise LinearSolveError("forced Newton failure")
            return arithmetic.solve(matrix, rhs)

        return NonlinearSystem(
            residual=residual, jacobian=jacobian, solve_linear=linear,
            residual_norm=arithmetic.norm,
            initial_guess=lambda: arithmetic.vector([1 if exact else "0.1"]),
            scalar=arithmetic.scalar, isfinite=arithmetic.isfinite,
        ), counts

    def test_all_levels_preserve_native_states_work_counts_and_accepted_observer(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            reference = None
            for verbosity in range(4):
                with self.subTest(backend=arithmetic.backend, verbosity=verbosity):
                    system, counts = self.system(arithmetic)
                    events, accepted = [], []
                    stream = io.StringIO()
                    reporter = ProgressReporter(
                        verbosity=verbosity, progress_callback=events.append,
                        arithmetic=arithmetic, solver="test", method="newton",
                        progress_interval=0, stream=stream).bind(norm="euclidean")

                    def observe(state, record):
                        accepted.append(record)
                        state[0] = arithmetic.scalar(999)

                    result = solve_newton(
                        system, atol="1e-30" if arithmetic.use_mpmath else "1e-12",
                        callback=observe, progress=reporter)
                    self.assertTrue(result.converged)
                    snapshot = (tuple(result.state), tuple(result.history), counts.copy())
                    if reference is None:
                        reference = snapshot
                    else:
                        self.assertEqual(snapshot, reference)
                    self.assertEqual(tuple(accepted), result.history)
                    completed = [event for event in events if event.kind == "iteration_end"]
                    self.assertEqual(len(completed), len(result.history))
                    self.assertTrue(any(event.kind == "trial_rejected" for event in events))
                    self.assertEqual(events[0].kind, "stage_start")
                    self.assertEqual(events[-1].kind, "newton_end")
                    self.assertEqual(stream.getvalue() == "", verbosity == 0)
                    if arithmetic.use_mpmath:
                        for event in completed:
                            self.assertIsInstance(event.data["residual_norm"],
                                                  type(arithmetic.scalar(0)))

    def test_start_is_observable_before_first_residual_and_exact_roots_need_no_steps(self):
        arithmetic = NumpyArithmetic()
        events = []
        system, counts = self.system(arithmetic, exact=True)
        original = system.residual

        def residual(state, q):
            self.assertEqual(events[-1].kind, "stage_start")
            return original(state, q)

        from dataclasses import replace
        reporter = ProgressReporter(progress_callback=events.append, arithmetic=arithmetic)
        result = solve_newton(replace(system, residual=residual), progress=reporter)
        self.assertTrue(result.converged)
        self.assertEqual(counts, dict(residual=1, jacobian=0, linear=0))
        self.assertFalse(any(event.kind == "iteration_start" for event in events))
        self.assertTrue(events[-1].data["converged"])

    def test_fixed_stage_reports_native_parameter_and_global_iteration_labels(self):
        arithmetic = MpmathArithmetic(30)
        events = []
        parameter = arithmetic.scalar("0.375")
        system = NonlinearSystem(
            residual=lambda z, q: arithmetic.vector([z[0]-q]),
            jacobian=lambda z, q: arithmetic.matrix([[1]]),
            solve_linear=arithmetic.solve, residual_norm=arithmetic.norm,
            initial_guess=lambda: arithmetic.vector([0]), scalar=arithmetic.scalar,
            isfinite=arithmetic.isfinite,
        )
        result = solve_newton(
            system, parameter=parameter, stage=4, iteration_offset=11, atol="1e-22",
            progress=ProgressReporter(progress_callback=events.append, arithmetic=arithmetic))
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.history[0].iteration, 12)
        self.assertEqual(result.history[0].stage, 4)
        self.assertEqual(result.homotopy, parameter)
        starts = [event for event in events if event.kind == "stage_start"]
        ends = [event for event in events if event.kind == "stage_end"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0].data["converged"])
        self.assertTrue(all(event.data["homotopy"] == parameter
                            for event in events if "homotopy" in event.data))
        self.assertTrue(all(event.data["stage"] == 4
                            for event in events if "stage" in event.data))
        self.assertTrue(all(event.data["iteration"] == 12
                            for event in events if "iteration" in event.data))
        self.assertFalse(any(event.kind in ("restart_start", "homotopy_transition")
                             for event in events))

    def test_adjoint_trials_and_accepted_steps_have_distinct_events(self):
        arithmetic = MpmathArithmetic(30)
        events, accepted = [], []
        system, _ = self.system(arithmetic, fail_first=True)
        result = solve_newton(
            system, atol="1e-22", newton_fallback="adjoint",
            callback=lambda state, record: accepted.append(record),
            progress=ProgressReporter(progress_callback=events.append, arithmetic=arithmetic))
        self.assertTrue(result.converged)
        trials = [event for event in events if event.kind == "trial_evaluation"]
        self.assertTrue(trials)
        self.assertTrue(all("accepted" not in event.data for event in trials))
        recovery = [event for event in events if event.kind == "recovery_end"]
        self.assertTrue(recovery[0].data["accepted"])
        self.assertEqual(len(accepted), len(result.history))
        completed = [event for event in events if event.kind == "iteration_end"]
        self.assertEqual(sum(event.data["step_kind"] == "adjoint" for event in completed),
                         result.adjoint_steps)

    def test_numeric_shaped_observer_errors_are_not_treated_as_solver_failures(self):
        arithmetic = NumpyArithmetic()
        system, counts = self.system(arithmetic)

        def observer(event):
            if event.kind == "phase_start" and event.data.get("phase") == "jacobian":
                raise OverflowError("observer defect")

        with self.assertRaises(ProgressCallbackError) as caught:
            solve_newton(system,
                         progress=ProgressReporter(progress_callback=observer,
                                                   arithmetic=arithmetic))
        self.assertIsInstance(caught.exception.__cause__, OverflowError)
        self.assertEqual(counts, dict(residual=1, jacobian=0, linear=0))

    def test_initializer_events_preserve_the_combined_budget_and_selected_state(self):
        reference = None
        for verbosity in range(4):
            with self.subTest(verbosity=verbosity):
                arithmetic = NumpyArithmetic()
                system, _ = self.system(arithmetic)
                events = []
                reporter = ProgressReporter(
                    verbosity=verbosity, progress_callback=events.append,
                    stream=io.StringIO(), arithmetic=arithmetic)
                result = initialize_fsolve(system, np.array([0.1]), max_evaluations=40,
                                           progress=reporter)
                actual = (tuple(result.state), result.accepted, result.reason,
                          result.evaluations, result.function_evaluations,
                          result.jacobian_evaluations, result.final_norm)
                if reference is None:
                    reference = actual
                else:
                    self.assertEqual(actual, reference)
                self.assertEqual(events[0].kind, "initializer_start")
                self.assertEqual(events[-1].kind, "initializer_end")
                self.assertLessEqual(result.evaluations, 40)
                self.assertFalse(any(event.kind == "iteration_end" for event in events))

    def test_direct_scipy_progress_reports_evaluations_and_verification_not_iterations(self):
        from overscreening_breakdown.collocation.spectral1d import solve_nonlinear_problem
        from overscreening_breakdown.manufactured import rational_case

        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                events = []
                solver = solve_nonlinear_problem(
                    6, problem=rational_case(nonlinear=True, initial="exp").make_problem(),
                    use_method=method, tolerance=1e-10, use_globalization=False,
                    progress_callback=events.append)
                text = io.StringIO()
                with redirect_stdout(text):
                    solver.solve_problem()
                self.assertTrue(solver.is_converged())
                self.assertEqual(text.getvalue(), "")
                self.assertTrue(any(event.kind == "scipy_evaluation" for event in events))
                self.assertTrue(any(event.kind == "scipy_stop" for event in events))
                self.assertTrue(any(event.kind == "scipy_verification" for event in events))
                self.assertFalse(any(event.kind == "iteration_end" for event in events))
                self.assertEqual(events[-1].kind, "solve_end")


if __name__ == "__main__":
    unittest.main()
