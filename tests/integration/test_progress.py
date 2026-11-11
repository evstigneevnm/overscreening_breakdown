"""Unified progress describes live work without changing the numerical solve."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest

import mpmath as mp

from overscreening_breakdown import solve
from overscreening_breakdown.diagnostics import ProgressCallbackError, ProgressEvent
from overscreening_breakdown.manufactured import rational_case


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)
        mp.mp.prec = 77

    def problem(self, coefficient="constant"):
        return rational_case(coefficient, nonlinear=True, beta="0.1", initial="exp").make_problem()

    def options(self, solver, use_mpmath=False, precision=30):
        return dict(solver=solver, use_mpmath=use_mpmath, precision=precision,
                    tolerance="1e-{}".format(precision-10) if use_mpmath else "1e-10",
                    use_globalization=False)

    def counted(self, problem):
        counts = {"f": 0, "f_u": 0}

        def wrap(callback, name):
            def evaluate(x, u, arithmetic):
                counts[name] += 1
                return callback(x, u, arithmetic)
            return evaluate

        return replace(problem, f=wrap(problem.f, "f"), f_u=wrap(problem.f_u, "f_u")), counts

    def test_all_verbosity_levels_preserve_both_solvers_and_backends(self):
        for solver, resolution in (("collocation", 6), ("galerkin", 4)):
            for use_mpmath in (False, True):
                original, original_counts = self.counted(self.problem())
                baseline = solve(original, resolution, **self.options(solver, use_mpmath))
                reference = (tuple(baseline.coefficients), baseline.newton_result.history,
                             baseline.newton_result.iterations, dict(original_counts))
                for level in range(4):
                    with self.subTest(solver=solver, mpmath=use_mpmath, verbosity=level):
                        problem, counts = self.counted(self.problem())
                        events, output = [], io.StringIO()
                        with redirect_stdout(output):
                            result = solve(problem, resolution, verbosity=level,
                                           progress_callback=events.append,
                                           **self.options(solver, use_mpmath))
                        self.assertTrue(result.converged)
                        fingerprint = (tuple(result.coefficients), result.newton_result.history,
                                       result.newton_result.iterations, dict(counts))
                        self.assertEqual(fingerprint, reference)
                        self.assertEqual(events[0].kind, "solve_start")
                        self.assertEqual(events[-1].kind, "solve_end")
                        self.assertTrue(events[-1].data["converged"])
                        self.assertTrue(all(isinstance(e, ProgressEvent) for e in events))
                        self.assertEqual(events[-1].data["norm"],
                                         "rms" if solver == "collocation" else "euclidean")
                        if level == 0:
                            self.assertEqual(output.getvalue(), "")
                        else:
                            self.assertIn("solve_start", output.getvalue())
                            self.assertIn("solve_end", output.getvalue())
                        if level >= 2:
                            self.assertIn("iteration_end", output.getvalue())
                        if level == 3:
                            self.assertIn("phase=jacobian", output.getvalue())
                            self.assertIn("phase=linear_solve", output.getvalue())
                        self.assertEqual(mp.mp.prec, 77)

    def test_scipy_reports_evaluations_without_fabricated_accepted_iterations(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            original, original_counts = self.counted(self.problem())
            baseline = solve(original, 6, method=method, use_globalization=False)
            reference = (tuple(baseline.coefficients), baseline.scipy_result, dict(original_counts))
            for level in range(4):
                with self.subTest(method=method, verbosity=level):
                    problem, counts = self.counted(self.problem())
                    events, output = [], io.StringIO()
                    with redirect_stdout(output):
                        result = solve(problem, 6, method=method, verbosity=level,
                                       progress_callback=events.append, progress_interval=0,
                                       use_globalization=False)
                    self.assertTrue(result.converged)
                    diagnostic = result.scipy_result
                    fingerprint = (tuple(result.coefficients), diagnostic, dict(counts))
                    self.assertEqual(fingerprint, reference)
                    evaluations = [e for e in events if e.kind == "scipy_evaluation"]
                    self.assertTrue(evaluations)
                    # Older SciPy versions exclude preflight calls from nfev;
                    # event counters describe actual wrapper evaluations and
                    # the stop event retains SciPy's separately reported count.
                    self.assertEqual([e.data["function_evaluations"] for e in evaluations],
                                     list(range(1, len(evaluations)+1)))
                    self.assertFalse(any(e.kind in ("iteration_start", "iteration_end") for e in events))
                    stopped = [e for e in events if e.kind == "scipy_stop"]
                    self.assertEqual(len(stopped), 1)
                    self.assertEqual(stopped[0].data["nfev"], diagnostic.nfev)
                    verified = [e for e in events if e.kind == "scipy_verification"]
                    self.assertEqual(verified[-1].data["residual_norm"], diagnostic.residual_norm)
                    self.assertEqual(verified[-1].data["target_norm"], diagnostic.target_norm)
                    self.assertEqual(events[-1].data["scipy_success"], diagnostic.scipy_success)

    def test_initializer_budget_and_outcome_are_distinct_from_main_solve(self):
        original, original_counts = self.counted(self.problem())
        baseline = solve(original, 6, method="root_no_jac", initializer="fsolve",
                         initializer_max_evaluations=40, use_globalization=False)
        reference = (tuple(baseline.coefficients), baseline.scipy_result, dict(original_counts),
                     baseline.initialization_result.evaluations)
        for level in (0, 3):
            with self.subTest(verbosity=level):
                problem, counts = self.counted(self.problem())
                events = []
                with redirect_stdout(io.StringIO()):
                    result = solve(problem, 6, method="root_no_jac", initializer="fsolve",
                                   initializer_max_evaluations=40, verbosity=level,
                                   progress_callback=events.append, use_globalization=False)
                self.assertTrue(result.converged)
                end = next(e for e in events if e.kind == "initializer_end")
                self.assertEqual(end.data["evaluations"], result.initialization_result.evaluations)
                self.assertLessEqual(end.data["evaluations"], 40)
                self.assertEqual(end.data["accepted"], result.initialization_result.accepted)
                self.assertTrue(any(e.kind == "scipy_evaluation" for e in events))
                fingerprint = (tuple(result.coefficients), result.scipy_result, dict(counts),
                               result.initialization_result.evaluations)
                self.assertEqual(fingerprint, reference)

    def test_header_and_phase_start_are_visible_before_source_evaluation(self):
        for solver, resolution in (("collocation", 6), ("galerkin", 4)):
            with self.subTest(solver=solver):
                original = self.problem()
                events, output, checked = [], io.StringIO(), []

                def source(x, u, arithmetic):
                    if not checked:
                        self.assertEqual(events[0].kind, "solve_start")
                        self.assertTrue(any(e.kind == "phase_start" for e in events))
                        self.assertIn("solve_start", output.getvalue())
                        self.assertIn("phase_start", output.getvalue())
                        checked.append(True)
                    return original.f(x, u, arithmetic)

                with redirect_stdout(output):
                    result = solve(replace(original, f=source), resolution, verbosity=1,
                                   progress_callback=events.append, **self.options(solver))
                self.assertTrue(result.converged)
                self.assertTrue(checked)

    def test_quiet_reporting_does_not_suppress_user_callback_output(self):
        for solver, resolution in (("collocation", 6), ("galerkin", 4)):
            with self.subTest(solver=solver):
                original = self.problem()
                called = []

                def source(x, u, arithmetic):
                    if not called:
                        print("user source was evaluated")
                        called.append(True)
                    return original.f(x, u, arithmetic)

                output = io.StringIO()
                with redirect_stdout(output):
                    result = solve(replace(original, f=source), resolution,
                                   verbosity=0, **self.options(solver))
                self.assertTrue(result.converged)
                self.assertEqual(output.getvalue(), "user source was evaluated\n")

    def test_native_event_values_remain_in_the_solution_owner_after_restore(self):
        for solver, resolution in (("collocation", 6), ("galerkin", 4)):
            with self.subTest(solver=solver):
                events = []
                result = solve(self.problem(), resolution, progress_callback=events.append,
                               **self.options(solver, True, 120))
                self.assertTrue(result.converged)
                self.assertEqual(mp.mp.prec, 77)
                native_values = []

                def inspect(value):
                    if isinstance(value, tuple):
                        for item in value:
                            inspect(item)
                    elif hasattr(value, "_mpf_"):
                        native_values.append(value)
                        self.assertIs(value.context, result.arithmetic.context)

                for event in events:
                    for value in event.data.values():
                        inspect(value)
                self.assertTrue(native_values)
                targets = [e.data["target_norm"] for e in events if "target_norm" in e.data]
                self.assertTrue(targets)
                self.assertLess(targets[-1], result.arithmetic.scalar("1e-100"))

    def test_normal_failures_report_their_distinct_acceptance_conditions(self):
        for solver, resolution in (("collocation", 6), ("galerkin", 4)):
            with self.subTest(solver=solver):
                events = []
                result = solve(self.problem(), resolution, max_iterations=0,
                               progress_callback=events.append, **self.options(solver))
                self.assertFalse(result.converged)
                self.assertEqual(events[-1].kind, "solve_end")
                self.assertEqual(events[-1].data["reason"], "max_iterations")
                self.assertFalse(events[-1].data["nonlinear_converged"])
        events = []
        result = solve(self.problem("exponential"), 4, solver="galerkin", quadrature_order=5,
                       quadrature_atol="1e-10", quadrature_rtol="1e-10", max_refinements=0,
                       progress_callback=events.append)
        self.assertTrue(result.nonlinear_converged)
        self.assertFalse(result.quadrature_converged)
        self.assertFalse(result.converged)
        self.assertEqual(events[-1].data["reason"], "quadrature_not_converged")
        checks = [e for e in events if e.kind == "quadrature_check"]
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0].data["converged"])
        for name in ("residual_difference", "residual_target", "jacobian_difference", "jacobian_target"):
            self.assertIn(name, checks[0].data)

    def test_exact_restart_still_reports_final_verification_without_iterations(self):
        for solver, resolution in (("collocation", 6), ("galerkin", 4)):
            with self.subTest(solver=solver):
                problem = self.problem()
                first = solve(problem, resolution, **self.options(solver))
                events = []
                result = solve(problem, resolution, initial_guess=first,
                               progress_callback=events.append, **self.options(solver))
                self.assertTrue(result.converged)
                self.assertEqual(result.newton_result.iterations, 0)
                self.assertFalse(any(e.kind == "iteration_end" for e in events))
                self.assertTrue(any(e.kind == "stage_start" for e in events))
                self.assertTrue(events[-1].data["converged"])
                self.assertEqual(events[-1].data["homotopy"], 0)
                if solver == "galerkin":
                    self.assertTrue(any(e.kind == "quadrature_check" for e in events))

    def test_callback_exceptions_cannot_be_misclassified_as_numeric_failures(self):
        configurations = (
            ("collocation", dict()),
            ("collocation", dict(use_mpmath=True, precision=30, tolerance="1e-20")),
            ("galerkin", dict(use_mpmath=True, precision=30, tolerance="1e-20")),
            ("collocation", dict(method="root_no_jac")),
            ("collocation", dict(initializer="fsolve")),
        )
        for solver, options in configurations:
            with self.subTest(solver=solver, options=options):
                calls = []

                def observer(event):
                    calls.append(event)
                    numeric_phase = (event.kind == "phase_start"
                                     and event.data.get("phase") == "jacobian")
                    if numeric_phase or event.kind in ("scipy_evaluation", "initializer_evaluation"):
                        raise OverflowError("observer defect")

                with self.assertRaisesRegex(ProgressCallbackError, "observer defect") as failure:
                    solve(self.problem(), 6 if solver == "collocation" else 4, solver=solver,
                          progress_callback=observer, use_globalization=False, **options)
                self.assertIsInstance(failure.exception.__cause__, OverflowError)
                self.assertFalse(any(e.kind == "solve_end" for e in calls))
                self.assertEqual(mp.mp.prec, 77)

    def test_model_exceptions_propagate_and_are_reported_as_exceptions(self):
        for solver, resolution in (("collocation", 6), ("galerkin", 4)):
            with self.subTest(solver=solver):
                events = []

                def broken(x, u, arithmetic):
                    raise RuntimeError("model callback defect")

                with self.assertRaisesRegex(RuntimeError, "model callback defect"):
                    solve(replace(self.problem(), f=broken), resolution,
                          progress_callback=events.append, **self.options(solver))
                self.assertEqual(events[-1].kind, "solve_error")
                self.assertEqual(events[-1].data["error_type"], "RuntimeError")
                self.assertFalse(any(e.kind == "solve_end" for e in events))

    def test_notebook_interruption_is_reported_and_restores_caller_precision(self):
        for solver, resolution in (("collocation", 6), ("galerkin", 4)):
            with self.subTest(solver=solver):
                events, output = [], io.StringIO()

                def interrupt(event):
                    events.append(event)
                    if event.kind == "stage_start":
                        raise KeyboardInterrupt

                with redirect_stdout(output), self.assertRaises(KeyboardInterrupt):
                    solve(self.problem(), resolution, verbosity=1,
                          progress_callback=interrupt, **self.options(solver, True, 50))
                self.assertEqual(mp.mp.prec, 77)
                self.assertIn("solve_interrupted", output.getvalue())
                self.assertFalse(any(e.kind == "solve_end" for e in events))

    def test_removed_verbose_keyword_and_bool_verbosity_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "verbose"):
            solve(self.problem(), 6, verbose=True)
        for value in (True, False):
            with self.subTest(verbosity=value), self.assertRaises(ValueError):
                solve(self.problem(), 6, verbosity=value)


if __name__ == "__main__":
    unittest.main()
