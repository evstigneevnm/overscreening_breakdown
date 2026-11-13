"""Behavioral guarantees of reference construction and adaptive continuation."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import numpy as np

from overscreening_breakdown import ContinuationOptions, ScreenedLinearReference
from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.continuation import (
    AffineReference, ResidualHomotopy, solve_continuation,
)
from overscreening_breakdown.diagnostics import ProgressReporter
from overscreening_breakdown.nonlinear_solver import (
    LinearSolveError, NonlinearSystem, solve_newton,
)


class ContinuationTests(unittest.TestCase):
    def system(self, arithmetic, *, wall=False, initial=0):
        counts = dict(residual=0, jacobian=0, linear=0, reference=0)

        def residual(state, q):
            self.assertEqual(q, 0, "physical callbacks never receive a homotopy parameter")
            counts["residual"] += 1
            values = [state[1 if wall else 0]**3-1]
            return arithmetic.vector([state[0]-100]+values if wall else values)

        def jacobian(state, q):
            self.assertEqual(q, 0)
            counts["jacobian"] += 1
            return arithmetic.matrix([[1, 0], [0, 3*state[1]**2]] if wall
                                     else [[3*state[0]**2]])

        def linear(matrix, rhs):
            counts["linear"] += 1
            try:
                return arithmetic.solve(matrix, rhs)
            except (np.linalg.LinAlgError, ZeroDivisionError) as error:
                raise LinearSolveError(str(error)) from error

        system = NonlinearSystem(
            residual=residual, jacobian=jacobian, solve_linear=linear,
            residual_norm=arithmetic.norm, scalar=arithmetic.scalar,
            isfinite=arithmetic.isfinite,
            initial_guess=lambda: arithmetic.vector([100, initial] if wall else [initial]),
            restart=lambda *args: self.fail("continuation must not invoke the old restart callback"),
        )

        def reference(strength):
            counts["reference"] += 1
            return (arithmetic.matrix([[1, 0], [0, strength]] if wall else [[strength]]),
                    arithmetic.vector([-100, -strength/4] if wall else [-strength/4]))

        return system, reference, counts

    def tolerance(self, arithmetic):
        return "1e-{}".format(max(9, arithmetic.dps-8)) if arithmetic.use_mpmath else "1e-12"

    def test_recovery_and_verified_physical_endpoint_in_all_precisions(self):
        owners = [NumpyArithmetic()]+[MpmathArithmetic(dps) for dps in (15, 30, 50, 100, 120)]
        for a in owners:
            with self.subTest(mpmath=a.use_mpmath, dps=a.dps):
                system, factory, counts = self.system(a, wall=True)
                events = []
                result = solve_continuation(
                    system, reference_factory=factory, atol=self.tolerance(a),
                    max_iterations=30,
                    progress=ProgressReporter(progress_callback=events.append, arithmetic=a),
                )
                self.assertTrue(result.converged)
                self.assertEqual(result.homotopy, 0)
                self.assertFalse(result.stages[0].converged)
                self.assertEqual(result.stages[0].reason, "linear_solve_failed")
                self.assertTrue(result.reference_result.converged)
                self.assertEqual(counts["reference"], 1)
                self.assertGreater(result.continuation_attempts, 0)
                self.assertTrue(any(stage.homotopy > 0 for stage in result.stages))
                self.assertTrue(result.stages[-1].converged)
                self.assertEqual(result.stages[-1].homotopy, 0)
                self.assertEqual(result.state[0], 100)
                self.assertLess(abs(result.state[1]-1), a.scalar(self.tolerance(a)))
                self.assertLessEqual(result.residual_norm, result.stages[-1].target_norm)
                self.assertEqual(result.iterations, sum(stage.iterations for stage in result.stages))
                self.assertEqual(result.continuation_attempts, len(result.stages)-1)
                labels = [record.iteration for record in result.history]
                self.assertEqual(labels, sorted(set(labels)))
                for value in list(result.reference_result.state)+[result.reference_result.strength]:
                    self.assertIsInstance(value, type(a.scalar(0)))
                self.assertTrue(any(event.kind == "reference_end" for event in events))
                self.assertTrue(any(event.kind == "physical_verification_end"
                                    and event.data["converged"] for event in events))
                self.assertEqual(events[-1].kind, "continuation_end")

    def test_direct_success_and_zero_budget_exact_restart_do_not_build_reference(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            system, factory, counts = self.system(a, initial=1)
            result = solve_continuation(
                system, reference_factory=factory,
                options=ContinuationOptions(max_total_iterations=0, max_stages=0))
            self.assertTrue(result.converged)
            self.assertEqual(result.iterations, 0)
            self.assertEqual(counts["reference"], 0)
            self.assertEqual(counts["residual"], 2)  # Stage evaluation and fresh verification.
            self.assertIsNone(result.reference_result)

    def test_failed_stage_rolls_back_state_and_parameter_and_reduces_step(self):
        a = MpmathArithmetic(50)
        system, factory, _ = self.system(a, wall=True)
        starts = []
        rejected_state = []

        def stage_solver(stage_system, **options):
            q = options.get("parameter", 0)
            starts.append((q, tuple(stage_system.initial_guess())))
            if q == a.scalar("0.75"):
                options["max_iterations"] = 1
                result = solve_newton(stage_system, **options)
                self.assertFalse(result.converged)
                rejected_state.append(tuple(result.state))
                return result
            return solve_newton(stage_system, **options)

        events = []
        with patch("overscreening_breakdown.continuation.solve_newton", side_effect=stage_solver):
            result = solve_continuation(
                system, reference_factory=factory, atol="1e-40", max_iterations=30,
                progress=ProgressReporter(progress_callback=events.append, arithmetic=a))
        self.assertTrue(result.converged)
        self.assertEqual(starts[1][0], a.scalar("0.75"))
        self.assertEqual(starts[2][0], a.scalar("0.875"))
        self.assertEqual(starts[1][1], starts[2][1])
        self.assertNotEqual(rejected_state[0], starts[2][1])
        self.assertTrue(all(state[0] == 100 for _, state in starts))
        rejected = [event for event in events if event.kind == "continuation_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].data["accepted_homotopy"], 1)

    def test_stage_and_global_iteration_limits_never_certify_positive_parameter(self):
        for options, reason in (
            (ContinuationOptions(max_stages=1), "max_stages"),
            (ContinuationOptions(max_total_iterations=2), "max_total_iterations"),
            (ContinuationOptions(max_total_iterations=0), "max_total_iterations"),
        ):
            a = NumpyArithmetic()
            system, factory, _ = self.system(a)
            result = solve_continuation(system, reference_factory=factory, options=options)
            self.assertFalse(result.converged)
            self.assertEqual(result.reason, reason)
            self.assertLessEqual(result.iterations, options.max_total_iterations)
            self.assertLessEqual(result.continuation_attempts, options.max_stages)

    def test_repeated_rejection_is_bounded_and_returns_last_accepted_reference(self):
        a = NumpyArithmetic()
        system, factory, _ = self.system(a, wall=True)

        def fail_stage(stage_system, **options):
            if options.get("parameter", 0) > 0:
                options["max_iterations"] = 0
            return solve_newton(stage_system, **options)

        with patch("overscreening_breakdown.continuation.solve_newton", side_effect=fail_stage):
            result = solve_continuation(
                system, reference_factory=factory,
                options=ContinuationOptions(min_step="0.1"))
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "min_parameter_step")
        self.assertEqual(result.homotopy, 1)
        self.assertEqual(tuple(result.state), (100, 0.25))
        self.assertEqual(result.continuation_attempts, 2)

    def test_reference_failure_is_reported_without_masking_programming_errors(self):
        a = NumpyArithmetic()
        system, _, _ = self.system(a)
        singular = lambda strength: (a.matrix([[0]]), a.vector([1]))
        result = solve_continuation(system, reference_factory=singular)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "reference_construction_failed")

        def invalid(strength):
            raise RuntimeError("user reference implementation failed")

        with self.assertRaisesRegex(RuntimeError, "user reference"):
            solve_continuation(system, reference_factory=invalid)

    def test_reference_solution_requires_residual_verification(self):
        a = NumpyArithmetic()
        system, factory, _ = self.system(a)
        original = system.solve_linear

        def wrong_linear_solution(matrix, rhs):
            # Preserve the initial singular-Jacobian failure, then return an
            # incorrect finite reference solution rather than raising.
            if matrix[0, 0] == 0:
                return original(matrix, rhs)
            return a.vector([5])

        result = solve_continuation(replace(system, solve_linear=wrong_linear_solution),
                                    reference_factory=factory)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "reference_not_converged")
        self.assertFalse(result.reference_result.converged)
        self.assertEqual(result.continuation_attempts, 0)

    def test_reference_roundoff_is_corrected_without_relaxing_the_target(self):
        a = MpmathArithmetic(50)
        system, factory, _ = self.system(a)
        original = system.solve_linear
        perturbed = [False]

        def linear(matrix, rhs):
            answer = original(matrix, rhs)
            if matrix[0, 0] == 1 and not perturbed[0]:
                perturbed[0] = True
                answer[0] += a.scalar("1e-8")
            return answer

        events = []
        result = solve_continuation(
            replace(system, solve_linear=linear), reference_factory=factory, atol="1e-40",
            progress=ProgressReporter(progress_callback=events.append, arithmetic=a))
        self.assertTrue(result.converged)
        self.assertTrue(perturbed[0])
        self.assertEqual(result.reference_result.linear_solves, 2)
        self.assertEqual(result.reference_result.target_norm, a.scalar("1e-40"))
        self.assertLessEqual(result.reference_result.residual_norm, a.scalar("1e-40"))
        timed_solves = [event for event in events if event.kind == "phase_end"
                        and event.data.get("phase") == "reference_linear_solve"]
        self.assertEqual(len(timed_solves), result.reference_result.linear_solves)
        self.assertEqual(timed_solves[-1].data["correction"], 1)
        self.assertTrue(all(event.data["success"] for event in timed_solves))

    def test_adjoint_recovery_operates_at_the_frozen_intermediate_parameter(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            with self.subTest(dps=a.dps, mpmath=a.use_mpmath):
                system, factory, _ = self.system(a)
                injected = [False]

                def stage_solver(stage_system, **options):
                    if options.get("parameter", 0) > 0 and not injected[0]:
                        original = stage_system.solve_linear
                        calls = [0]

                        def linear(matrix, rhs):
                            calls[0] += 1
                            if calls[0] == 1:
                                injected[0] = True
                                raise LinearSolveError("exercise recovery at positive q")
                            return original(matrix, rhs)

                        stage_system = replace(stage_system, solve_linear=linear)
                    return solve_newton(stage_system, **options)

                with patch("overscreening_breakdown.continuation.solve_newton",
                           side_effect=stage_solver):
                    result = solve_continuation(
                        system, reference_factory=factory, atol=self.tolerance(a),
                        newton_fallback="adjoint")
                self.assertTrue(result.converged)
                self.assertTrue(injected[0])
                self.assertTrue(any(record.kind == "adjoint" and record.homotopy > 0
                                    for record in result.history))
                self.assertEqual(result.adjoint_steps,
                                 sum(stage.adjoint_steps for stage in result.stages))
                self.assertEqual(result.adjoint_evaluations,
                                 sum(stage.adjoint_evaluations for stage in result.stages))

    def test_a_stage_success_flag_cannot_replace_fresh_physical_verification(self):
        a = NumpyArithmetic()
        system, factory, _ = self.system(a)
        fake = solve_newton(system, max_iterations=0)
        fake = replace(fake, converged=True, residual_norm=a.scalar(0))
        with patch("overscreening_breakdown.continuation.solve_newton", return_value=fake):
            result = solve_continuation(system, reference_factory=factory)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "physical_verification_failed")
        self.assertEqual(result.residual_norm, 1)

    def test_proxy_endpoints_do_not_evaluate_unused_callbacks_and_copy_reference_data(self):
        a = NumpyArithmetic()
        system, factory, counts = self.system(a)
        matrix, offset = factory(a.scalar(1))
        reference = AffineReference(matrix, offset, system)
        proxy = ResidualHomotopy(system, reference)
        matrix[0, 0], offset[0] = 999, 999
        physical_counts = counts.copy()
        self.assertEqual(proxy.residual(a.vector([1]), 1)[0], 0.75)
        self.assertEqual(proxy.jacobian(a.vector([1]), 1)[0, 0], 1)
        self.assertEqual(counts, physical_counts)
        with patch.object(reference, "residual", side_effect=AssertionError("unused reference")):
            self.assertEqual(proxy.residual(a.vector([1]), 0)[0], 0)
        for q in (-1, 2, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                proxy.residual(a.vector([1]), q)

    def test_blended_jacobian_and_adjoint_merit_use_the_complete_current_system(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50), MpmathArithmetic(120)):
            with self.subTest(dps=a.dps, mpmath=a.use_mpmath):
                system, factory, _ = self.system(a, wall=True)
                matrix, offset = factory(a.scalar("1.7"))
                system = replace(system, merit_gradient=lambda r, j: a.transpose_matvec(j, r),
                                 dot=a.dot, adjoint_direction=lambda g: -g)
                reference = AffineReference(matrix, offset, system)
                proxy = ResidualHomotopy(system, reference)
                state, direction = a.vector(["100.2", "0.4"]), a.vector(["0.3", "-0.7"])
                h = a.scalar("1e-{}".format(a.dps//3) if a.use_mpmath else "1e-5")
                bound = a.scalar("1e-{}".format(a.dps//2) if a.use_mpmath else "1e-8")
                for text in ("0.2", "0.7"):
                    q = a.scalar(text)
                    residual, jacobian = proxy.residual(state, q), proxy.jacobian(state, q)
                    finite_difference = (proxy.residual(state+h*direction, q)
                                         -proxy.residual(state-h*direction, q))/(2*h)
                    product = jacobian*direction if a.use_mpmath else jacobian@direction
                    self.assertLess(a.norm(finite_difference-product), bound)
                    geometry = proxy.nonlinear_system(state)
                    gradient = geometry.merit_gradient(residual, jacobian)
                    merit_difference = (a.norm(proxy.residual(state+h*direction, q))**2
                                        -a.norm(proxy.residual(state-h*direction, q))**2)/(4*h)
                    self.assertLess(abs(a.dot(gradient, direction)-merit_difference), bound)

    def test_verbosity_does_not_change_work_or_native_results(self):
        a = MpmathArithmetic(30)
        baseline = None
        for level in range(4):
            system, factory, counts = self.system(a)
            stream = io.StringIO()
            with redirect_stdout(io.StringIO()):
                result = solve_continuation(
                    system, reference_factory=factory, atol="1e-22",
                    progress=ProgressReporter(verbosity=level, arithmetic=a,
                                              stream=stream, progress_interval=0))
            snapshot = (tuple(result.state), result.history, counts)
            if baseline is None:
                baseline = snapshot
            else:
                self.assertEqual(snapshot, baseline)
            self.assertEqual(stream.getvalue() == "", level == 0)

    def test_option_validation_and_native_integer_budgets(self):
        a = NumpyArithmetic()
        for options in (
            ContinuationOptions(initial_step=0), ContinuationOptions(min_step="0.5"),
            ContinuationOptions(max_step=2), ContinuationOptions(growth=1),
            ContinuationOptions(max_stages=-1), ContinuationOptions(max_total_iterations=True),
            ContinuationOptions(max_reference_corrections=-1),
            ContinuationOptions(reference=ScreenedLinearReference(strength=0)),
            ContinuationOptions(reference=ScreenedLinearReference(strength=float("nan"))),
        ):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                options.bind(a.scalar, a.isfinite)
        system, factory, _ = self.system(a)
        result = solve_continuation(
            system, reference_factory=factory,
            options=ContinuationOptions(max_total_iterations=np.int64(2)))
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "max_total_iterations")

