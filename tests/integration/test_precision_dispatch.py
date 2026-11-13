"""Binding and observer boundaries preserve the requested numerical owner."""

import unittest
from unittest.mock import Mock, patch

import mpmath as mp

from overscreening_breakdown import MpmathArithmetic, solve
from overscreening_breakdown.diagnostics import ProgressReporter
from overscreening_breakdown.manufactured import rational_case
from overscreening_breakdown.manufactured.common import NativeFluxDescription


class PrecisionDispatchTests(unittest.TestCase):
    def setUp(self):
        self.original_global_bits = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", self.original_global_bits)

    def description(self, owners):
        original = rational_case(nonlinear=False).make_problem()

        class CaptureOwner:
            def bind(self, arithmetic):
                owners.append(arithmetic)
                return original.bind(arithmetic)

        return CaptureOwner()

    def test_custom_binding_cannot_change_precision_and_stamp_a_fresh_snapshot(self):
        original = rational_case().make_problem()

        class ChangeOwner:
            def bind(self, arithmetic):
                arithmetic.context.prec += 1
                return original.bind(arithmetic)

        for solver in ("collocation", "galerkin"):
            with self.subTest(solver=solver), self.assertRaisesRegex(
                    RuntimeError, "physical problem binding precision changed.*bits"):
                solve(ChangeOwner(), 4, solver=solver, use_mpmath=True, precision=30)
        self.assertEqual(mp.mp.prec, self.original_global_bits)

    def test_native_parameter_callbacks_cannot_mix_precisions_during_binding(self):
        for changed in ("delta", "boundaries"):
            a, calls = MpmathArithmetic(50), []

            def delta(owner):
                calls.append("delta")
                value = owner.scalar(2)
                if changed == "delta":
                    owner.context.prec += 1
                return value

            def boundaries(owner):
                calls.append("boundaries")
                value = ((owner.scalar(1), None, None, 0, None), (None,)*5)
                if changed == "boundaries":
                    owner.context.prec += 1
                return value

            description = NativeFluxDescription(delta, boundaries, lambda x, u, a: u,
                                                lambda x, u, a: a.scalar(1))
            with self.subTest(callback=changed), self.assertRaisesRegex(
                    RuntimeError, "manufactured problem binding precision changed"):
                description.bind(a)
            self.assertEqual(calls, ["delta"] if changed == "delta" else ["delta", "boundaries"])

    def test_final_observer_cannot_make_a_converged_return_value_stale(self):
        for solver, resolution in (("collocation", 8), ("galerkin", 4)):
            owners, events = [], []

            def observer(event):
                events.append(event.kind)
                if event.kind == "solve_end":
                    owners[0].context.prec += 1

            with self.subTest(solver=solver), self.assertRaisesRegex(
                    RuntimeError, "precision changed.*bits"):
                solve(self.description(owners), resolution, solver=solver,
                      use_mpmath=True, precision=30, tolerance="1e-20",
                      use_globalization=False, progress_callback=observer)
            self.assertEqual(events.count("solve_end"), 1)
            self.assertEqual(mp.mp.prec, self.original_global_bits)

    def test_start_observer_change_stops_before_problem_binding(self):
        a = MpmathArithmetic(30)

        def observer(event):
            if event.kind == "solve_start":
                a.context.prec += 1

        with patch("overscreening_breakdown.solvers.MpmathArithmetic", return_value=a), \
                patch("overscreening_breakdown.solvers._bind") as bind:
            with self.assertRaisesRegex(RuntimeError, "precision changed.*bits"):
                solve(rational_case().make_problem(), 4, solver="galerkin",
                      use_mpmath=True, precision=30, progress_callback=observer)
            bind.assert_not_called()

    def test_phase_observer_change_stops_before_basis_construction(self):
        owners = []

        def observer(event):
            if event.kind == "phase_start" and event.data.get("phase") == "basis":
                owners[0].context.prec += 1

        with patch("overscreening_breakdown.solvers.GalerkinBasis") as basis:
            with self.assertRaisesRegex(RuntimeError, "precision changed.*bits"):
                solve(self.description(owners), 4, solver="galerkin",
                      use_mpmath=True, precision=30, progress_callback=observer)
            basis.assert_not_called()

    def test_initial_projection_checks_precision_before_converting_callback_result(self):
        calls = []

        def guess(x, arithmetic):
            calls.append(x)
            arithmetic.context.prec += 1
            return "nan"

        with self.assertRaisesRegex(RuntimeError, "precision changed.*bits"):
            solve(rational_case().make_problem(), 4, solver="galerkin",
                  use_mpmath=True, precision=30, initial_guess=guess)
        self.assertEqual(len(calls), 1)

    def test_reporter_views_keep_the_original_stamp_even_without_output(self):
        a, observer = MpmathArithmetic(50), Mock()
        reporter = ProgressReporter(arithmetic=a)
        view = reporter.bind(phase="example")
        observed = ProgressReporter(arithmetic=a, progress_callback=observer)
        a.context.prec += 1
        for operation in (lambda: reporter.emit("silent"), lambda: view.emit("silent"),
                          lambda: reporter.bind(phase="new"), lambda: reporter.wants("test"),
                          lambda: observed.emit("test")):
            with self.subTest(operation=operation), self.assertRaisesRegex(
                    RuntimeError, "precision changed.*bits"):
                operation()
        observer.assert_not_called()
