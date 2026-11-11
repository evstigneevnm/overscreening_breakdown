"""Streaming progress is immutable, scoped, and independent of numerical work."""

from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from decimal import Decimal
import io
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.diagnostics import (
    ProgressCallbackError, ProgressEvent, ProgressReporter,
)


class Clock:
    def __init__(self):
        self.time = 0.0

    def __call__(self):
        return self.time


class FlushedStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        super().flush()


class DiagnosticsTests(unittest.TestCase):
    def test_quiet_mode_and_observer_are_independent(self):
        stream = io.StringIO()
        quiet = ProgressReporter(stream=stream)
        self.assertFalse(quiet.enabled)
        self.assertFalse(quiet.wants("iteration", level=2))
        quiet.emit("solve_start")
        self.assertEqual(stream.getvalue(), "")
        events = []
        observed = ProgressReporter(progress_callback=events.append, stream=stream)
        self.assertTrue(observed.enabled)
        self.assertTrue(observed.wants("trial", level=3, throttle=True))
        for level in (1, 2, 3):
            observed.emit("event_" + str(level), level=level, throttle=True)
        self.assertEqual([event.level for event in events], [1, 2, 3])
        self.assertEqual(stream.getvalue(), "")

    def test_verbosity_filters_text_without_changing_observations(self):
        for level in range(4):
            with self.subTest(verbosity=level):
                events, output = [], io.StringIO()
                reporter = ProgressReporter(level, events.append, stream=output)
                for event_level in (1, 2, 3):
                    reporter.emit("level_" + str(event_level), level=event_level)
                self.assertEqual(len(events), 3)
                self.assertEqual(len(output.getvalue().splitlines()), level)

    def test_bound_context_is_scoped_and_event_data_is_immutable(self):
        events = []
        clock = Clock()
        reporter = ProgressReporter(progress_callback=events.append, solver="galerkin",
                                    method="newton", clock=clock)
        child = reporter.bind(phase="nonlinear", stage=1, interfaces=(1, 2))
        clock.time = 2.5
        child.emit("iteration", level=2, residual_norm=np.float64("0.125"))
        reporter.emit("solve_end", converged=True)
        first, last = events
        self.assertEqual((first.solver, first.method, first.elapsed), ("galerkin", "newton", 2.5))
        self.assertEqual(first.data["interfaces"], (1, 2))
        self.assertNotIn("phase", last.data)
        with self.assertRaises(TypeError):
            first.data["stage"] = 99
        with self.assertRaises(FrozenInstanceError):
            first.kind = "changed"
        for mutable in ([1], {"a": 1}, np.array([1.]), np.array(1.),
                        NumpyArithmetic(), MpmathArithmetic(30).vector([1])):
            with self.subTest(value_type=type(mutable)), self.assertRaises(TypeError):
                child.emit("bad", bad=mutable)

    def test_direct_event_snapshots_its_input_mapping(self):
        data = {"count": 1, "parameters": (("L", Decimal("0.125")),)}
        event = ProgressEvent("start", 1, "collocation", "newton", 0, data)
        data["count"] = 2
        self.assertEqual(event.data["count"], 1)
        with self.assertRaises(TypeError):
            ProgressEvent("start", 1, "", "", 0, {1: "not a named field"})

    def test_throttling_is_per_kind_and_phase_and_wants_does_not_consume_it(self):
        events, output, clock = [], io.StringIO(), Clock()
        reporter = ProgressReporter(2, events.append, stream=output, clock=clock,
                                    progress_interval=1)
        first = reporter.bind(phase="scipy")
        first.emit("evaluation", level=2, throttle=True, evaluations=1)
        clock.time = 0.5
        first.emit("evaluation", level=2, throttle=True, evaluations=2)
        reporter.bind(phase="initializer").emit("evaluation", level=2, throttle=True)
        first.emit("jacobian_evaluation", level=2, throttle=True)
        clock.time = 1
        first.emit("evaluation", level=2, throttle=True, evaluations=3)
        self.assertEqual(len(events), 5)
        self.assertEqual(len(output.getvalue().splitlines()), 4)

        plain = ProgressReporter(2, stream=io.StringIO(), clock=clock, progress_interval=1)
        self.assertTrue(plain.wants("evaluation", level=2, throttle=True))
        self.assertTrue(plain.wants("evaluation", level=2, throttle=True))
        plain.emit("evaluation", level=2, throttle=True)
        self.assertFalse(plain.wants("evaluation", level=2, throttle=True))
        clock.time += 1
        self.assertTrue(plain.wants("evaluation", level=2, throttle=True))
        plain.emit("evaluation", level=2, throttle=True)
        self.assertFalse(plain.wants("evaluation", level=2, throttle=True))

    def test_phase_and_solve_end_messages_are_not_throttled(self):
        output = io.StringIO()
        reporter = ProgressReporter(1, stream=output, clock=Clock(), progress_interval=100)
        for _ in range(2):
            reporter.emit("phase_start", throttle=True, phase="assembly")
            reporter.emit("phase_end", throttle=True, phase="assembly")
            reporter.emit("solve_end", throttle=True, converged=True)
        self.assertEqual(len(output.getvalue().splitlines()), 6)

    def test_phases_emit_before_work_and_preserve_failure_causes(self):
        events, clock = [], Clock()
        reporter = ProgressReporter(progress_callback=events.append, clock=clock)
        with reporter.phase("assembly", N=4) as scoped:
            self.assertEqual(events[0].kind, "phase_start")
            self.assertEqual(events[0].data["phase"], "assembly")
            clock.time = 2
            scoped.emit("row", level=3, index=1)
        self.assertEqual([event.kind for event in events], ["phase_start", "row", "phase_end"])
        self.assertEqual(events[-1].data["duration"], 2)
        self.assertTrue(events[-1].data["success"])
        with self.assertRaisesRegex(OverflowError, "outside numeric range"):
            with reporter.phase("residual", level=3):
                raise OverflowError("outside numeric range")
        self.assertEqual(events[-1].kind, "phase_end")
        self.assertFalse(events[-1].data["success"])
        self.assertEqual(events[-1].data["error_type"], "OverflowError")

    def test_observer_errors_are_not_numerical_failures_and_are_not_repeated(self):
        output, calls = io.StringIO(), []
        def broken(event):
            calls.append(event.kind)
            raise OverflowError("observer defect")
        reporter = ProgressReporter(1, broken, stream=output)
        with self.assertRaisesRegex(ProgressCallbackError, "observer defect") as raised:
            reporter.emit("solve_start")
        self.assertIsInstance(raised.exception.__cause__, OverflowError)
        reporter.emit("solve_error", reason="observer failed")
        self.assertEqual(calls, ["solve_start"])
        self.assertIn("solve_error", output.getvalue())

    def test_keyboard_interruption_propagates_without_reentering_observer(self):
        calls = []
        def cancel(event):
            calls.append(event.kind)
            raise KeyboardInterrupt
        reporter = ProgressReporter(progress_callback=cancel)
        with self.assertRaises(KeyboardInterrupt):
            reporter.emit("solve_start")
        reporter.emit("solve_error")
        self.assertEqual(calls, ["solve_start"])

    def test_native_precision_values_survive_formatting_and_global_context_changes(self):
        original_bits = mp.mp.prec
        high = MpmathArithmetic(120)
        value = high.scalar("1e-30000")
        correction = high.scalar(1)+high.scalar("1e-80")
        events, output = [], io.StringIO()
        reporter = ProgressReporter(2, events.append, arithmetic=high, stream=output)
        reporter.emit("iteration", level=2, norm="RMS", residual_norm=value,
                      parameter=correction, target_norm=high.scalar("1e-30001"))
        self.assertEqual(events[0].data["residual_norm"], value)
        self.assertEqual(events[0].data["parameter"], correction)
        self.assertIsInstance(events[0].data["parameter"], type(high.scalar(0)))
        self.assertIn("e-30000", output.getvalue())
        self.assertIn("norm=RMS", output.getvalue())
        self.assertEqual(mp.mp.prec, original_bits)

    def test_foreign_mpf_events_remain_owned_after_global_precision_is_restored(self):
        owner = MpmathArithmetic(120)
        events = []
        reporter = ProgressReporter(progress_callback=events.append, arithmetic=owner)
        original_bits = mp.mp.prec
        with mp.workdps(120):
            foreign = mp.mpf(1)+mp.mpf("1e-80")
            context = reporter.bind(homotopy=foreign, parameters=(("q", foreign),))
            context.emit("iteration", level=2, residual_norm=foreign)
            original_value = foreign._mpf_
        self.assertEqual(mp.mp.prec, original_bits)
        event = events[0]
        for value in (event.data["homotopy"], event.data["parameters"][0][1],
                      event.data["residual_norm"]):
            self.assertIs(value.context, owner.context)
            self.assertEqual(value._mpf_, original_value)
            self.assertLess(abs((value-1)-owner.scalar("1e-80")), owner.scalar("1e-118"))
        # Direct event construction has no numerical owner and does not
        # silently reinterpret a caller's scalar context.
        direct = ProgressEvent("iteration", 2, "", "", 0, {"value": foreign})
        self.assertIs(direct.data["value"], foreign)

    def test_output_resolves_current_stdout_and_flushes_without_replacing_it(self):
        original = __import__("sys").stdout
        reporter = ProgressReporter(1, solver="collocation", method="root_no_jac")
        output = FlushedStream()
        with redirect_stdout(output):
            reporter.emit("solve_start", norm="RMS")
        self.assertIs(__import__("sys").stdout, original)
        self.assertIn("[collocation/root_no_jac]", output.getvalue())
        self.assertGreater(output.flushes, 0)

    def test_invalid_options_are_rejected_without_bool_verbosity_aliases(self):
        for value in (-1, 4, True, np.bool_(False), 1.5, "2", None):
            with self.subTest(verbosity=value), self.assertRaises(ValueError):
                ProgressReporter(value)
        for value in (-1, float("nan"), float("inf"), True, "1", None):
            with self.subTest(interval=value), self.assertRaises(ValueError):
                ProgressReporter(progress_interval=value)
        with self.assertRaises(TypeError):
            ProgressReporter(progress_callback=object())
        with self.assertRaises(TypeError):
            ProgressReporter(stream=object())
        with self.assertRaises(TypeError):
            ProgressReporter(clock=None)
        reporter = ProgressReporter()
        for level in (0, 4, True):
            with self.subTest(level=level), self.assertRaises(ValueError):
                reporter.emit("bad", level=level)
        for kind in ("", None, 1):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                reporter.emit(kind)
        with self.assertRaises(ValueError):
            with reporter.phase(""):
                pass


if __name__ == "__main__":
    unittest.main()
