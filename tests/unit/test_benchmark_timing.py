"""Deterministic benchmark accounting; no machine-speed assertions."""

import json
import unittest

import mpmath as mp

from benchmarks.timing import PhaseTimings, summarize_records
from overscreening_breakdown.diagnostics import ProgressEvent, ProgressReporter


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class PhaseTimingTests(unittest.TestCase):
    def test_nested_phases_partition_elapsed_time_without_double_counting(self):
        clock = Clock()
        collector = PhaseTimings(clock=clock)
        reporter = ProgressReporter(progress_callback=collector, clock=clock)
        clock.now = 1
        with reporter.phase("assembly"):
            clock.now = 2
            with reporter.phase("basis"):
                clock.now = 4
                with reporter.phase("tabulation"):
                    clock.now = 5
            clock.now = 6
            with reporter.phase("basis"):
                clock.now = 7
            clock.now = 8
        report = collector.finish(10)
        self.assertTrue(report["complete"])
        self.assertEqual([span["parent_id"] for span in report["spans"]], [None, 0, 1, 0])
        self.assertEqual([span["inclusive_seconds"] for span in report["spans"]], [7, 3, 1, 1])
        self.assertEqual([span["exclusive_seconds"] for span in report["spans"]], [3, 2, 1, 1])
        self.assertEqual(report["by_phase"]["basis"]["count"], 2)
        self.assertEqual(report["by_phase"]["basis"]["inclusive_seconds"], 4)
        self.assertEqual(report["by_phase"]["basis"]["exclusive_seconds"], 3)
        self.assertEqual(report["unattributed_seconds"], 3)
        self.assertEqual(sum(item["exclusive_seconds"] for item in report["by_phase"].values())
                         + report["unattributed_seconds"], 10)
        self.assertIs(collector.finish(10), report)
        json.dumps(report, allow_nan=False)

    def test_failed_phase_is_complete_but_preserves_failure(self):
        clock = Clock()
        collector = PhaseTimings(clock=clock)
        reporter = ProgressReporter(progress_callback=collector, clock=clock)
        with self.assertRaisesRegex(ValueError, "failed source"):
            with reporter.phase("residual", stage=2, iteration=3):
                clock.now = 4
                raise ValueError("failed source")
        report = collector.finish(5)
        span = report["spans"][0]
        self.assertTrue(report["complete"])
        self.assertFalse(span["success"])
        self.assertEqual(span["context"], {"stage": 2, "iteration": 3})
        self.assertEqual(span["error_type"], "ValueError")
        self.assertEqual(report["by_phase"]["residual"]["failed_count"], 1)

    def test_missing_end_events_are_partial_and_keep_native_metadata_as_text(self):
        clock = Clock()
        collector = PhaseTimings(clock=clock)
        context = mp.mp.clone()
        context.dps = 60
        q = context.mpf("0.12345678901234567890123456789012345678901234567890123456789")
        collector(ProgressEvent("phase_start", 1, "galerkin", "newton", 0,
                                {"phase": "assembly", "homotopy": q}))
        clock.now = 1
        collector(ProgressEvent("phase_start", 1, "galerkin", "newton", 1,
                                {"phase": "basis"}))
        clock.now = 3
        # The owner can already be stale at this point. finish uses only copied
        # scalar metadata and its independent clock, never a numerical callback.
        context.prec += 1
        report = collector.finish(4)
        self.assertFalse(report["complete"])
        self.assertEqual(report["incomplete_spans"], 2)
        self.assertEqual([span["inclusive_seconds"] for span in report["spans"]], [3, 2])
        self.assertEqual([span["exclusive_seconds"] for span in report["spans"]], [1, 2])
        self.assertIsNone(report["spans"][0]["success"])
        self.assertEqual(report["spans"][0]["duration_source"], "observer_clock_partial")
        self.assertIn("12345678901234567890123456789012345678901234567890123456789",
                      report["spans"][0]["context"]["homotopy"])
        json.dumps(report, allow_nan=False)

    def test_malformed_event_sequences_are_flagged_instead_of_inventing_totals(self):
        clock = Clock()
        collector = PhaseTimings(clock=clock)
        collector(ProgressEvent("phase_end", 1, "", "", 0,
                                {"phase": "missing", "duration": 1}))
        collector(ProgressEvent("phase_start", 1, "", "", 0, {"phase": "outer"}))
        collector(ProgressEvent("phase_start", 1, "", "", 0, {"phase": "inner"}))
        collector(ProgressEvent("phase_end", 1, "", "", 2,
                                {"phase": "inner", "duration": 2, "success": True}))
        collector(ProgressEvent("phase_end", 1, "", "", 2,
                                {"phase": "outer", "duration": 1, "success": True}))
        report = collector.finish(0.5)
        self.assertFalse(report["complete"])
        self.assertEqual(len(report["accounting_issues"]), 3)
        self.assertIsNone(report["spans"][0]["exclusive_seconds"])
        self.assertIsNone(report["unattributed_seconds"])

    def test_finished_collector_cannot_be_reused_for_another_solve(self):
        collector = PhaseTimings()
        collector.finish(0)
        with self.assertRaisesRegex(RuntimeError, "finished"):
            collector(ProgressEvent("solve_start", 1, "", "", 0))
        with self.assertRaises(ValueError):
            PhaseTimings().finish(float("nan"))


class RepetitionSummaryTests(unittest.TestCase):
    def record(self, seconds, *, config="a", warmup=False, converged=True,
               diagnostic=True, error=None):
        return {"config_id": config, "warmup": warmup,
                "converged": converged, "diagnostic_success": diagnostic,
                "error_type": error, "solve_seconds": seconds,
                "work": {"newton_iterations": seconds * 2}}

    def test_warmups_failures_and_comparable_configurations_remain_separate(self):
        rows = [self.record(100, warmup=True),
                self.record(200, warmup=True, converged=False),
                self.record(1), self.record(3),
                self.record(9, converged=False), self.record(11, diagnostic=False),
                self.record(13, error="RuntimeError"), self.record(2, config="b")]
        a, b = summarize_records(rows)
        self.assertEqual((a["attempted"], a["succeeded"], a["failed"]), (5, 2, 3))
        self.assertEqual((a["warmup_attempted"], a["warmup_failed"]), (2, 1))
        self.assertEqual(a["timings"]["all_attempts"]["solve_seconds"]["median"], 9)
        success = a["timings"]["successful_attempts"]["solve_seconds"]
        self.assertEqual(success, {"count": 2, "min": 1, "max": 3, "median": 2,
                                   "q1": 1.5, "q3": 2.5, "iqr": 1})
        self.assertEqual(a["work"]["failed_attempts"]["newton_iterations"]["max"], 26)
        self.assertEqual(b["attempted"], 1)
        self.assertEqual(b["timings"]["all_attempts"]["solve_seconds"]["iqr"], 0)
        json.dumps([a, b], allow_nan=False)

    def test_missing_and_nonfinite_measurements_do_not_become_zero(self):
        rows = [{"config_id": "a", "converged": False, "error_type": "ValueError"},
                {"config_id": "a", "converged": False, "solve_seconds": float("nan")},
                {"config_id": "a", "converged": False, "solve_seconds": None}]
        report = summarize_records(rows)[0]
        self.assertEqual(report["failed"], 3)
        self.assertEqual(report["timings"]["all_attempts"], {})

    def test_warmup_only_groups_have_no_measured_distribution(self):
        report = summarize_records([self.record(1, warmup=True)])[0]
        self.assertEqual(report["attempted"], 0)
        self.assertEqual(report["warmup_attempted"], 1)
        self.assertEqual(report["timings"]["all_attempts"], {})

    def test_phase_distributions_label_failures_and_exclude_partial_durations(self):
        def observed(seconds, *, partial=False, invalid=False, **options):
            row = self.record(seconds, **options)
            row["phase_timings"] = {
                "accounting_issues": ["invalid nesting"] if invalid else [],
                "by_phase": {"linear_solve": {
                    "count": 2, "completed_count": 1 if partial else 2,
                    "incomplete_count": int(partial), "failed_count": 0,
                    "inclusive_seconds": seconds, "exclusive_seconds": seconds,
                }},
            }
            return row

        rows = [observed(1), observed(3, converged=False),
                observed(50, partial=True, converged=False),
                observed(100, invalid=True, converged=False),
                observed(200, warmup=True)]
        report = summarize_records(rows)[0]
        all_attempts = report["phases"]["all_attempts"]["linear_solve"]
        self.assertEqual(all_attempts["attempts"], 4)
        self.assertEqual(all_attempts["complete_attempts"], 2)
        self.assertEqual(all_attempts["incomplete_attempts"], 1)
        self.assertEqual(all_attempts["incomplete_spans"], 1)
        self.assertEqual(all_attempts["invalid_attempts"], 1)
        self.assertEqual(all_attempts["durations"]["exclusive_seconds"]["median"], 2)
        self.assertEqual(all_attempts["counts"]["count"]["count"], 4)
        failures = report["phases"]["failed_attempts"]["linear_solve"]
        self.assertEqual(failures["durations"]["inclusive_seconds"]["median"], 3)
        successes = report["phases"]["successful_attempts"]["linear_solve"]
        self.assertEqual(successes["durations"]["inclusive_seconds"]["median"], 1)


if __name__ == "__main__":
    unittest.main()
