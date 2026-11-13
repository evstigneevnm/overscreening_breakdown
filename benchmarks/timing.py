"""Benchmark phase accounting and repetition summaries.

Solver progress phases are sequential and may nest. Inclusive phase times are
useful for inspection; only exclusive times are additive. These are ordinary
wall-clock measurements, independent of the solver's arithmetic precision.
"""

from collections import Counter
from decimal import Decimal
import math
from numbers import Integral, Real
from statistics import median
from time import perf_counter


def _snapshot(value):
    """Retain scalar event context without retaining a numerical owner."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if hasattr(value, "_mpf_"):
        return value.context.nstr(value, n=value.context.dps + 8, strip_zeros=False)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Real):
        result = float(value)
        return result if math.isfinite(result) else str(value)
    if isinstance(value, tuple):
        return [_snapshot(item) for item in value]
    return str(value)


def _seconds(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


class PhaseTimings:
    """Observe progress without evaluating callbacks or changing arithmetic.

    Completed durations come from ``phase_end``. A missing end event (for
    example, following a precision ownership failure) is closed at ``finish``
    using the observer's clock and explicitly marked incomplete. Such a span
    is a partial observation, not a successfully completed measurement.
    """

    def __init__(self, *, clock=perf_counter):
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._stack = []
        self._spans = []
        self._issues = []
        self._event_counts = Counter()
        self._finished = None

    def __call__(self, event):
        if self._finished is not None:
            raise RuntimeError("a finished timing collector cannot observe another solve")
        self._event_counts[event.kind] += 1
        if event.kind not in ("phase_start", "phase_end"):
            return
        phase = event.data.get("phase")
        if not isinstance(phase, str) or not phase:
            self._issues.append("{} without a phase name".format(event.kind))
            return
        if event.kind == "phase_start":
            span = {
                "id": len(self._spans),
                "parent_id": self._stack[-1]["id"] if self._stack else None,
                "phase": phase,
                "solver": event.solver,
                "method": event.method,
                "context": {key: _snapshot(value) for key, value in event.data.items()
                            if key != "phase"},
                "start_elapsed_seconds": float(event.elapsed),
                "_observed_start": self._clock(),
            }
            self._spans.append(span)
            self._stack.append(span)
            return
        if not self._stack or (phase, event.solver, event.method) != (
                self._stack[-1]["phase"], self._stack[-1]["solver"],
                self._stack[-1]["method"]):
            self._issues.append("unmatched phase_end: {}".format(phase))
            return
        span = self._stack.pop()
        duration = _seconds(event.data.get("duration"))
        if duration is None:
            duration = max(0.0, self._clock() - span["_observed_start"])
            self._issues.append("invalid or missing duration: {}".format(phase))
            source, complete = "observer_clock", False
        else:
            source, complete = "phase_end", True
        span.update(
            inclusive_seconds=duration, complete=complete,
            success=bool(event.data.get("success", False)), duration_source=source,
            end_elapsed_seconds=float(event.elapsed),
        )
        for key in ("error_type", "reason"):
            if key in event.data:
                span[key] = _snapshot(event.data[key])

    def finish(self, solve_seconds):
        """Return JSON-safe measurements; repeated calls return the same result."""
        if self._finished is not None:
            return self._finished
        total = _seconds(solve_seconds)
        if total is None:
            raise ValueError("solve_seconds must be finite and nonnegative")
        if self._stack:
            now = self._clock()
            for span in self._stack:
                span.update(
                    inclusive_seconds=max(0.0, now - span["_observed_start"]),
                    complete=False, success=None, duration_source="observer_clock_partial",
                    end_elapsed_seconds=None,
                )
            self._stack.clear()
        children = Counter()
        for span in self._spans:
            if span["parent_id"] is not None:
                children[span["parent_id"]] += span["inclusive_seconds"]
        tolerance = 1e-9 * max(1.0, total)
        by_phase = {}
        for span in self._spans:
            exclusive = span["inclusive_seconds"] - children[span["id"]]
            if exclusive < -tolerance:
                self._issues.append("children exceed parent duration: span {}".format(span["id"]))
                exclusive = None
            elif exclusive < 0:
                exclusive = 0.0
            span["exclusive_seconds"] = exclusive
            del span["_observed_start"]
            aggregate = by_phase.setdefault(span["phase"], {
                "count": 0, "completed_count": 0, "incomplete_count": 0,
                "failed_count": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0,
            })
            aggregate["count"] += 1
            aggregate["completed_count" if span["complete"] else "incomplete_count"] += 1
            aggregate["failed_count"] += span["success"] is False
            aggregate["inclusive_seconds"] += span["inclusive_seconds"]
            if exclusive is None:
                aggregate["exclusive_seconds"] = None
            elif aggregate["exclusive_seconds"] is not None:
                aggregate["exclusive_seconds"] += exclusive
        covered = sum(span["inclusive_seconds"] for span in self._spans
                      if span["parent_id"] is None)
        unattributed = total - covered
        if unattributed < -tolerance:
            self._issues.append("root phase durations exceed solve_seconds")
            unattributed = None
        elif unattributed < 0:
            unattributed = 0.0
        incomplete = sum(not span["complete"] for span in self._spans)
        self._finished = {
            "spans": self._spans, "by_phase": by_phase,
            "unattributed_seconds": unattributed,
            "incomplete_spans": incomplete,
            "complete": not incomplete and not self._issues,
            "accounting_issues": self._issues,
            "event_counts": dict(self._event_counts),
            "duration_note": (
                "Completed inclusive durations use reporter phase_end clocks; exclusive durations "
                "subtract direct children. Unattributed time is outside root phases. Durations "
                "include reporting overhead; incomplete spans use partial observer-clock estimates."
            ),
        }
        return self._finished


def _quartile(values, fraction):
    """Linear interpolation of sorted observations, including singleton groups."""
    position = (len(values) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (position - left)


def _statistics(values):
    values = sorted(value for value in (_seconds(item) for item in values)
                    if value is not None)
    if not values:
        return None
    q1, q3 = _quartile(values, 0.25), _quartile(values, 0.75)
    return {"count": len(values), "min": values[0], "max": values[-1],
            "median": median(values), "q1": q1, "q3": q3, "iqr": q3-q1}


def _successful(row):
    return (bool(row.get("converged")) and bool(row.get("diagnostic_success"))
            and not row.get("error_type"))


def _phase_summaries(rows):
    observations = {}
    for row in rows:
        report = row.get("phase_timings", {})
        for name, aggregate in report.get("by_phase", {}).items():
            observations.setdefault(name, []).append((report, aggregate))
    result = {}
    for name, attempts in sorted(observations.items()):
        complete = [(report, phase) for report, phase in attempts
                    if not report.get("accounting_issues")
                    and not phase.get("incomplete_count", 0)
                    and phase.get("exclusive_seconds") is not None]
        phase_summary = {
            "attempts": len(attempts), "complete_attempts": len(complete),
            "incomplete_attempts": sum(bool(phase.get("incomplete_count", 0))
                                       for _, phase in attempts),
            "incomplete_spans": sum(phase.get("incomplete_count", 0)
                                    for _, phase in attempts),
            "invalid_attempts": sum(bool(report.get("accounting_issues"))
                                    for report, _ in attempts),
            "counts": {}, "durations": {},
        }
        for field in ("count", "completed_count", "incomplete_count", "failed_count"):
            stats = _statistics(phase.get(field) for _, phase in attempts)
            if stats is not None:
                phase_summary["counts"][field] = stats
        for field in ("inclusive_seconds", "exclusive_seconds"):
            stats = _statistics(phase.get(field) for _, phase in complete)
            if stats is not None:
                phase_summary["durations"][field] = stats
        result[name] = phase_summary
    return result


def summarize_records(records):
    """Group comparable attempts while retaining failures and excluding warmups.

    Summaries label all, successful, and failed measured attempts separately.
    Raw records remain the authority for scientific results and individual work
    counts. No statistics from a warmup enter a measured distribution.
    """
    groups = {}
    for row in records:
        groups.setdefault(row["config_id"], []).append(row)
    result = []
    for identifier, rows in groups.items():
        warmups = [row for row in rows if row.get("warmup", False)
                   or row.get("repetition_kind") == "warmup"]
        measured = [row for row in rows if not (row.get("warmup", False)
                    or row.get("repetition_kind") == "warmup")]
        successful = [row for row in measured if _successful(row)]
        failed = [row for row in measured if not _successful(row)]
        summary = {
            "configuration_id": identifier, "attempted": len(measured),
            "succeeded": len(successful), "failed": len(failed),
            "warmup_attempted": len(warmups),
            "warmup_failed": sum(not _successful(row) for row in warmups),
            "timings": {}, "work": {}, "phases": {},
        }
        timing_names = ("solve_seconds", "diagnostic_seconds", "export_seconds",
                        "process_wall_seconds", "attempt_seconds")
        work_names = sorted({name for row in measured for name in row.get("work", {})})
        for label, subset in (("all_attempts", measured),
                              ("successful_attempts", successful),
                              ("failed_attempts", failed)):
            summary["timings"][label] = {
                name: stats for name in timing_names
                if (stats := _statistics(row.get(name) for row in subset)) is not None
            }
            summary["work"][label] = {
                name: stats for name in work_names
                if (stats := _statistics(row.get("work", {}).get(name) for row in subset)) is not None
            }
            summary["phases"][label] = _phase_summaries(subset)
        result.append(summary)
    return result
