"""Timing must remain reportable after a solver's numerical owner becomes stale."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mpmath as mp

from benchmarks import run_manufactured as runner


def test_each_attempt_verifies_and_records_its_actual_backend_before_solving():
    wrong = "python" if mp.libmp.BACKEND == "gmpy" else "gmpy"
    options = runner.parser().parse_args([])
    with patch.object(runner, "solve") as solve:
        row = runner._execute_attempt(options, "rational:constant", 4, 50, None,
                                      expected_backend=wrong)
    solve.assert_not_called()
    assert not row["converged"]
    assert row["error_type"] == "ValueError"
    assert "backend" in row["error"]
    assert row["mpmath_backend"] == mp.libmp.BACKEND
    assert "solve_seconds" not in row


def test_precision_failure_is_fatal_to_each_solve_and_preserves_partial_timing():
    real_solve = runner.solve
    owners = []

    def interrupted_solve(problem, *args, **kwargs):
        owned = []

        class CaptureBinding:
            def bind(self, arithmetic):
                owned.append(arithmetic)
                owners.append(arithmetic)
                return problem.bind(arithmetic)

        collector = kwargs["progress_callback"]

        def observer(event):
            collector(event)
            if event.kind == "phase_start" and event.data.get("phase") == "jacobian":
                owned[0].context.prec += 1

        kwargs["progress_callback"] = observer
        return real_solve(CaptureBinding(), *args, **kwargs)

    previous = mp.mp.prec
    with TemporaryDirectory(dir="/tmp") as directory, \
            patch.object(runner, "solve", side_effect=interrupted_solve), \
            redirect_stdout(io.StringIO()):
        status = runner.main([
            "--case", "rational:constant", "--N", "4", "--arithmetic", "mpmath",
            "--precision", "50", "--tolerance", "1e-30", "--repetitions", "2",
            "--output-dir", directory,
        ])
        report = json.loads((Path(directory)/"results.json").read_text())
    assert status == 1
    assert mp.mp.prec == previous
    assert len(owners) == 2 and owners[0] is not owners[1]
    assert len(report["records"]) == 2
    for row in report["records"]:
        assert row["error_type"] == "RuntimeError"
        assert "precision changed" in row["error"]
        assert not row["converged"]
        timing = row["phase_timings"]
        assert not timing["complete"]
        assert timing["incomplete_spans"] >= 1
        partial = [span for span in timing["spans"] if not span["complete"]]
        assert any(span["phase"] == "jacobian" for span in partial)
        assert all(span["success"] is None for span in partial)
        assert row["work"]["newton_iterations"] is None
        assert row["work"]["observed_iteration_starts"] == 1
        assert row["work"]["source"] == "partial_events_only"
        assert isinstance(row["solve_seconds"], float)
    summary = report["summaries"][0]
    assert summary["failed"] == 2 and summary["succeeded"] == 0
    assert "newton_iterations" not in summary["work"]["failed_attempts"]
    jacobian = summary["phases"]["failed_attempts"]["jacobian"]
    assert jacobian["incomplete_attempts"] == 2
    assert jacobian["durations"] == {}
