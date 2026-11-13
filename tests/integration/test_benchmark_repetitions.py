"""Independent benchmark repetitions and exported timing-accounting contracts."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp

from benchmarks import run_manufactured
from benchmarks.run_precision import main as precision_main


class BenchmarkRepetitionTests(unittest.TestCase):
    def run_study(self, arguments, *, precision=False):
        with tempfile.TemporaryDirectory(prefix="overscreening-repetitions-", dir="/tmp") as directory:
            runner = precision_main if precision else run_manufactured.main
            with redirect_stdout(io.StringIO()):
                status = runner(arguments + ["--diagnostic-order", "8", "--output-dir", directory])
            report = json.loads((Path(directory)/"results.json").read_text())
            payloads = [json.loads((Path(directory)/row["attempt_file"]).read_text())
                        for row in report["records"]]
        return status, report, payloads

    def assert_accounting(self, record):
        for name in ("solve_seconds", "diagnostic_seconds", "attempt_seconds", "export_seconds"):
            self.assertIsInstance(record[name], float)
            self.assertGreaterEqual(record[name], 0)
        timing = record["phase_timings"]
        self.assertTrue(timing["complete"], timing["accounting_issues"])
        self.assertEqual(timing["incomplete_spans"], 0)
        self.assertEqual(timing["accounting_issues"], [])
        self.assertIn("exclusive", timing["duration_note"])
        self.assertIn("reporting overhead", timing["duration_note"])
        total = sum(phase["exclusive_seconds"] for phase in timing["by_phase"].values())
        total += timing["unattributed_seconds"]
        self.assertAlmostEqual(total, record["solve_seconds"], places=8)
        self.assertGreater(timing["by_phase"]["residual"]["count"], 0)
        self.assertEqual(record["scipy_internal_linear_solve_seconds"], None)
        self.assertEqual(record["linear_solve_timing"], "native_factorization_and_solve")

    def test_same_and_fresh_modes_preserve_warmups_and_measure_both_domains_and_arithmetics(self):
        original_bits = mp.mp.prec
        for process_mode in ("same", "fresh"):
            for arithmetic in ("numpy", "mpmath"):
                for selector in ("rational:constant", "finite:constant"):
                    with self.subTest(mode=process_mode, arithmetic=arithmetic, case=selector):
                        status, report, payloads = self.run_study([
                            "--case", selector, "--N", "4", "--arithmetic", arithmetic,
                            "--precision", "30", "--process-mode", process_mode,
                            "--warmup-repetitions", "1", "--repetitions", "2",
                            "--no-globalization", "--max-refinements", "0",
                        ])
                        self.assertEqual(status, 0)
                        records = report["records"]
                        self.assertEqual(len(records), 3)
                        self.assertEqual([row["warmup"] for row in records], [True, False, False])
                        self.assertEqual([row["repetition"] for row in records], [1, 1, 2])
                        self.assertEqual(len({row["config_id"] for row in records}), 1)
                        self.assertEqual([row["attempt_id"] for row in records], [1, 2, 3])
                        self.assertEqual(len({row["attempt_file"] for row in records}), 3)
                        summary, = report["summaries"]
                        self.assertEqual((summary["attempted"], summary["succeeded"], summary["failed"]),
                                         (2, 2, 0))
                        self.assertEqual(summary["warmup_attempted"], 1)
                        self.assertEqual(summary["warmup_failed"], 0)
                        self.assertEqual(summary["timings"]["all_attempts"]["solve_seconds"]["count"], 2)
                        self.assertEqual(summary["work"]["all_attempts"]["newton_iterations"]["count"], 2)
                        if process_mode == "same":
                            self.assertEqual({row["process_id"] for row in records}, {os.getpid()})
                            self.assertTrue(all(row["process_wall_seconds"] is None for row in records))
                        else:
                            self.assertEqual(len({row["process_id"] for row in records}), 3)
                            self.assertNotIn(os.getpid(), {row["process_id"] for row in records})
                            for row in records:
                                self.assertIsInstance(row["process_wall_seconds"], float)
                                self.assertGreaterEqual(row["process_wall_seconds"], row["solve_seconds"])
                        for row, payload in zip(records, payloads):
                            self.assertTrue(row["converged"] and row["diagnostic_success"])
                            self.assertEqual(row["process_mode"], process_mode)
                            self.assertEqual(row["precision_bits"], 103 if arithmetic == "mpmath" else 53)
                            self.assertEqual(payload["coefficients"], row["coefficients"])
                            self.assertNotIn("export_seconds", payload)
                            self.assertIsNone(payload.get("process_wall_seconds"))
                            self.assert_accounting(row)
                        policies = {row["initialization_policy"] for row in records}
                        self.assertEqual(policies, {"linear_endpoint_lift_zero_w"}
                                         if selector.startswith("finite:") else {None})
        self.assertEqual(mp.mp.prec, original_bits)

    def test_default_repetition_controls_still_execute_one_attempt(self):
        status, report, payloads = self.run_study(["--N", "4"])
        self.assertEqual(status, 0)
        row, = report["records"]
        self.assertFalse(row["warmup"])
        self.assertEqual(row["repetition"], 1)
        self.assertEqual(row["process_mode"], "same")
        self.assertEqual(len(payloads), 1)
        self.assertEqual(report["metadata"]["options"]["warmup_repetitions"], 0)
        self.assertTrue(report["metadata"]["measurement"]["instrumentation_included"])

    def test_precision_runner_uses_the_same_repetition_and_summary_pipeline(self):
        original_bits = mp.mp.prec
        status, report, _ = self.run_study([
            "--case", "finite:constant", "--N", "4", "--quadrature-order", "8",
            "--precisions", "30", "40", "--warmup-repetitions", "1", "--repetitions", "2",
        ], precision=True)
        self.assertEqual(status, 0)
        self.assertEqual(report["metadata"]["study"], "precision")
        self.assertEqual(len(report["records"]), 6)
        self.assertEqual(len(report["summaries"]), 2)
        self.assertEqual({row["precision_bits"] for row in report["records"]}, {103, 136})
        for summary in report["summaries"]:
            self.assertEqual(summary["warmup_attempted"], 1)
            self.assertEqual(summary["succeeded"], 2)
        self.assertEqual(mp.mp.prec, original_bits)

    def test_nonconvergence_preserves_every_attempt_including_failed_warmup(self):
        status, report, _ = self.run_study([
            "--N", "4", "--no-globalization", "--max-iterations", "0",
            "--warmup-repetitions", "1", "--repetitions", "2",
        ])
        self.assertEqual(status, 1)
        self.assertEqual(len(report["records"]), 3)
        self.assertTrue(all(not row["converged"] for row in report["records"]))
        summary, = report["summaries"]
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["succeeded"], 0)
        self.assertEqual(summary["warmup_failed"], 1)
        self.assertEqual(summary["timings"]["successful_attempts"], {})
        self.assertEqual(summary["timings"]["failed_attempts"]["solve_seconds"]["count"], 2)

    def test_same_process_attempts_reconstruct_descriptions_owners_and_unchanged_controls(self):
        original_case, original_solve = run_manufactured._case, run_manufactured.solve
        descriptions, calls, owners = [], [], []

        def case_factory(*args, **kwargs):
            case = original_case(*args, **kwargs)
            descriptions.append(case)
            return case

        def solve(problem, *args, **kwargs):
            calls.append((problem, kwargs))
            solution = original_solve(problem, *args, **kwargs)
            owners.append(solution.arithmetic)
            return solution

        with patch("benchmarks.run_manufactured._case", side_effect=case_factory):
            with patch("benchmarks.run_manufactured.solve", side_effect=solve):
                status, report, _ = self.run_study([
                    "--case", "finite-nonlinear:constant", "--N", "4", "--arithmetic", "mpmath",
                    "--precision", "30", "--tolerance", "1e-20", "--no-globalization",
                    "--warmup-repetitions", "1", "--repetitions", "2",
                ])
        self.assertEqual(status, 0)
        self.assertEqual(len(descriptions), 4)  # validation, then one fresh description per attempt
        self.assertEqual(len({id(case) for case in descriptions}), 4)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({id(problem) for problem, _ in calls}), 3)
        self.assertEqual(len({id(owner) for owner in owners}), 3)
        controls = ("solver", "L", "use_mpmath", "precision", "method", "tolerance",
                    "max_iterations", "use_globalization", "quadrature_order", "subdivisions",
                    "quadrature_atol", "quadrature_rtol", "max_refinements", "verbosity")
        baseline = {key: calls[0][1][key] for key in controls}
        for _, options in calls:
            self.assertEqual({key: options[key] for key in controls}, baseline)
            self.assertNotIn("restart", options)
            self.assertNotIn("initial_guess", options)
        self.assertEqual(len({id(options["progress_callback"]) for _, options in calls}), 3)
        self.assertTrue(all(row["initialization_policy"] == "linear_endpoint_lift_zero_w"
                            for row in report["records"]))

    def test_invalid_repetition_controls_are_rejected(self):
        for arguments in (["--repetitions", "0"], ["--repetitions", "-1"],
                          ["--warmup-repetitions", "-1"], ["--process-mode", "warm"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    run_manufactured.main(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_exception_work_counts_preserve_observed_events_without_claiming_complete_counts(self):
        result = run_manufactured._work_counts({
            "error_type": "ValueError", "phase_timings": {
                "event_counts": {"iteration_start": 3, "iteration_end": 2, "stage_start": 1},
                "by_phase": {"linear_solve": {"count": 3}},
            },
        })
        self.assertEqual(result["source"], "partial_events_only")
        self.assertEqual(result["observed_iteration_starts"], 3)
        self.assertEqual(result["observed_iteration_ends"], 2)
        self.assertEqual(result["observed_stage_starts"], 1)
        self.assertEqual(result["phase_calls"], {"linear_solve": 3})
        for field in ("newton_iterations", "newton_stages", "continuation_attempts",
                      "adjoint_steps", "quadrature_solves", "quadrature_refinements"):
            self.assertIsNone(result[field])

    def test_legacy_direct_assembly_and_linear_solve_have_correct_method_and_timing(self):
        status, report, _ = self.run_study([
            "--case", "linear:8", "--solver", "legacy_collocation", "--N", "8",
        ])
        self.assertEqual(status, 0)
        row, = report["records"]
        self.assertEqual(row["method"], "direct")
        self.assert_accounting(row)
        spans = row["phase_timings"]["spans"]
        self.assertEqual({span["method"] for span in spans}, {"direct"})
        phases = row["phase_timings"]["by_phase"]
        self.assertEqual(phases["collocation_assembly"]["count"], 1)
        self.assertEqual(phases["linear_solve"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
