"""Bounded benchmark CLI checks; heavy numerical studies remain opt-in."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mpmath as mp

from benchmarks.run_manufactured import main


class BenchmarkRunnerTests(unittest.TestCase):
    def run_study(self, arguments, *, precision_study=False):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            output = Path(directory)
            with redirect_stdout(io.StringIO()):
                status = main(arguments+["--diagnostic-order", "12", "--output-dir", directory],
                              precision_study=precision_study)
            report = json.loads((output/"results.json").read_text())
            with (output/"results.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(report["records"]))
            self.assertIn("versions", rows[0])
            self.assertIn("revision", rows[0])
            self.assertIn("versions", report["metadata"])
            self.assertIn("revision", report["metadata"])
            return status, report

    def test_small_resolution_study_records_actual_unknowns_and_physical_errors(self):
        status, report = self.run_study([
            "--case", "rational:constant", "--study", "resolution", "--N", "4", "5"])
        self.assertEqual(status, 0)
        self.assertEqual([row["unknown_count"] for row in report["records"]], [7, 9])
        for row in report["records"]:
            self.assertTrue(row["nonlinear_converged"])
            self.assertTrue(row["quadrature_converged"])
            self.assertIsNone(row["scipy"])
            self.assertLess(float(row["errors"]["u_l2"]), 1e-10)
            self.assertEqual(row["diagnostic_domain"], ["0", "inf"])

    def test_fixed_precision_study_retains_native_digits_and_global_context(self):
        old_bits = mp.mp.prec
        status, report = self.run_study([
            "--case", "rational:constant", "--N", "4", "--quadrature-order", "8",
            "--precisions", "30", "50"], precision_study=True)
        self.assertEqual(status, 0)
        self.assertEqual(mp.mp.prec, old_bits)
        coarse, fine = report["records"]
        self.assertEqual([coarse["precision_bits"], fine["precision_bits"]], [103, 169])
        self.assertEqual(coarse["final_quadrature_order"], fine["final_quadrature_order"])
        self.assertLess(mp.mpf(fine["errors"]["u_l2"]), mp.mpf("1e-45"))
        self.assertLess(mp.mpf(fine["errors"]["u_l2"]), mp.mpf(coarse["errors"]["u_l2"])/mp.mpf("1e10"))
        self.assertGreater(len(fine["coefficients"][0]), 40)
        for row in (coarse, fine):
            # Each refined-plan record now includes its complete Newton state.
            # Serialize that native mpmath column with the same decimal
            # precision as the independently retained final coefficients.
            final = row["quadrature_history"][-1]["newton_result"]
            self.assertEqual(final["state"], row["coefficients"])
            self.assertEqual(len(final["state"]), row["unknown_count"])
            self.assertTrue(all(isinstance(value, str) for value in final["state"]))
            self.assertEqual(final["adjoint_steps"], row["newton"]["adjoint_steps"])
            self.assertEqual(final["adjoint_evaluations"], row["newton"]["adjoint_evaluations"])
            self.assertEqual(final["iterations"], row["newton"]["iterations"])

    def test_legacy_only_operator_remains_an_explicit_bounded_study(self):
        status, report = self.run_study([
            "--case", "linear:8", "--solver", "legacy_collocation", "--N", "8"])
        self.assertEqual(status, 0)
        row = report["records"][0]
        self.assertEqual(row["method"], "direct")
        self.assertEqual(row["operator"], [None, None, None, None, 1])
        self.assertIn("u_l2_finite_interval", row["errors"])
        self.assertNotIn("u_l2", row["errors"])
        self.assertEqual(row["diagnostic_domain"], ["0", "8"])
        self.assertIsNone(row["scipy"])

    def test_numpy_scipy_studies_export_stop_status_and_verified_residuals(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                status, report = self.run_study([
                    "--case", "rational-nonlinear:constant", "--solver", "collocation",
                    "--N", "6", "--method", method, "--no-globalization"])
                self.assertEqual(status, 0)
                row = report["records"][0]
                diagnostics = row["scipy"]
                self.assertTrue(row["converged"] and row["nonlinear_converged"])
                self.assertIsNone(row["newton"])
                self.assertEqual(diagnostics["method"], method)
                self.assertTrue(diagnostics["converged"])
                self.assertEqual(diagnostics["reason"], "converged")
                self.assertIsInstance(diagnostics["scipy_success"], bool)
                self.assertIsInstance(diagnostics["status"], int)
                self.assertIsInstance(diagnostics["message"], str)
                self.assertGreater(diagnostics["nfev"], 0)
                self.assertEqual(float(diagnostics["homotopy"]), 0)
                self.assertLessEqual(float(diagnostics["residual_norm"]),
                                     float(diagnostics["target_norm"]))
                self.assertLessEqual(float(diagnostics["residual_norm"]),
                                     float(diagnostics["residual_max"]))
                self.assertLess(float(row["errors"]["u_l2_finite_interval"]), 1e-8)

    def test_legacy_scipy_study_records_verified_nonlinear_convergence(self):
        status, report = self.run_study([
            "--case", "nonlinear:1", "--solver", "legacy_collocation", "--N", "8",
            "--method", "root_no_jac", "--no-globalization"])
        self.assertEqual(status, 0)
        row = report["records"][0]
        self.assertTrue(row["nonlinear_converged"])
        self.assertIsNone(row["newton"])
        self.assertEqual(row["scipy"]["method"], "root_no_jac")
        self.assertTrue(row["scipy"]["converged"])
        self.assertLessEqual(float(row["scipy"]["residual_norm"]),
                             float(row["scipy"]["target_norm"]))
        self.assertGreater(row["scipy"]["nfev"], 0)

    def test_nonconvergence_is_recorded_and_returns_failure(self):
        status, report = self.run_study([
            "--case", "rational:constant", "--N", "4", "--max-iterations", "0",
            "--no-globalization"])
        self.assertEqual(status, 1)
        row = report["records"][0]
        self.assertFalse(row["converged"])
        self.assertFalse(row["nonlinear_converged"])
        self.assertEqual(row["newton"]["reason"], "max_iterations")

    def test_diagnostic_failure_retains_solver_convergence_and_returns_failure(self):
        status, report = self.run_study([
            "--case", "rational:constant", "--solver", "collocation", "--N", "6",
            "--diagnostic-endpoint", "0"])
        self.assertEqual(status, 1)
        row = report["records"][0]
        self.assertTrue(row["converged"])
        self.assertTrue(row["nonlinear_converged"])
        self.assertFalse(row["diagnostic_success"])
        self.assertEqual(row["diagnostic_error_type"], "ValueError")
        self.assertIn("endpoint", row["diagnostic_error"])

    def test_backend_mismatch_and_cartesian_sweep_are_rejected(self):
        wrong_backend = "python" if mp.libmp.BACKEND == "gmpy" else "gmpy"
        for arguments in (["--backend", wrong_backend], ["--N", "4", "8"],
                          ["--study", "quadrature", "--quadrature-order", "8", "12"],
                          ["--case", "linear:8", "--solver", "galerkin"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main(arguments)
                self.assertEqual(result.exception.code, 2)

    def test_precision_and_experiment_modules_have_bounded_cli_entry_points(self):
        root = Path(__file__).resolve().parents[2]
        for module in ("benchmarks.run_precision", "experiments.manufactured_solutions"):
            with self.subTest(module=module):
                process = subprocess.run([sys.executable, "-B", "-m", module, "--help"],
                                         cwd=root, capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("--output-dir", process.stdout)


if __name__ == "__main__":
    unittest.main()
