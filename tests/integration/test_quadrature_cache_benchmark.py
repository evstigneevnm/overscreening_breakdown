"""Check cache-study ownership and accounting without performance thresholds."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp

from benchmarks.run_quadrature_cache import _trial, main, parser
from overscreening_breakdown.arithmetic import MpmathArithmetic
from overscreening_breakdown.quadrature import QuadratureFactory


class QuadratureCacheBenchmarkTests(unittest.TestCase):
    def test_trials_create_distinct_owners_and_factories_but_reuse_their_plan(self):
        factories = []

        def construct(*args, **kwargs):
            result = QuadratureFactory(*args, **kwargs)
            factories.append(result)
            return result

        options = parser().parse_args(["--order", "4", "--lookups", "3"])
        with patch("benchmarks.run_quadrature_cache.QuadratureFactory", side_effect=construct):
            first, second = _trial(options), _trial(options)
        self.assertIsNot(factories[0], factories[1])
        self.assertIsNot(factories[0].arithmetic, factories[1].arithmetic)
        for record in (first, second):
            self.assertTrue(record["cache_identity_verified"])
            self.assertEqual(record["lookup_count"], 3)
            self.assertEqual(record["node_count"], 4)

    def test_deterministic_clock_separates_preparation_build_and_lookup_batch(self):
        options = parser().parse_args(["--order", "3", "--lookups", "2"])
        ticks = iter([0, 1, 2, 4, 5, 8, 10, 18])
        result = _trial(options, clock=lambda: next(ticks))
        self.assertEqual(result["owner_seconds"], 1)
        self.assertEqual(result["factory_seconds"], 2)
        self.assertEqual(result["plan_build_seconds"], 3)
        self.assertEqual(result["cached_lookup_seconds"], 8)
        self.assertEqual(result["cached_lookup_seconds_per_call"], 4)

    def test_numpy_and_mpmath_cli_keep_geometry_precision_and_native_clock_values(self):
        original_bits = mp.mp.prec
        for arithmetic in ("numpy", "mpmath"):
            with self.subTest(arithmetic=arithmetic), tempfile.TemporaryDirectory(dir="/tmp") as directory:
                with redirect_stdout(io.StringIO()):
                    status = main(["--arithmetic", arithmetic, "--precision", "30",
                                   "--R", "4", "--interfaces", "1", "2", "--order", "4",
                                   "--subdivisions", "2", "--lookups", "3", "--repetitions", "2",
                                   "--output-dir", directory])
                report = json.loads((Path(directory)/"quadrature-cache.json").read_text())
                self.assertEqual(status, 0)
                self.assertEqual(report["configuration"]["domain"], ["0", "4"])
                self.assertIsNone(report["configuration"]["L"])
                self.assertEqual(report["summary"]["successful"], 2)
                self.assertEqual(report["summary"]["failed"], 0)
                for record in report["records"]:
                    self.assertEqual(record["node_count"], 24)
                    self.assertEqual(record["precision_bits"], 103 if arithmetic == "mpmath" else 53)
                    self.assertIsInstance(record["plan_build_seconds"], float)
                    self.assertTrue(record["cache_identity_verified"])
        self.assertEqual(mp.mp.prec, original_bits)

    def test_failure_is_retained_and_excluded_from_successful_timing_statistics(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            with patch("benchmarks.run_quadrature_cache._trial", side_effect=RuntimeError("cache failure")):
                with redirect_stdout(io.StringIO()):
                    status = main(["--repetitions", "2", "--output-dir", directory])
            report = json.loads((Path(directory)/"quadrature-cache.json").read_text())
        self.assertEqual(status, 1)
        self.assertEqual(report["summary"]["failed"], 2)
        self.assertIsNone(report["summary"]["timings"]["plan_build_seconds"])
        self.assertEqual([row["repetition"] for row in report["records"]], [0, 1])
        self.assertEqual(report["records"][0]["error_type"], "RuntimeError")

    def test_backend_and_invalid_controls_are_rejected(self):
        wrong = "python" if mp.libmp.BACKEND == "gmpy" else "gmpy"
        for arguments in (["--backend", wrong], ["--lookups", "0"], ["--repetitions", "0"],
                          ["--L", "nan"], ["--R", "0"], ["--R", "3", "--interfaces", "3"],
                          ["--interfaces", "2", "1"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_precision_change_rebuilds_cached_plan_and_invalidates_old_snapshot(self):
        arithmetic = MpmathArithmetic(30)
        factory = QuadratureFactory(arithmetic, interfaces=("1",), domain=("0", "4"))
        original = factory.plan(4)
        self.assertIs(factory.plan(4), original)
        arithmetic.context.prec += 1
        with self.assertRaisesRegex(RuntimeError, "precision changed"):
            original.check_precision()
        rebuilt = factory.plan(4)
        self.assertIsNot(rebuilt, original)
        self.assertIs(factory.plan(4), rebuilt)
        rebuilt.check_precision()

    def test_module_cli_has_a_bounded_entry_point(self):
        process = subprocess.run(
            [sys.executable, "-B", "-m", "benchmarks.run_quadrature_cache", "--help"],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--lookups", process.stdout)


if __name__ == "__main__":
    unittest.main()
