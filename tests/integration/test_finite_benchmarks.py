"""Finite manufactured cases run through the exported benchmark interfaces."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp

from benchmarks.run_manufactured import _case, main, parser
from benchmarks.run_precision import main as precision_main
from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.quadrature import QuadratureFactory


class FiniteBenchmarkTests(unittest.TestCase):
    def run_study(self, arguments, *, precision=False):
        with tempfile.TemporaryDirectory(prefix="overscreening-finite-benchmark-", dir="/tmp") as directory:
            output = Path(directory)
            runner = precision_main if precision else main
            with redirect_stdout(io.StringIO()):
                status = runner(arguments+[
                    "--diagnostic-order", "12", "--output-dir", directory,
                ])
            report = json.loads((output/"results.json").read_text())
            with (output/"results.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(report["records"]))
        for csv_row, record in zip(rows, report["records"]):
            self.assertEqual(json.loads(csv_row["case_parameters"]), record["case_parameters"])
        return status, report

    def assert_finite_record(self, record, radius, *, unknowns):
        a = MpmathArithmetic(100)
        # Compare decimal output after rounding back into its recorded owner:
        # a native 50-digit binary endpoint need not equal the raw decimal at
        # 100 digits, but serialization must recover the same native value.
        a.context.prec = record["precision_bits"]
        end = a.scalar(radius)
        self.assertTrue(record["diagnostic_success"])
        self.assertEqual(record["unknown_count"], unknowns)
        self.assertIsNone(record["L"])
        self.assertIsNone(record.get("far_u_trace"))
        self.assertEqual([a.scalar(value) for value in record["physical"]["domain"]], [0, end])
        self.assertEqual([a.scalar(value) for value in record["diagnostic_domain"]], [0, end])
        points = [a.scalar(value) for value in record["diagnostic_sample_points"]]
        self.assertTrue(points)
        self.assertTrue(all(0 <= point <= end for point in points))
        self.assertEqual(points[0], 0)
        self.assertEqual(points[-1], end)
        for field in ("u", "w", "flux"):
            self.assertIn(field+"_l2", record["errors"])
            self.assertNotIn(field+"_l2_finite_interval", record["errors"])
            self.assertTrue(a.isfinite(a.scalar(record["errors"][field+"_l2"])))

    def test_linear_and_nonlinear_constant_and_smooth_cases_are_exact_space_runs(self):
        for selector in ("finite:constant", "finite:smooth",
                         "finite-nonlinear:constant", "finite-nonlinear:smooth"):
            with self.subTest(case=selector):
                status, report = self.run_study([
                    "--case", selector, "--R", "4", "--delta", "2", "--beta", "0.2",
                    "--N", "4", "--no-globalization",
                ])
                self.assertEqual(status, 0)
                row = report["records"][0]
                self.assertTrue(row["converged"])
                self.assert_finite_record(row, "4", unknowns=8)
                self.assertLess(float(row["errors"]["u_l2"]), 1e-8)
                self.assertLess(float(row["errors"]["w_l2"]), 1e-8)
                settings = row["case_parameters"]
                self.assertEqual(settings["R"], "4")
                self.assertEqual(settings["delta"], "2")
                self.assertEqual(settings["beta"], "0.2" if "nonlinear" in selector else "0")
                self.assertEqual(settings["interfaces"], 0)
                self.assertIsNone(settings["material_style"])
                self.assertEqual(row["initialization_policy"], "linear_endpoint_lift_zero_w")

    def test_layered_selectors_preserve_material_and_interface_splits(self):
        plans = []

        def diagnostic_factory(*args, **kwargs):
            factory = QuadratureFactory(*args, **kwargs)
            build = factory.plan

            def plan(*args, **kwargs):
                result = build(*args, **kwargs)
                plans.append(result)
                return result

            factory.plan = plan
            return factory

        for selector, style in (("finite:layered", "constant"),
                                ("finite-nonlinear:layered", "graded")):
            plans.clear()
            with self.subTest(case=selector, style=style), patch(
                    "benchmarks.run_manufactured.QuadratureFactory", side_effect=diagnostic_factory):
                status, report = self.run_study([
                    "--case", selector, "--R", "4", "--delta", "2",
                    "--interfaces", "3", "--material-style", style,
                    "--N", "6", "--quadrature-atol", "1e-7", "--quadrature-rtol", "1e-7",
                ])
                self.assertEqual(status, 0)
                row = report["records"][0]
                self.assertTrue(row["converged"])
                self.assert_finite_record(row, "4", unknowns=12)
                self.assertEqual(len(row["physical"]["epsilon_interfaces"]), 3)
                self.assertEqual([float(value) for value in row["physical"]["epsilon_interfaces"]],
                                 [1, 2, 3])
                settings = row["case_parameters"]
                self.assertEqual(settings["coefficient"], "layered")
                self.assertEqual(settings["interfaces"], 3)
                self.assertEqual(settings["material_style"], style)
                self.assertIsInstance(settings["material"], dict)
                self.assertTrue(plans)
                plan = plans[-1]
                self.assertEqual(plan.domain, (0, 4))
                boundaries = {end for panel in plan.panels for end in panel}
                self.assertTrue(all(plan.mapping.to_reference(point) in boundaries
                                    for point in (1, 2, 3)))

    def test_all_interface_counts_reach_the_registered_case_factory(self):
        for count in (1, 2, 3, 5, 7, 10):
            for style in ("constant", "graded"):
                options = parser().parse_args([
                    "--case", "finite:layered", "--R", "4", "--interfaces", str(count),
                    "--material-style", style,
                ])
                case = _case("finite:layered", options)
                bound = case.make_problem().bind(NumpyArithmetic())
                with self.subTest(interfaces=count, style=style):
                    self.assertEqual(case.supported_solvers, ("galerkin",))
                    self.assertEqual(bound.domain, (0, 4))
                    self.assertEqual(len(bound.epsilon.interfaces), count)
                    self.assertTrue(all(0 < point < 4 for point in bound.epsilon.interfaces))

    def test_short_and_long_segments_use_the_actual_domain_for_diagnostics(self):
        for radius in ("0.5", "12"):
            with self.subTest(R=radius):
                status, report = self.run_study([
                    "--case", "finite:constant", "--R", radius, "--N", "4",
                ])
                self.assertEqual(status, 0)
                self.assert_finite_record(report["records"][0], radius, unknowns=8)
                self.assertLess(float(report["records"][0]["errors"]["u_l2"]), 1e-8)

    def test_constant_case_supports_finite_collocation(self):
        for arithmetic in ("numpy", "mpmath"):
            with self.subTest(arithmetic=arithmetic):
                status, report = self.run_study([
                    "--case", "finite:constant", "--R", "0.5", "--N", "4",
                    "--solver", "collocation", "--arithmetic", arithmetic, "--precision", "30",
                    "--tolerance", "1e-20" if arithmetic == "mpmath" else "1e-10",
                ])
                self.assertEqual(status, 0)
                row = report["records"][0]
                self.assert_finite_record(row, "0.5", unknowns=8)
                self.assertLess(float(row["errors"]["u_l2"]), 1e-8)
                self.assertEqual(row["initialization_policy"], "wall_exponential")

    def test_precision_runner_retains_finite_geometry_and_native_digits(self):
        original_bits = mp.mp.prec
        radius = "4.000000000000000000000000000000000000000001"
        status, report = self.run_study([
            "--case", "finite:constant", "--R", radius, "--N", "4",
            "--quadrature-order", "8", "--precisions", "30", "50",
        ], precision=True)
        self.assertEqual(status, 0)
        self.assertEqual(mp.mp.prec, original_bits)
        low, high = report["records"]
        self.assertEqual([low["precision_bits"], high["precision_bits"]], [103, 169])
        self.assertEqual(high["case_parameters"]["R"], radius)
        self.assert_finite_record(high, radius, unknowns=8)
        self.assertGreater(MpmathArithmetic(100).scalar(high["physical"]["domain"][1]), 4)
        self.assertGreater(len(high["coefficients"][0]), 40)
        self.assertLess(float(high["errors"]["u_l2"]), 1e-40)
        self.assertLess(float(high["errors"]["u_l2"]), float(low["errors"]["u_l2"])*1e-10)

    def test_finite_nonconvergence_retains_case_data_and_diagnostic_results(self):
        status, report = self.run_study([
            "--case", "finite:constant", "--R", "4", "--N", "4",
            "--max-iterations", "0", "--no-globalization",
        ])
        self.assertEqual(status, 1)
        row = report["records"][0]
        self.assertFalse(row["converged"])
        self.assertEqual(row["case_parameters"]["R"], "4")
        self.assertEqual(row["newton"]["reason"], "max_iterations")
        self.assert_finite_record(row, "4", unknowns=8)

    def test_invalid_finite_parameters_and_solver_combinations_are_rejected(self):
        output = tempfile.TemporaryDirectory(prefix="overscreening-invalid-finite-", dir="/tmp")
        self.addCleanup(output.cleanup)
        invalid = [
            ["--R", "0"], ["--R", "-1"], ["--R", "nan"], ["--R", "inf"],
            ["--delta", "0"], ["--delta", "-1"], ["--delta", "nan"],
            ["--beta", "-0.1"], ["--beta", "nan"], ["--beta", "inf"],
            ["--interfaces", "4"], ["--material-style", "invalid"],
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--case", "finite-nonlinear:layered", "--output-dir", output.name,
                          *arguments])
                self.assertEqual(error.exception.code, 2)
        for selector, solver in (("finite:smooth", "collocation"),
                                 ("finite:layered", "collocation"),
                                 ("finite:constant", "legacy_collocation"),
                                 ("finite:unknown", "galerkin")):
            with self.subTest(case=selector, solver=solver), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--case", selector, "--solver", solver, "--output-dir", output.name])
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
