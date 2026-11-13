"""A bounded, user-facing workflow for layered Galerkin experiments."""

from contextlib import redirect_stdout
import ast
import csv
import io
from itertools import product
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt

from overscreening_breakdown import BSKProblem, load_solution
from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.layered import LayeredMaterial, preset_material
from overscreening_breakdown.notebook_examples import discontinuous_galerkin as guide
from tests.examples.notebook_tools import code_tree, execute_allowed_cells


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT/"examples/notebooks/execution_example_galerkin_discontinuous.ipynb"
SMALL_N = 8
NUMPY_ATOL = "1e-10"
MPMATH_ATOL = "1e-35"
SMALL_NUMERICS = dict(quadrature_order=20, quadrature_atol="1e-8",
                      quadrature_rtol="1e-8")
CORE_CELL_IDS = (
    "setup", "experiment-inputs", "material-preview", "bsk-problem-definition",
    "linear-problem-definition", "nonlinear-problem-definition", "selected-experiment",
    "selected-plots", "material-gallery", "linear-verification", "linear-relative-errors",
    "nonlinear-verification", "nonlinear-relative-errors", "export",
)
EXTRA_RUN_IDS = ("b920119c-607f-47b3-93a4-9d2c28e91b46", "8606a13e-edc5-443c-8ecb-3ea5035e761f")
EXTRA_PLOT_IDS = ("4c070641-4c47-4367-a420-00909d5bd4b6", "5ae767a0-47a8-425c-94c1-a440006d0ed0")
CORE_CONTROLS = {
    "experiment-inputs": dict(layers=None, interface_count=3, material_style="constant", accuracy="standard",
                               model="BSK", delta="1", wall_potential=0.05, gamma="0.5", sigma="2", mu="0.1",
                               verbosity=0, plot_x_min="0", plot_x_max="8",
                               iteration_xlim=None, quadrature_xlim=None),
    "nonlinear-problem-definition": {"beta_nonlinear": "0.125"},
    "selected-experiment": {"N_selected": 8, "newton_atol_selected": "1e-10"},
    "linear-verification": {"N_linear": 8, "newton_atol_linear": "1e-10"},
    "nonlinear-verification": {"N_nonlinear": 8, "newton_atol_nonlinear": "1e-10"},
    "export": {"save_results": True},
}


def notebook_code(source):
    return code_tree(source)


class DiscontinuousGalerkinWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)

    def execute_notebook(self, cell_ids, controls):
        original = NOTEBOOK.read_bytes()
        notebook = json.loads(original)
        temporary = tempfile.TemporaryDirectory(prefix="overscreening-layered-notebook-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        namespace = {"__name__": "__main__", "resolution_check": None}
        with patch.dict(os.environ, {"OVERSCREENING_OUTPUT_DIR": temporary.name}), \
                patch.object(plt, "show"), patch("IPython.display.display"), \
                patch.object(guide, "run_defined_problem", wraps=guide.run_defined_problem) as runs, \
                redirect_stdout(io.StringIO()):
            report = execute_allowed_cells(notebook, cell_ids, namespace,
                                             controls_by_cell=controls, filename=str(NOTEBOOK))
            calls = tuple(runs.call_args_list)
        self.assertEqual(NOTEBOOK.read_bytes(), original)
        return namespace, calls, report, Path(temporary.name)

    def test_basis_resolution_is_required_and_invalid_values_fail_before_solving(self):
        material = preset_material(1)
        with patch.object(guide, "solve", side_effect=AssertionError("invalid N must not solve")) as solve:
            with self.assertRaisesRegex(TypeError, "N"):
                guide.run_experiment(material, newton_atol=NUMPY_ATOL)
            for N in (None, True, False, -1, 0, 1, 4.0, 4.5, "8"):
                with self.subTest(N=N), self.assertRaisesRegex(ValueError, "N"):
                    guide.run_experiment(material, N=N, newton_atol=NUMPY_ATOL)
            for numerical_N in (4, 8):
                with self.subTest(numerical_N=numerical_N), self.assertRaisesRegex(ValueError, "N"):
                    guide.run_experiment(material, N=4, newton_atol=NUMPY_ATOL,
                                         numerical={"N": numerical_N})
            solve.assert_not_called()

    def test_every_accuracy_profile_preserves_explicit_resolution_without_overrides(self):
        for accuracy in ("standard", "strict_integration", "high_precision"):
            with self.subTest(accuracy=accuracy):
                self.assertNotIn("N", guide.ACCURACY_PRESETS[accuracy])
                self.assertNotIn("tolerance", guide.ACCURACY_PRESETS[accuracy])
                self.assertNotIn("newton_atol", guide.ACCURACY_PRESETS[accuracy])
                atol = "3e-32" if accuracy == "high_precision" else "7e-11"
                result = guide.run_experiment(preset_material(1), N=4,
                                              newton_atol=atol, model="manufactured_linear",
                                              accuracy=accuracy)
                self.assertTrue(result.solution.converged)
                self.assertEqual(result.solution.N, 4)
                self.assertEqual(result.solution.unknown_count, 7)
                self.assertEqual(result.settings["numerical"]["N"], 4)
                self.assertEqual(result.settings["numerical"]["quadrature_order"], 12)
                self.assertEqual(result.solution.quadrature_history[0].order, 12)
                self.assertEqual(result.solution.arithmetic.use_mpmath, accuracy == "high_precision")
                arithmetic = result.solution.arithmetic
                for item in result.solution.quadrature_history:
                    for stage in item.newton_result.stages:
                        if stage.homotopy == 0:
                            self.assertEqual(stage.target_norm, arithmetic.scalar(atol))
                self.assertLessEqual(result.solution.newton_result.residual_norm,
                                     arithmetic.scalar(atol))

    def test_absolute_newton_tolerance_is_required_and_validated_before_solving(self):
        material = preset_material(1)
        invalid = (None, True, False, 0, -1, "0", "-1e-10", "nan", "inf", "-inf",
                   float("nan"), float("inf"), "not-a-number")
        with patch.object(guide, "solve", side_effect=AssertionError("invalid tolerance must not solve")) as solve:
            with self.assertRaisesRegex(TypeError, "newton_atol"):
                guide.run_experiment(material, N=4)
            for accuracy, atol in product(("standard", "high_precision"), invalid):
                with self.subTest(accuracy=accuracy, atol=atol), self.assertRaisesRegex(ValueError, "newton_atol"):
                    guide.run_experiment(material, N=4, newton_atol=atol, accuracy=accuracy)
            for name, value in product(("tolerance", "newton_atol"), (NUMPY_ATOL, "1e-5")):
                with self.subTest(advanced=name, value=value), self.assertRaisesRegex(ValueError, "newton_atol"):
                    guide.run_experiment(material, N=4, newton_atol=NUMPY_ATOL,
                                         numerical={name: value})
            with self.assertRaisesRegex(ValueError, "newton_atol"):
                guide.run_experiment(material, N=4, newton_atol="1e-350")
            solve.assert_not_called()

    def test_tiny_mpmath_tolerance_is_preserved_without_float_underflow(self):
        result = guide.run_experiment(
            preset_material(1), N=4, newton_atol="1e-350", accuracy="high_precision",
            model="manufactured_linear", numerical={"max_iterations": 0, "use_globalization": False},
        )
        arithmetic = result.solution.arithmetic
        self.assertTrue(arithmetic.use_mpmath)
        self.assertFalse(result.solution.converged)
        self.assertGreater(arithmetic.scalar("1e-350"), 0)
        self.assertEqual(result.settings["numerical"]["newton_atol"], "1e-350")
        self.assertEqual(arithmetic.scalar(result.settings["numerical"]["tolerance"]),
                         arithmetic.scalar("1e-350"))
        self.assertEqual(result.solution.newton_result.stages[-1].target_norm,
                         arithmetic.scalar("1e-350"))

    def test_numpy_scalar_tolerances_preserve_the_requested_spelling(self):
        for atol in (np.float64("1e-10"), np.longdouble("1.0000000000000000001e-10")):
            with self.subTest(type=type(atol).__name__):
                result = guide.run_experiment(preset_material(1), N=4, newton_atol=atol,
                                              model="manufactured_linear")
                self.assertTrue(result.solution.converged)
                self.assertEqual(result.settings["numerical"]["newton_atol"], str(atol))
                self.assertEqual(result.solution.newton_result.stages[-1].target_norm,
                                 result.solution.arithmetic.scalar(atol))

    def test_automatic_quadrature_and_explicit_order_preserve_the_basis_size(self):
        for order, expected in ((None, 14), (17, 17)):
            with self.subTest(order=order):
                output = io.StringIO()
                with redirect_stdout(output):
                    result = guide.run_experiment(
                        preset_material(1), N=5, newton_atol=NUMPY_ATOL,
                        model="manufactured_linear", verbosity=1,
                        numerical={"quadrature_order": order},
                    )
                self.assertTrue(result.solution.converged)
                self.assertEqual(result.solution.N, 5)
                self.assertEqual(result.settings["numerical"]["quadrature_order"], expected)
                self.assertEqual(result.solution.quadrature_history[0].order, expected)
                self.assertIn("N=5", output.getvalue().splitlines()[0])
                self.assertIn("9 unknowns", output.getvalue().splitlines()[0])

    def test_every_requested_material_count_runs_the_bsk_example(self):
        for kind, count in product(("constant", "graded"), (1, 2, 3, 5, 7, 10)):
            with self.subTest(kind=kind, interfaces=count):
                result = guide.run_experiment(
                    preset_material(count, kind=kind), N=SMALL_N, newton_atol=NUMPY_ATOL,
                    model="BSK", verbosity=0,
                    numerical=SMALL_NUMERICS,
                )
                self.assertTrue(result.solution.converged)
                self.assertTrue(result.solution.nonlinear_converged)
                self.assertTrue(result.solution.quadrature_converged)
                self.assertEqual(result.model, "BSK")
                self.assertIsNone(result.case)
                self.assertEqual(len(result.solution.physical.epsilon.interfaces), count)
                self.assertEqual(result.solution.solver_type, "galerkin")
                self.assertEqual(result.solution.u(result.solution.arithmetic.inf), 0)
                self.assertLess(abs(result.solution.u(0)-0.05), 1e-8)

    def test_known_answer_models_and_graded_material_use_the_same_workflow(self):
        material = preset_material(3, kind="graded")
        for model in ("manufactured_linear", "manufactured_nonlinear"):
            with self.subTest(model=model):
                result = guide.run_experiment(
                    material, N=SMALL_N, newton_atol=NUMPY_ATOL,
                    model=model, verbosity=0, numerical=SMALL_NUMERICS,
                )
                self.assertTrue(result.solution.converged)
                self.assertIsNotNone(result.case)
                arithmetic = result.solution.arithmetic
                for x in (arithmetic.scalar("0.25"), arithmetic.scalar("1.5")):
                    self.assertTrue(arithmetic.isfinite(result.case.u(x, arithmetic)))
                    self.assertTrue(arithmetic.isfinite(result.solution.flux(x)))
                self.assertEqual(result.material.kind, "graded")

    def test_high_precision_run_does_not_change_global_mpmath_precision(self):
        mp.mp.prec = 77
        result = guide.run_experiment(
            preset_material(1), N=4, newton_atol=MPMATH_ATOL,
            model="manufactured_nonlinear", accuracy="high_precision",
            verbosity=0, numerical=dict(quadrature_order=12,
                                      quadrature_atol="1e-10", quadrature_rtol="1e-10"),
        )
        self.assertTrue(result.solution.converged)
        arithmetic = result.solution.arithmetic
        self.assertTrue(arithmetic.use_mpmath)
        self.assertEqual(arithmetic.dps, 50)
        self.assertIsInstance(result.solution.u(arithmetic.scalar("0.375")),
                              type(arithmetic.scalar(0)))
        self.assertEqual(mp.mp.prec, 77)

    def test_custom_piecewise_smooth_function_uses_declared_interfaces(self):
        def epsilon(x, arithmetic):
            base = arithmetic.scalar(1 if x < arithmetic.scalar("0.7") else 2)
            return base+x/(10*(1+x))

        material = Coefficient(epsilon, lower_bound="1", upper_bound="2.1",
                               wall_value="1", interfaces=("0.7",))
        result = guide.run_experiment(material, N=SMALL_N, newton_atol=NUMPY_ATOL,
                                      verbosity=0, numerical=SMALL_NUMERICS)
        self.assertTrue(result.solution.converged)
        self.assertEqual(len(result.solution.physical.epsilon.interfaces), 1)
        declaration = result.settings["material"]
        self.assertFalse(declaration["reconstructible"])
        self.assertIn("function", declaration["note"])
        with tempfile.TemporaryDirectory(prefix="overscreening-custom-export-", dir="/tmp") as temporary:
            paths = guide.export_experiment(result, temporary)
            exported = json.loads(Path(paths["settings"]).read_text())
            self.assertEqual(exported["material"]["type"], "custom_function")
            self.assertEqual(exported["material"]["interfaces"], ["0.7"])
            self.assertFalse(exported["material"]["reconstructible"])

    def test_report_distinguishes_equations_integration_and_unchecked_resolution(self):
        result = guide.run_experiment(preset_material(1), N=SMALL_N,
                                      newton_atol=NUMPY_ATOL, numerical=SMALL_NUMERICS, verbosity=0)
        report = guide.report_experiment(result)
        self.assertIsInstance(report, str)
        text = report.lower()
        self.assertIn("equations", text)
        self.assertIn("integration", text)
        self.assertIn("resolution", text)
        self.assertIn("not checked", text)
        self.assertIn("N={}".format(SMALL_N), report)
        self.assertIn("{} unknowns".format(2*SMALL_N-1), report)
        self.assertIn("Newton absolute residual tolerance", report)
        self.assertIn("Final physical residual norm", report)
        failed = guide.run_experiment(
            preset_material(1), N=SMALL_N, newton_atol=NUMPY_ATOL,
            numerical=dict(SMALL_NUMERICS, max_iterations=0,
                                              use_globalization=False), verbosity=0,
        )
        self.assertFalse(failed.solution.converged)
        failed_report = guide.report_experiment(failed).lower()
        self.assertTrue(any(word in failed_report for word in ("failed", "not solved", "no")))
        self.assertNotIn("equations solved: yes", failed_report)

    def test_exports_preserve_the_material_and_typed_solution(self):
        material = LayeredMaterial((("0", "1.0000000000000000000000000000000000000001"),
                                    ("1", "2")))
        result = guide.run_experiment(
            material, N=4, newton_atol=MPMATH_ATOL,
            model="manufactured_linear", accuracy="high_precision", verbosity=0,
            numerical=dict(quadrature_order=12, quadrature_atol="1e-10",
                           quadrature_rtol="1e-10"),
        )
        self.assertTrue(result.solution.converged)
        with tempfile.TemporaryDirectory(prefix="overscreening-layered-export-", dir="/tmp") as temporary:
            paths = guide.export_experiment(result, temporary)
            self.assertTrue(paths)
            settings_path = next(Path(path) for name, path in paths.items() if "settings" in name)
            solution_path = next(Path(path) for name, path in paths.items() if "solution" in name)
            profile_path = next(Path(path) for name, path in paths.items()
                                if "profile" in name or str(path).endswith(".csv"))
            settings = json.loads(settings_path.read_text())
            self.assertEqual(settings["numerical"]["N"], 4)
            self.assertEqual(settings["numerical"]["newton_atol"], MPMATH_ATOL)
            self.assertEqual(result.solution.arithmetic.scalar(settings["numerical"]["tolerance"]),
                             result.solution.arithmetic.scalar(MPMATH_ATOL))
            self.assertIn("1.0000000000000000000000000000000000000001",
                          json.dumps(settings))
            snapshot = json.loads(solution_path.read_text())
            self.assertEqual(snapshot["solver_type"], "galerkin")
            self.assertTrue(snapshot["status"]["converged"])
            self.assertEqual(snapshot["basis"]["N"], 4)
            self.assertEqual(snapshot["basis"]["unknown_count"], 7)
            self.assertEqual(snapshot["arithmetic"]["backend"], "mpmath")
            self.assertTrue(snapshot["coefficients"]["u"])
            self.assertTrue(snapshot["coefficients"]["w"])
            loaded = load_solution(solution_path, problem=result.case.make_problem(),
                                   solver="galerkin")
            self.assertTrue(loaded.arithmetic.use_mpmath)
            self.assertEqual(loaded.N, 4)
            self.assertEqual(loaded.arithmetic.dps, result.solution.arithmetic.dps)
            x = result.solution.arithmetic.scalar("0.375")
            for field in ("u", "w", "flux"):
                self.assertEqual(getattr(loaded, field)(loaded.arithmetic.scalar("0.375")),
                                 getattr(result.solution, field)(x))
            with profile_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 5)
            self.assertIn("x", rows[0])
            self.assertIn("u", rows[0])
            arithmetic = result.solution.arithmetic
            for row in rows:
                self.assertTrue(arithmetic.isfinite(arithmetic.scalar(row["x"])))
                self.assertTrue(arithmetic.isfinite(arithmetic.scalar(row["u"])))

    def test_invalid_model_and_accuracy_are_clear_errors(self):
        material = preset_material(1)
        for options in (dict(model="unknown"), dict(accuracy="unknown"), dict(accuracy="refined")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "model|accuracy"):
                guide.run_experiment(material, N=SMALL_N, newton_atol=NUMPY_ATOL, **options)

    def test_resolution_comparison_reports_field_changes_separately_from_solver_success(self):
        coarse = guide.run_experiment(
            preset_material(1), N=4, newton_atol=NUMPY_ATOL, verbosity=0,
            numerical=dict(quadrature_order=None, quadrature_atol="1e-8",
                           quadrature_rtol="1e-8"),
        )
        self.assertTrue(coarse.solution.converged)
        comparison = guide.compare_resolution(coarse, fine_N=7, newton_atol="2e-11", verbosity=0)
        self.assertIs(comparison.coarse, coarse)
        self.assertEqual(comparison.fine.solution.N, 7)
        self.assertEqual(comparison.fine.settings["numerical"]["N"], 7)
        fine_order = comparison.fine.settings["numerical"]["quadrature_order"]
        self.assertGreaterEqual(fine_order, 2*comparison.fine.solution.N+4)
        self.assertGreaterEqual(fine_order, coarse.solution.plan.order)
        self.assertTrue(comparison.fine.solution.converged)
        self.assertEqual(comparison.fine.settings["numerical"]["newton_atol"], "2e-11")
        self.assertEqual(comparison.fine.solution.newton_result.stages[-1].target_norm,
                         comparison.fine.solution.arithmetic.scalar("2e-11"))
        self.assertEqual(set(comparison.relative_changes), {"u", "w", "flux"})
        arithmetic = comparison.fine.solution.arithmetic
        self.assertTrue(all(arithmetic.isfinite(value) and value >= 0
                            for value in comparison.relative_changes.values()))
        report = guide.report_experiment(coarse, resolution=comparison).lower()
        self.assertIn("resolution", report)
        self.assertIn("compared n=4 and n=7", report)
        self.assertNotIn("resolution: not checked", report)
        with tempfile.TemporaryDirectory(prefix="overscreening-resolution-export-", dir="/tmp") as temporary:
            paths = guide.export_experiment(coarse, temporary, resolution=comparison)
            settings = json.loads(Path(paths["settings"]).read_text())
            self.assertEqual(settings["resolution"]["coarse_N"], 4)
            self.assertEqual(settings["resolution"]["fine_N"], 7)

    def test_finer_resolution_is_required_and_validated_before_another_solve(self):
        coarse = guide.run_experiment(preset_material(1), N=4, newton_atol=NUMPY_ATOL,
                                      numerical=SMALL_NUMERICS)
        with patch.object(guide, "run_defined_problem", side_effect=AssertionError("invalid fine_N must not solve")) as run:
            with self.assertRaisesRegex(TypeError, "fine_N"):
                guide.compare_resolution(coarse, newton_atol=NUMPY_ATOL)
            for fine_N in (None, True, 0, 1, 3, 4, 5.0, 5.5, "8"):
                with self.subTest(fine_N=fine_N), self.assertRaisesRegex(ValueError, "fine_N"):
                    guide.compare_resolution(coarse, fine_N=fine_N, newton_atol=NUMPY_ATOL)
            run.assert_not_called()

    def test_comparison_and_sweep_require_their_own_valid_newton_tolerance(self):
        coarse = guide.run_experiment(preset_material(1), N=4, newton_atol=NUMPY_ATOL,
                                      model="manufactured_linear", numerical=SMALL_NUMERICS)
        invalid = (True, 0, -1, "nan", "inf", "1e-350")
        with patch.object(guide, "solve", side_effect=AssertionError("invalid tolerance must not solve")) as solve:
            with self.assertRaisesRegex(TypeError, "newton_atol"):
                guide.compare_resolution(coarse, fine_N=5)
            for atol in invalid:
                with self.subTest(operation="comparison", atol=atol), self.assertRaisesRegex(ValueError, "newton_atol"):
                    guide.compare_resolution(coarse, fine_N=5, newton_atol=atol)
            solve.assert_not_called()
        with patch.object(guide, "run_experiment", side_effect=AssertionError("invalid sweep must not start")) as run:
            with self.assertRaisesRegex(TypeError, "newton_atol"):
                guide.run_preset_sweep(N_by_interfaces={1: 4}, interface_counts=(1,))
            for atol in invalid:
                with self.subTest(operation="sweep", atol=atol), self.assertRaisesRegex(ValueError, "newton_atol"):
                    guide.run_preset_sweep(N_by_interfaces={1: 4}, interface_counts=(1,),
                                           newton_atol=atol)
            for name in ("tolerance", "newton_atol"):
                with self.subTest(advanced=name), self.assertRaisesRegex(ValueError, "newton_atol"):
                    guide.run_preset_sweep(N_by_interfaces={1: 4}, interface_counts=(1,),
                                           newton_atol=NUMPY_ATOL, numerical={name: NUMPY_ATOL})
            run.assert_not_called()

    def test_high_precision_comparison_preserves_physics_and_uses_the_fine_tolerance(self):
        coarse = guide.run_experiment(preset_material(1), N=4, newton_atol="1e-30",
                                      accuracy="high_precision", model="manufactured_linear", delta="2")
        comparison = guide.compare_resolution(coarse, fine_N=5, newton_atol="4e-32")
        fine = comparison.fine
        self.assertTrue(fine.solution.converged)
        self.assertEqual(fine.solution.arithmetic.use_mpmath, coarse.solution.arithmetic.use_mpmath)
        self.assertEqual(fine.solution.arithmetic.dps, 50)
        self.assertEqual(fine.settings["physical"], coarse.settings["physical"])
        self.assertEqual(fine.solution.physical.delta, 2)
        self.assertEqual(fine.solution.physical.boundary_conditions,
                         coarse.solution.physical.boundary_conditions)
        self.assertEqual(fine.solution.physical.epsilon.interfaces,
                         coarse.solution.physical.epsilon.interfaces)
        self.assertEqual(coarse.settings["numerical"]["newton_atol"], "1e-30")
        self.assertEqual(fine.settings["numerical"]["newton_atol"], "4e-32")
        self.assertEqual(fine.solution.newton_result.stages[-1].target_norm,
                         fine.solution.arithmetic.scalar("4e-32"))
        for x, u in (("0.2", "0.3"), ("2", "0.1")):
            self.assertEqual(fine.solution.physical.physical_rhs(x, u),
                             coarse.solution.physical.physical_rhs(x, u))

    def test_sweep_requires_every_selected_resolution_before_any_work(self):
        invalid_options = (
            dict(N_by_interfaces=None),
            dict(N_by_interfaces=[(1, 4)]),
            dict(N_by_interfaces={}),
            dict(N_by_interfaces={1: 4}, interface_counts=(1, 3)),
            dict(N_by_interfaces={1: 4, 3: True}, interface_counts=(1, 3)),
            dict(N_by_interfaces={1: 4, 3: 1}, interface_counts=(1, 3)),
            dict(N_by_interfaces={1: 4, 3: 4.5}, interface_counts=(1, 3)),
            dict(N_by_interfaces={1: 4, 4: 5}, interface_counts=(1, 4)),
            dict(N_by_interfaces={1: 4}, numerical={"N": 4}),
        )
        with patch.object(guide, "run_experiment", side_effect=AssertionError("invalid sweep must not start")) as run:
            with self.assertRaisesRegex(TypeError, "N_by_interfaces"):
                guide.run_preset_sweep(interface_counts=(1,), newton_atol=NUMPY_ATOL)
            for options in invalid_options:
                arguments = dict(interface_counts=(1,), models=("BSK",), newton_atol=NUMPY_ATOL)
                arguments.update(options)
                with self.subTest(options=options), self.assertRaises(ValueError):
                    guide.run_preset_sweep(**arguments)
            run.assert_not_called()

    def test_sweep_uses_each_chosen_resolution_and_ignores_unselected_entries(self):
        results = guide.run_preset_sweep(
            N_by_interfaces={1: 4, 2: 5, 10: None}, interface_counts=(1, 2),
            newton_atol=NUMPY_ATOL,
            models=("manufactured_linear",), numerical=SMALL_NUMERICS,
        )
        self.assertEqual([result.solution.N for result in results], [4, 5])
        self.assertEqual([result.material.interface_count for result in results], [1, 2])
        self.assertTrue(all(result.solution.converged for result in results))

    def test_optional_sweep_preserves_the_selected_physical_parameters(self):
        results = guide.run_preset_sweep(
            N_by_interfaces={1: SMALL_N},
            newton_atol=NUMPY_ATOL,
            interface_counts=(1,), models=("BSK", "manufactured_nonlinear"),
            delta="2", wall_potential="0.07", mu="0.2", gamma="0.3", sigma="3",
            numerical=SMALL_NUMERICS, verbosity=0,
        )
        self.assertEqual([result.model for result in results], ["BSK", "manufactured_nonlinear"])
        self.assertTrue(all(result.solution.converged for result in results))
        bsk, manufactured = results
        for result in results:
            self.assertEqual(result.solution.physical.delta, 2)
            self.assertEqual(result.settings["physical"]["delta"], "2")
        self.assertAlmostEqual(bsk.solution.u(0), 0.07, places=12)
        arithmetic = bsk.solution.arithmetic
        reference = BSKProblem(delta="2", sigma="3", mu="0.2", gamma="0.3").bind(arithmetic)
        for x, u in (("0.2", "0.3"), ("3", "-0.1")):
            self.assertAlmostEqual(bsk.solution.physical.physical_rhs(x, u),
                                   reference.physical_rhs(x, u), places=14)
        for name, value in (("wall_potential", "0.07"), ("mu", "0.2"),
                            ("gamma", "0.3"), ("sigma", "3")):
            self.assertEqual(bsk.settings["physical"][name], value)
        # Known-answer checks deliberately retain their prescribed wall data.
        self.assertEqual(manufactured.solution.physical.boundary_conditions[0],
                         (1, None, None, -6, None))

    def test_notebook_uses_explicit_resolution_keywords_and_portable_imports(self):
        notebook = json.loads(NOTEBOOK.read_text())
        identifiers = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(isinstance(value, str) and value for value in identifiers))
        self.assertEqual(len(identifiers), len(set(identifiers)))
        sources = ["".join(cell.get("source", ())) for cell in notebook["cells"]]
        text = "\n".join(sources)
        self.assertNotIn("%cd", text)
        self.assertNotIn("sys.path", text)
        self.assertIn("overscreening_breakdown", text)
        required = {"run_defined_problem": ("N", "newton_atol", "material"),
                    "compare_resolution": ("fine_N", "newton_atol")}
        seen = set()
        definitions = set()
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = notebook_code("".join(cell["source"]))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    definitions.add(node.name)
                if not isinstance(node, ast.Call):
                    continue
                name = (node.func.id if isinstance(node.func, ast.Name)
                        else node.func.attr if isinstance(node.func, ast.Attribute) else None)
                self.assertNotIn(name, ("run_experiment", "run_manufactured_convergence", "run_preset_sweep"))
                if name not in required:
                    continue
                seen.add(name)
                arguments = {keyword.arg: keyword.value for keyword in node.keywords}
                for keyword in required[name]:
                    self.assertIn(keyword, arguments,
                                  "Every {} call should expose {}".format(name, keyword))
                numerical = arguments.get("numerical")
                if isinstance(numerical, ast.Dict):
                    advanced = {key.value for key in numerical.keys if isinstance(key, ast.Constant)}
                    self.assertFalse(advanced.intersection(("N", "tolerance", "newton_atol")))
        self.assertEqual(seen, set(required))
        self.assertTrue({"define_bsk_problem", "define_linear_problem", "define_nonlinear_problem"}
                        .issubset(definitions))

    def test_notebook_core_experiments_execute_without_modifying_saved_user_work(self):
        directory_before = Path.cwd()
        namespace, calls, report, outputs = self.execute_notebook(CORE_CELL_IDS, CORE_CONTROLS)
        self.assertEqual(len(calls), 3, "The explicit core selection must remain bounded")
        self.assertTrue(set(EXTRA_RUN_IDS+EXTRA_PLOT_IDS).issubset(report.skipped))
        self.assertNotIn("high-precision-example", report.executed)
        for call in calls:
            self.assertIn("N", call.kwargs)
            self.assertIn("newton_atol", call.kwargs)
            advanced = call.kwargs.get("numerical") or {}
            self.assertFalse(set(advanced).intersection(("N", "tolerance", "newton_atol")))
        self.assertFalse(namespace["run_large_bsk"] or namespace["run_extreme_bsk_mpmath"])
        cases = namespace["notebook_cases"]
        self.assertEqual(set(cases), {"selected", "linear-check", "nonlinear-check"})
        self.assertTrue(all(case.experiment.solution.converged for case in cases.values()))
        for result, problem_name in ((namespace["experiment"], "selected_problem"),
                                     (namespace["linear_check"], "linear_problem"),
                                     (namespace["nonlinear_check"], "nonlinear_problem")):
            self.assertIs(result.problem, namespace[problem_name])
            self.assertEqual(result.solution.N, 8)
        for filename in ("settings.json", "solution.json", "profiles.csv"):
            self.assertEqual(len(list(outputs.rglob(filename))), 3)
        self.assertGreaterEqual(len(list(outputs.rglob("*.pdf"))), 4)
        self.assertEqual(Path.cwd(), directory_before)

    def test_disabled_extra_cells_and_their_plots_perform_no_solves(self):
        cells = ("setup", "experiment-inputs", "material-preview", "bsk-problem-definition")+EXTRA_RUN_IDS+EXTRA_PLOT_IDS
        controls = {name: value for name, value in CORE_CONTROLS.items() if name in cells}
        controls.update({EXTRA_RUN_IDS[0]: {"run_large_bsk": False},
                         EXTRA_RUN_IDS[1]: {"run_extreme_bsk_mpmath": False}})
        namespace, calls, _, _ = self.execute_notebook(cells, controls)
        self.assertFalse(calls)
        self.assertIsNone(namespace["bsk_large_experiment"])
        self.assertIsNone(namespace["bsk_extreme_experiment"])
        self.assertEqual(namespace["bsk_large_figures"], {})
        self.assertEqual(namespace["bsk_extreme_figures"], {})
        self.assertEqual(namespace["notebook_cases"], {})

    def test_small_enabled_extras_keep_results_figures_and_exported_cases_together(self):
        cells = CORE_CELL_IDS+EXTRA_RUN_IDS+EXTRA_PLOT_IDS
        controls = {name: dict(value) for name, value in CORE_CONTROLS.items()}
        controls["experiment-inputs"]["wall_potential"] = 0.001
        controls[EXTRA_RUN_IDS[0]] = dict(run_large_bsk=True, N_large=8, newton_atol_large="1e-10")
        controls[EXTRA_RUN_IDS[1]] = dict(run_extreme_bsk_mpmath=True, N_extreme=8,
                                          newton_atol_extreme="1e-20")
        namespace, calls, _, _ = self.execute_notebook(cells, controls)
        self.assertEqual(len(calls), 5)
        self.assertEqual(namespace["model"], "BSK")
        expected = {"selected": ("BSK", 0.001), "bsk-large": ("BSK_large", 0.01),
                    "bsk-extreme": ("BSK_large_large", 1.0)}
        cases, saved = namespace["notebook_cases"], namespace["saved_files"]
        self.assertEqual(set(cases), {"selected", "bsk-large", "bsk-extreme", "linear-check", "nonlinear-check"})
        self.assertEqual(set(saved), set(cases))
        for name, (model, wall) in expected.items():
            case = cases[name]
            result = case.experiment
            self.assertEqual(case.name, name)
            self.assertEqual(result.model, model)
            self.assertTrue(result.solution.converged)
            self.assertAlmostEqual(float(result.solution.u(0)), wall, places=14)
            axis = next(axis for axis in case.figures["fields"].axes
                        if "potential" in axis.get_ylabel().lower())
            line = next(line for line in axis.lines if len(line.get_xdata()) > 2)
            self.assertAlmostEqual(float(line.get_ydata()[0]), wall, places=14)
            settings = json.loads(Path(saved[name]["settings"]).read_text())
            self.assertEqual(settings["case_name"], name)
            self.assertEqual(settings["model"], model)
            restored = load_solution(saved[name]["solution"], problem=result.problem, solver="galerkin")
            self.assertEqual(tuple(restored.coefficients), tuple(result.solution.coefficients))
        self.assertIs(cases["selected"].experiment, namespace["experiment"])
        self.assertIs(cases["bsk-large"].experiment, namespace["bsk_large_experiment"])
        self.assertIs(cases["bsk-extreme"].experiment, namespace["bsk_extreme_experiment"])
        self.assertIsNot(cases["selected"].figures["fields"], cases["bsk-large"].figures["fields"])
        self.assertIsNot(cases["bsk-large"].figures["fields"], cases["bsk-extreme"].figures["fields"])

