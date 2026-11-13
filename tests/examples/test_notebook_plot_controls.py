"""Notebook view controls redraw stored solutions without changing numerics."""

import ast
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib import pyplot as plt

from overscreening_breakdown.notebook_examples import discontinuous_galerkin as guide
from tests.examples.notebook_tools import code_tree, execute_allowed_cells
from tests.examples.test_discontinuous_galerkin_notebook import (
    CORE_CELL_IDS as HALF_CELLS, CORE_CONTROLS as HALF_CONTROLS, NOTEBOOK as HALF_NOTEBOOK,
)
from tests.examples.test_segment_galerkin_notebook import (
    CELL_CONTROLS as FINITE_CONTROLS, CORE_CELL_IDS as FINITE_CELLS, NOTEBOOK as FINITE_NOTEBOOK,
)


PHYSICAL_PLOTS = {"plot_material", "plot_material_gallery", "plot_experiment", "plot_errors",
                  "plot_homotopy_comparison", "plot_domain_comparison"}


class NotebookPlotControlTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")

    def test_explicit_windows_and_core_view_control_wiring(self):
        for path, finite, settings_id in ((HALF_NOTEBOOK, False, "experiment-inputs"),
                                           (FINITE_NOTEBOOK, True, "experiment-settings")):
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text())
                settings = next(cell for cell in notebook["cells"] if cell["id"] == settings_id)
                controls = {
                    target.id
                    for statement in code_tree("".join(settings["source"])).body
                    if isinstance(statement, ast.Assign)
                    for target in statement.targets if isinstance(target, ast.Name)
                }
                # Saved display settings are editable research inputs. Check
                # their presence and wiring; execution tests supply their own
                # bounded windows independently of the user's chosen values.
                self.assertTrue({"plot_x_min", "plot_x_max", "iteration_xlim",
                                 "quadrature_xlim"}.issubset(controls))
                core_cells = set(FINITE_CELLS if finite else HALF_CELLS)
                count = 0
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    for call in ast.walk(code_tree("".join(cell["source"]))):
                        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                            continue
                        keywords = {item.arg: item.value for item in call.keywords}
                        if call.func.id in PHYSICAL_PLOTS:
                            count += 1
                            self.assertIn("x_min", keywords)
                            self.assertIn("x_max", keywords)
                            # Optional research cases can use their own explicit
                            # windows, e.g. x_max=30 for a larger-potential field.
                            if cell["id"] in core_cells:
                                self.assertEqual(ast.unparse(keywords["x_min"]), "plot_x_min")
                                if finite and call.func.id in ("plot_material", "plot_material_gallery"):
                                    self.assertEqual(ast.unparse(keywords["x_max"]),
                                                     "R if plot_x_max is None else plot_x_max")
                                else:
                                    self.assertEqual(ast.unparse(keywords["x_max"]), "plot_x_max")
                        if call.func.id == "plot_convergence":
                            self.assertNotIn("x_min", keywords)
                            self.assertNotIn("x_max", keywords)
                            for name in ("iteration_xlim", "quadrature_xlim"):
                                self.assertIn(name, keywords)
                                if cell["id"] in core_cells:
                                    self.assertEqual(ast.unparse(keywords[name]), name)
                self.assertGreater(count, 5)

    def test_bounded_core_windows_reach_all_case_figures_and_redraw_without_solves(self):
        for path, cells, template, settings_id, finite, upper in (
            (HALF_NOTEBOOK, HALF_CELLS, HALF_CONTROLS, "experiment-inputs", False, "50"),
            (FINITE_NOTEBOOK, FINITE_CELLS, FINITE_CONTROLS, "experiment-settings", True, "3"),
        ):
            with self.subTest(notebook=path.name), tempfile.TemporaryDirectory(
                    prefix="overscreening-view-controls-", dir="/tmp") as temporary:
                plt.close("all")
                original = path.read_bytes()
                notebook = json.loads(original)
                controls = {name: copy.deepcopy(value) for name, value in template.items() if name in cells}
                controls[settings_id].update(plot_x_min="0.25", plot_x_max=upper,
                                             iteration_xlim=(0, 50), quadrature_xlim=(0, 100))
                if not finite:
                    controls["export"]["save_results"] = False
                namespace = {"__name__": "__main__", "resolution_check": None}
                with patch.dict(os.environ, {"OVERSCREENING_OUTPUT_DIR": temporary}), \
                        patch.object(plt, "show"), redirect_stdout(io.StringIO()), \
                        patch.object(guide, "run_defined_problem", wraps=guide.run_defined_problem) as run:
                    execute_allowed_cells(notebook, cells, namespace, controls_by_cell=controls,
                                          filename=str(path))
                    self.assertEqual(run.call_count, 3)
                results = (namespace["experiment"], namespace["linear_check"], namespace["nonlinear_check"])
                coefficients = [tuple(result.solution.coefficients) for result in results]
                errors = [guide.manufactured_error_metrics(result, order=25) for result in results[1:]]
                history = namespace["convergence_figure" if finite else "selected_checks_figure"]
                self.assertEqual(history.axes[0].get_xlim(), (0, 50))
                self.assertEqual(history.axes[1].get_xlim(), (0, 100))
                for number in plt.get_fignums():
                    figure = plt.figure(number)
                    if figure is history:
                        continue
                    for axis in figure.axes:
                        self.assertEqual(axis.get_xlim(), (0.25, float(upper)))

                # Execute just the notebook's plotting expressions on the same
                # stored cases; source cells and numerical controls stay intact.
                namespace.update(plot_x_min="0.5", plot_x_max="2")
                redrawn = []
                with patch.object(guide, "run_defined_problem", side_effect=AssertionError("display must not solve")):
                    for cell in notebook["cells"]:
                        if cell["cell_type"] != "code" or cell["id"] not in cells:
                            continue
                        for call in ast.walk(code_tree("".join(cell["source"]))):
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in PHYSICAL_PLOTS:
                                expression = ast.fix_missing_locations(ast.Expression(body=copy.deepcopy(call)))
                                redrawn.append(eval(compile(expression, str(path), "eval"), namespace))
                self.assertTrue(redrawn)
                for figure in redrawn:
                    for axis in figure.axes:
                        self.assertEqual(axis.get_xlim(), (0.5, 2))
                self.assertEqual([tuple(result.solution.coefficients) for result in results], coefficients)
                self.assertEqual([guide.manufactured_error_metrics(result, order=25) for result in results[1:]], errors)
                self.assertEqual(path.read_bytes(), original)
