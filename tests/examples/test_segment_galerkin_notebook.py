"""Bounded finite-interval notebook experiments and explicit material inputs."""

import ast
from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp
from matplotlib import pyplot as plt

from overscreening_breakdown import load_solution
from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.galerkin.finite_solution import FiniteGalerkinSolution
from overscreening_breakdown.manufactured import finite_segment_case
from overscreening_breakdown.notebook_examples import discontinuous_galerkin as guide
from overscreening_breakdown.notebook_examples.segment_galerkin import segment_material, segment_material_table


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT/"examples/notebooks/segment_execution_example_galerkin_discontinuous.ipynb"
FLAGS = ("run_n_convergence", "run_r_convergence", "run_high_precision", "run_custom_epsilon",
         "run_all_interfaces", "run_homotopy_demo", "save_results")
BOUNDED = dict(R="4", N_selected=6, N_linear=6, N_nonlinear=6, verbosity=0,
               interface_count=3, material_style="constant", layers=None, model="BSK", accuracy="standard",
               newton_atol_selected="1e-10", newton_atol_linear="1e-10", newton_atol_nonlinear="1e-10",
               **dict.fromkeys(FLAGS, False))
PREFIX = {"imports", "experiment-settings", "material-preview", "define-bsk", "define-manufactured"}
HARD = {"hard-definition", "hard-runs", "hard-stages"}
CORE_SOLVES = {"selected-solve", "linear-verification", "nonlinear-verification"}


class NotebookControls(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self.values:
                node.value = ast.copy_location(ast.Constant(self.values[target.id]), node.value)
        return self.generic_visit(node)


class SegmentGalerkinNotebookTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)

    def execute(self, *, cell_ids=None, skip=(), controls=None):
        original = NOTEBOOK.read_bytes()
        notebook = json.loads(original)
        values = dict(BOUNDED)
        values.update(controls or {})
        namespace = {"__name__": "__main__"}
        output = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="overscreening-segment-notebook-", dir="/tmp") as temporary:
            with patch.dict(os.environ, {"OVERSCREENING_OUTPUT_DIR": temporary}), \
                    patch.object(guide, "run_defined_problem", wraps=guide.run_defined_problem) as run, \
                    patch.object(plt, "show"), redirect_stdout(output):
                for cell in notebook["cells"]:
                    if (cell["cell_type"] != "code" or cell.get("id") in skip
                            or (cell_ids is not None and cell.get("id") not in cell_ids)):
                        continue
                    source = "\n".join(line for line in "".join(cell["source"]).splitlines()
                                       if not line.lstrip().startswith("%"))
                    tree = ast.fix_missing_locations(NotebookControls(values).visit(ast.parse(source)))
                    exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
                calls = tuple(run.call_args_list)
        self.assertEqual(NOTEBOOK.read_bytes(), original)
        return namespace, calls, output.getvalue()

    def test_structure_exposes_finite_problems_and_per_solve_numerical_choices(self):
        notebook = json.loads(NOTEBOOK.read_text())
        ids = [cell.get("id") for cell in notebook["cells"]]
        self.assertTrue(all(isinstance(value, str) and value for value in ids))
        self.assertEqual(len(ids), len(set(ids)))
        expected = PREFIX | HARD | CORE_SOLVES | {
            "selected-plots", "material-gallery", "linear-errors", "nonlinear-errors",
            "n-convergence", "r-convergence", "high-precision", "custom-epsilon",
            "all-interface-cases", "hard-plots", "export-results",
        }
        self.assertTrue(expected.issubset(ids))
        constructors = set()
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "code")
        self.assertNotIn("%cd", source)
        self.assertNotIn("sys.path", source)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("\n".join(line for line in "".join(cell["source"]).splitlines()
                                        if not line.lstrip().startswith("%")))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else None
                arguments = {keyword.arg for keyword in node.keywords}
                if name == "run_defined_problem":
                    self.assertTrue({"N", "newton_atol", "material"}.issubset(arguments))
                    self.assertTrue(node.args or "problem" in arguments)
                if name in ("BSKProblem", "FluxProblem"):
                    constructors.add(name)
                    self.assertTrue({"domain", "boundary_conditions", "initial_guess"}.issubset(arguments))
                self.assertNotIn(name, ("run_experiment", "solve_nonlinear_problem"))
        self.assertEqual(constructors, {"BSKProblem", "FluxProblem"})

    def test_bounded_core_runs_three_finite_experiments_and_keeps_defined_objects(self):
        namespace, calls, _ = self.execute()
        self.assertEqual(len(calls), 3)
        results = {id(value): value for value in namespace.values()
                   if isinstance(value, guide.ExperimentResult)}
        self.assertEqual({result.model for result in results.values()},
                         {"BSK", "manufactured_linear", "manufactured_nonlinear"})
        self.assertEqual(len(results), 3)
        for result in results.values():
            solution = result.solution
            self.assertIsInstance(solution, FiniteGalerkinSolution)
            self.assertTrue(solution.converged)
            self.assertEqual(solution.domain, (0, 4))
            self.assertEqual((solution.N, solution.unknown_count), (6, 12))
            self.assertEqual(solution.u(0), solution.physical.boundary_conditions[0][0])
            self.assertEqual(solution.u(4), solution.physical.boundary_conditions[1][0])
            self.assertTrue(any(call.args[0] is result.problem for call in calls))
        self.assertTrue(all(namespace[flag] is False for flag in FLAGS))

    def test_disabled_optional_sections_launch_no_solver_work(self):
        skip = CORE_SOLVES | {"selected-plots", "linear-errors", "nonlinear-errors"}
        _, calls, _ = self.execute(skip=skip)
        self.assertFalse(calls)

    def test_all_material_counts_are_scaled_inside_the_finite_interval(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            for kind in ("constant", "graded"):
                for count in (1, 2, 3, 5, 7, 10):
                    with self.subTest(mpmath=a.use_mpmath, kind=kind, interfaces=count):
                        material = segment_material(count, R="3.5", kind=kind)
                        bound = material.bind(a)
                        self.assertEqual(len(bound.interfaces), count)
                        for index, point in enumerate(bound.interfaces, 1):
                            self.assertGreater(point, 0)
                            self.assertLess(point, a.scalar("3.5"))
                            error = abs(point-a.scalar("3.5")*index/(count+1))
                            self.assertLess(error, a.scalar("1e-40" if a.use_mpmath else "1e-12"))
                        table = segment_material_table(material, R="3.5")
                        self.assertIsInstance(table, str)
                        self.assertNotIn("infinity", table.lower())

    def test_every_material_count_supports_the_visible_finite_bsk_definition(self):
        namespace, calls, _ = self.execute(cell_ids=PREFIX)
        self.assertFalse(calls)
        for count in (1, 2, 3, 5, 7, 10):
            with self.subTest(interfaces=count):
                material = segment_material(count, R="4")
                problem = namespace["define_bsk_problem"](
                    material.coefficient, R="4", wall_potential="0.05", wall_third_derivative="0",
                    right_potential="0", right_slope="0", delta="2", gamma="0.5", sigma="2", mu="0.1",
                )
                result = guide.run_defined_problem(problem, N=6, newton_atol="1e-10", material=material)
                self.assertTrue(result.solution.converged)
                self.assertEqual(result.solution.domain, (0, 4))
                self.assertEqual(len(result.solution.physical.epsilon.interfaces), count)

    def test_finite_metrics_export_and_observers_use_the_physical_domain(self):
        material = segment_material(2, R="3.5", kind="graded")
        case = finite_segment_case(material, R="3.5", delta="2", nonlinear=True)
        problem = case.make_problem()
        events = []
        observer = events.append
        result = guide.run_defined_problem(
            problem, N=6, newton_atol="1e-10", material=material, case=case,
            model="manufactured_nonlinear", numerical={"progress_callback": observer},
        )
        self.assertTrue(result.solution.converged)
        self.assertIs(result._options["progress_callback"], observer)
        self.assertNotIn("progress_callback", result.settings["numerical"])
        self.assertTrue(result.settings["progress_observer_attached"])
        metrics = guide.manufactured_error_metrics(result, order=33)
        for values in metrics.values():
            self.assertGreater(values["reference_norm"], 0)
            self.assertGreaterEqual(values["absolute"], 0)
            self.assertAlmostEqual(values["relative"], values["absolute"]/values["reference_norm"], places=14)
        before = len(events)
        comparison = guide.compare_resolution(result, fine_N=8, newton_atol="1e-10")
        self.assertTrue(comparison.fine.solution.converged)
        self.assertIs(comparison.fine._options["progress_callback"], observer)
        self.assertGreater(len(events), before)
        with tempfile.TemporaryDirectory(prefix="overscreening-finite-guide-export-", dir="/tmp") as temporary:
            paths = guide.export_experiment(result, temporary, samples=17)
            with Path(paths["profiles"]).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            coordinates = [float(row["x"]) for row in rows]
            self.assertEqual((min(coordinates), max(coordinates)), (0, 3.5))
            loaded = load_solution(paths["solution"], problem=problem, solver="galerkin")
            self.assertEqual(loaded.domain, (0, 3.5))
            self.assertEqual(tuple(loaded.coefficients), tuple(result.solution.coefficients))
            outside = Path(temporary)/"outside"
            with self.assertRaises(ValueError):
                guide.export_experiment(result, outside, x_max="4")
            self.assertFalse(outside.exists())
        with patch.object(guide, "solve", side_effect=AssertionError("finite L must not solve")) as solve:
            with self.assertRaisesRegex(ValueError, "L"):
                guide.run_defined_problem(problem, N=6, newton_atol="1e-10", material=material,
                                           numerical={"L": "1"})
            solve.assert_not_called()

    def test_actual_finite_homotopy_demonstration_recovers_in_numpy_and_mpmath(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                namespace, calls, _ = self.execute(
                    cell_ids=PREFIX | HARD,
                    controls=dict(run_homotopy_demo=True, hard_use_mpmath=use_mpmath,
                                  use_mpmath_hard=use_mpmath, hard_precision=30, precision_hard=30),
                )
                self.assertEqual(len(calls), 3)
                runs = namespace["hard_results"]
                limited, recovered, control = (runs[name].solution for name in
                                                ("direct_limited", "homotopy", "direct_control"))
                self.assertFalse(limited.converged)
                self.assertEqual(limited.newton_result.reason, "max_iterations")
                self.assertTrue(recovered.converged and control.converged)
                self.assertTrue(recovered.newton_result.reference_result.converged)
                self.assertEqual(recovered.newton_result.homotopy, 0)
                self.assertTrue(recovered.quadrature_converged)
                self.assertTrue(any(stage.converged and 0 < stage.homotopy < 1
                                    for stage in recovered.newton_result.stages))
                self.assertEqual(recovered.arithmetic.use_mpmath, use_mpmath)
                if use_mpmath:
                    self.assertEqual(recovered.arithmetic.dps, 30)
                self.assertEqual(recovered.domain, (0, 2))
                self.assertTrue(all(call.args[0] is runs["homotopy"].problem for call in calls))
                a = recovered.arithmetic
                for field in ("u", "w", "flux"):
                    for text in ("0", "0.25", "1", "1.75", "2"):
                        x = a.scalar(text)
                        left, right = getattr(recovered, field)(x), getattr(control, field)(x)
                        self.assertLess(abs(left-right), a.scalar("1e-6")*max(1, abs(right)))
