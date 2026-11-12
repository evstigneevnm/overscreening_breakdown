"""Bounded notebook comparison plots preserve numerical meaning and precision."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from overscreening_breakdown import MpmathArithmetic, solve
from overscreening_breakdown.continuation import ReferenceResult
from overscreening_breakdown.manufactured import rational_case
from overscreening_breakdown.nonlinear_solver import NewtonIteration, NewtonResult, NewtonStage


NOTEBOOK = Path(__file__).resolve().parents[2]/"examples/notebooks/execution_example.ipynb"
STRATEGIES = ("damped Newton", "homotopy", "adjoint fallback enabled")

INLINE_CHECK = r"""
import ast
import atexit
import base64
import json
import os
from pathlib import Path
import sys
import tempfile

notebook_path, output = Path(sys.argv[1]), Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="overscreening-comparison-inline-", dir="/tmp") as temporary:
    for variable, name in (("IPYTHONDIR", "ipython"), ("MPLCONFIGDIR", "matplotlib"),
                           ("JUPYTER_CONFIG_DIR", "jupyter-config"),
                           ("JUPYTER_DATA_DIR", "jupyter-data"),
                           ("JUPYTER_RUNTIME_DIR", "jupyter-runtime")):
        directory = Path(temporary, name)
        directory.mkdir()
        os.environ[variable] = str(directory)
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config
    config = Config()
    config.HistoryManager.enabled = False
    config.HistoryManager.hist_file = ":memory:"
    shell = TerminalInteractiveShell.instance(config=config)
    shell.display_formatter.active_types = ["text/plain", "image/png"]
    atexit.unregister(shell.atexit_operations)
    source = ("%matplotlib inline\n"
              "import numpy as np\n"
              "from matplotlib import pyplot as plt\n"
              "from overscreening_breakdown import solve\n"
              "from overscreening_breakdown.manufactured import rational_case\n")
    for cell in json.loads(notebook_path.read_text())["cells"]:
        text = "".join(cell.get("source", ()))
        if cell["cell_type"] != "code" or not any("def "+name in text for name in
                ("compare_globalization_strategies", "plot_globalization_comparisons")):
            continue
        tree = ast.parse(text)
        tree.body = [node for node in tree.body if isinstance(
            node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
        source += ast.unparse(tree)+"\n"
    source += ("comparisons=compare_globalization_strategies(verbosity=0)\n"
               "figures=plot_globalization_comparisons(comparisons,samples=21)\n"
               "assert set(figures)=={'solutions','differences','convergence'}\n"
               "plt.show()\n")
    try:
        with capture_output(display=True) as captured:
            result = shell.run_cell(source, store_history=False)
            result.raise_error()
        pngs = [item.data["image/png"] for item in captured.outputs
                if "image/png" in item.data]
        assert len(pngs) == 3, "Expected three notebook comparison figures"
        for index, png in enumerate(pngs, 1):
            data = base64.b64decode(png) if isinstance(png, str) else png
            assert data.startswith(b"\x89PNG\r\n\x1a\n")
            (output/("comparison-inline-{}.png".format(index))).write_bytes(data)
        print(json.dumps({"png_count": len(pngs)}))
    finally:
        shell.atexit_operations()
        TerminalInteractiveShell.clear_instance()
"""


def notebook_functions():
    """Load the two bounded helpers without executing any scientific sweeps."""
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = dict(np=np, plt=plt, solve=solve, rational_case=rational_case)
    found = set()
    names = ("compare_globalization_strategies", "plot_globalization_comparisons")
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", ()))
        if cell["cell_type"] != "code" or not any("def "+name in source for name in names):
            continue
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(
            node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
        found.update(name for name in names if name in namespace)
    if found != set(names):
        raise AssertionError("Notebook is missing the bounded comparison or plotting helper")
    return namespace


def solution_snapshot(solution):
    arithmetic = solution.arithmetic
    return (tuple(solution.coefficients), solution.newton_result.history,
            solution.newton_result.stages,
            arithmetic.context.prec if arithmetic.use_mpmath else None)


def synthetic_result(arithmetic, *, multiple_plans=False):
    """Distinct stage starts expose incorrect joins and lost plan offsets."""
    scalar = arithmetic.scalar

    def stage(q, initial, final, iterations, converged=True):
        return NewtonStage(scalar(q), scalar(initial), scalar(1), scalar(final),
                           iterations, converged, "converged" if converged else "max_iterations")

    def record(stage_number, iteration, q, residual):
        return NewtonIteration(stage_number, iteration, scalar(q), scalar(residual), scalar(1), 0)

    final = NewtonResult(
        arithmetic.vector([0]), True, "converged", scalar(0), scalar("0.125"), 2,
        (stage(0, 3, "0.125", 2),),
        (record(1, 1, 0, "1.5"), record(1, 2, 0, "0.125")),
    )
    if not multiple_plans:
        return final, ()
    reference = ReferenceResult(arithmetic.vector([0]), True, scalar("0.1"),
                                scalar(1), scalar(1), "converged")
    first = NewtonResult(
        arithmetic.vector([0]), True, "converged", scalar(0), scalar("0.25"), 4,
        (stage(0, 8, 4, 2, False), stage("0.5", 4, "0.5", 1),
         stage(0, 2, "0.25", 1)),
        (record(1, 1, 0, 6), record(2, 3, "0.5", "0.5"),
         record(3, 4, 0, "0.25")),
        reference_result=reference, continuation_attempts=2,
    )
    history = tuple(SimpleNamespace(
        order=order, subdivisions=1, newton_result=result,
        nonlinear_converged=True, quadrature_converged=index == 1,
    ) for index, (order, result) in enumerate(((8, first), (16, final))))
    return final, history


def synthetic_solution(arithmetic, *, perturbation="0", multiple_plans=False):
    result, plans = synthetic_result(arithmetic, multiple_plans=multiple_plans)
    perturbation = arithmetic.scalar(perturbation)

    def value(x):
        if np.ndim(x):
            return np.array([value(point) for point in x], dtype=object)
        x = arithmetic.scalar(x)
        return 1/(1+x)+perturbation

    return SimpleNamespace(
        arithmetic=arithmetic, u=value, newton_result=result,
        quadrature_history=plans, solver_type="galerkin", N=4,
        L=arithmetic.scalar(1), domain=(arithmetic.scalar(0), arithmetic.inf),
        unknown_count=7, coefficients=arithmetic.vector([0]*7),
        converged=True, nonlinear_converged=True, quadrature_converged=True,
        check_precision=lambda: None,
    )


class GlobalizationComparisonPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = notebook_functions()
        cls.comparisons = cls.namespace["compare_globalization_strategies"](verbosity=0)

    def setUp(self):
        self.addCleanup(plt.close, "all")

    def plot(self, comparisons, **options):
        with patch.object(plt, "show", side_effect=AssertionError("helper must not show")), patch.object(
            Figure, "savefig", side_effect=AssertionError("helper must not save")
        ), patch.dict(self.namespace, {"solve": lambda *args, **kwargs: self.fail(
            "Plotting must reuse stored solutions rather than solving again")}):
            return self.namespace["plot_globalization_comparisons"](comparisons, **options)

    def test_actual_numpy_and_mpmath_comparisons_render_without_changing_solutions(self):
        self.assertEqual(len(self.comparisons), 12)
        before = {key: solution_snapshot(value) for key, value in self.comparisons.items()}
        original_bits = mp.mp.prec
        for solver in ("collocation", "galerkin"):
            for backend in ("numpy", "mpmath"):
                group = [self.comparisons[(solver, backend, label)] for label in STRATEGIES]
                self.assertTrue(all(item.converged for item in group))
                self.assertTrue(all(tuple(item.coefficients) == tuple(group[0].coefficients)
                                    for item in group[1:]))
                self.assertTrue(all(item.newton_result.reference_result is None for item in group))
                self.assertTrue(all(item.newton_result.adjoint_steps == 0 for item in group))
        figures = self.plot(self.comparisons, samples=31)
        self.assertEqual(set(figures), {"solutions", "differences", "convergence"})
        root = os.environ.get("CI_ARTIFACT_DIR")
        with tempfile.TemporaryDirectory(prefix="overscreening-comparison-plots-", dir="/tmp") as temporary:
            directory = Path(root).resolve()/"globalization-comparison" if root else Path(temporary)
            directory.mkdir(parents=True, exist_ok=True)
            for name, figure in figures.items():
                self.assertIsInstance(figure, Figure)
                figure.canvas.draw()
                path = directory/(name+".pdf")
                figure.savefig(path)
                contents = path.read_bytes()
                self.assertTrue(contents.startswith(b"%PDF-"))
                self.assertGreater(len(contents), 1000)
        self.assertEqual(before, {key: solution_snapshot(value)
                                  for key, value in self.comparisons.items()})
        self.assertEqual(mp.mp.prec, original_bits)

    def test_tiny_mpmath_solution_difference_is_subtracted_before_float_conversion(self):
        arithmetic = MpmathArithmetic(80)
        baseline = synthetic_solution(arithmetic)
        perturbed = synthetic_solution(arithmetic, perturbation="1e-40")
        comparisons = {
            ("galerkin", "mpmath", "damped Newton"): baseline,
            ("galerkin", "mpmath", "homotopy"): perturbed,
            ("galerkin", "mpmath", "adjoint fallback enabled"): baseline,
        }
        figures = self.plot(comparisons, samples=5)
        lines = [line for axis in figures["differences"].axes for line in axis.lines
                 if "homotopy" in str(line.get_label()).lower()]
        self.assertTrue(lines, "The homotopy difference curve must be labelled")
        values = np.concatenate([np.asarray(line.get_ydata(), dtype=float).reshape(-1)
                                 for line in lines])
        values = values[np.isfinite(values)]
        self.assertTrue(values.size)
        np.testing.assert_allclose(values, 1e-40, rtol=1e-12, atol=0)

    def test_exactly_coincident_solutions_are_identified_without_a_positive_error_floor(self):
        arithmetic = MpmathArithmetic(80)
        solution = synthetic_solution(arithmetic)
        comparisons = {("galerkin", "mpmath", label): solution for label in STRATEGIES}
        figures = self.plot(comparisons, samples=5)
        figure = figures["differences"]
        labels = []
        for axis in figure.axes:
            labels.extend(axis.get_legend_handles_labels()[1])
            labels.extend(text.get_text() for text in axis.texts)
            for line in axis.lines:
                data = np.ma.asarray(line.get_ydata(), dtype=float).compressed()
                data = data[np.isfinite(data)]
                self.assertFalse(np.any(data > 0), "Exact zeros must not become fabricated positive errors")
        self.assertTrue(any("identical" in text.lower() or "zero" in text.lower()
                            for text in labels))
        figure.canvas.draw()

    def test_convergence_stages_and_quadrature_plans_have_distinct_segments(self):
        arithmetic = MpmathArithmetic(50)
        baseline = synthetic_solution(arithmetic)
        continued = synthetic_solution(arithmetic, multiple_plans=True)
        comparisons = {
            ("galerkin", "mpmath", "damped Newton"): baseline,
            ("galerkin", "mpmath", "homotopy"): continued,
            ("galerkin", "mpmath", "adjoint fallback enabled"): baseline,
        }
        saved = tuple((item.order, item.newton_result.history)
                      for item in continued.quadrature_history)
        figures = self.plot(comparisons, samples=5)
        segments = []
        for axis in figures["convergence"].axes:
            labelled = [line for line in axis.lines
                        if "homotopy" in str(line.get_label()).lower()]
            colors = {line.get_color() for line in labelled}
            for line in axis.lines:
                if line.get_color() not in colors:
                    continue
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                if x.size >= 2:
                    segments.append((tuple(x), tuple(y)))
        self.assertTrue(segments)
        ranges = {(min(x), max(x)) for x, _ in segments}
        self.assertTrue({(0, 2), (2, 3), (3, 4), (4, 6)}.issubset(ranges))
        self.assertFalse(any(low < 2 and high > 2 for low, high in ranges))
        self.assertFalse(any(low < 3 and high > 3 for low, high in ranges))
        self.assertFalse(any(low < 4 and high > 4 for low, high in ranges))
        self.assertTrue(any(x[-1] == 6 and y[-1] == 0.125 for x, y in segments))
        self.assertEqual(tuple(item.order for item in continued.quadrature_history),
                         tuple(order for order, _ in saved))
        self.assertEqual(tuple(item.newton_result.history for item in continued.quadrature_history),
                         tuple(history for _, history in saved))

    def test_bounded_comparison_displays_three_actual_inline_figures(self):
        with tempfile.TemporaryDirectory(prefix="overscreening-comparison-png-", dir="/tmp") as temporary:
            artifact_root = os.environ.get("CI_ARTIFACT_DIR")
            output = (Path(artifact_root).resolve()/"globalization-comparison/inline"
                      if artifact_root else Path(temporary))
            completed = subprocess.run(
                [sys.executable, "-B", "-c", INLINE_CHECK, str(NOTEBOOK), str(output)],
                cwd=NOTEBOOK.parents[2], capture_output=True, text=True, timeout=90,
            )
        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["png_count"], 3)

