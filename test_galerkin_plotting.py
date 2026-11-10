"""Actual mixed solutions rendered as bounded PDF and IPython PNG artifacts."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from matplotlib import pyplot as plt

from collocation_solution import CollocationSolution
from examples.flux_problem_collocation import make_problem as collocation_problem
from examples.flux_problem_galerkin import make_problem
from solution_plotting import plot_coefficients, plot_fields
from solvers import solve


INLINE_CHECK = textwrap.dedent(
    r"""
    import atexit
    import base64
    import json
    import os
    from pathlib import Path
    import sys
    import tempfile

    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="overscreening-galerkin-inline-", dir="/tmp") as temporary:
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
        try:
            with capture_output(display=True) as captured:
                result = shell.run_cell(
                    "%matplotlib inline\n"
                    "from examples.flux_problem_galerkin import make_problem\n"
                    "from solvers import solve\n"
                    "from solution_plotting import plot_fields\n"
                    "import matplotlib\n"
                    "assert getattr(matplotlib.RcParams, '_get', None) is not dict.get\n"
                    "from matplotlib import pyplot as plt\n"
                    "solution=solve(make_problem(),4,solver='galerkin',use_mpmath=True,"
                    "precision=50,tolerance='1e-35')\n"
                    "assert solution.converged and solution.quadrature_converged\n"
                    "assert solution.newton_result.homotopy==0\n"
                    "figure=plot_fields(solution,x=['0','0.125','0.5','1','2','4','10'])\n"
                    "assert len(figure.axes)==3\n"
                    "plt.show()\n",
                    store_history=False,
                )
                result.raise_error()
            pngs = [item.data["image/png"] for item in captured.outputs
                    if "image/png" in item.data]
            assert len(pngs)==1, "Expected exactly one actual Galerkin inline figure"
            for index, png in enumerate(pngs, 1):
                data = base64.b64decode(png) if isinstance(png, str) else png
                assert data.startswith(b"\x89PNG\r\n\x1a\n")
                (output/("galerkin-inline-{}.png".format(index))).write_bytes(data)
            print(json.dumps({"png_count": len(pngs)}))
        finally:
            shell.atexit_operations()
            TerminalInteractiveShell.clear_instance()
    """
)


class GalerkinPlottingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")
        temporary = tempfile.TemporaryDirectory(prefix="overscreening-galerkin-plots-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        artifact_root = os.environ.get("CI_ARTIFACT_DIR")
        self.output_directory = (Path(artifact_root).resolve()/"galerkin-plotting"
                                 if artifact_root else Path(temporary.name))
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def check_plots(self, use_mpmath):
        solution = solve(make_problem(), 4, solver="galerkin", use_mpmath=use_mpmath,
                         precision=50, tolerance="1e-35" if use_mpmath else "1e-11")
        self.assertTrue(solution.converged)
        self.assertEqual(solution.newton_result.homotopy, 0)
        arithmetic = solution.arithmetic
        middle = "1."+"0"*34+"1" if use_mpmath else "1"
        samples = [arithmetic.scalar(value) for value in ("0", "0.125", middle, "4", "10")]
        original_u = type(solution).u
        recorded = []

        def record_native_inputs(current, points):
            recorded.extend(points)
            if use_mpmath:
                self.assertTrue(all(isinstance(point, type(arithmetic.scalar(0))) for point in points))
            return original_u(current, points)

        with patch.object(type(solution), "u", new=record_native_inputs):
            with patch.object(plt, "show", side_effect=AssertionError("plot helper must not show")):
                figure = plot_fields(solution, x=samples)
                coefficients = plot_coefficients(solution)
        self.assertEqual(recorded, samples)
        if use_mpmath:
            self.assertGreater(recorded[2]-1, 0)
        self.assertEqual(len(figure.axes), 3)
        self.assertIn("7 unknowns", figure._suptitle.get_text())
        self.assertIn("N=4", figure._suptitle.get_text())
        for axis, callback in zip(figure.axes, (solution.u, solution.w, solution.flux)):
            expected = [float(callback(value)) for value in samples]
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), expected)
            self.assertIsNotNone(axis.get_legend())
        self.assertEqual(len(coefficients.axes), 2)
        self.assertEqual(len(coefficients.axes[0].lines[0].get_ydata()), 3)
        self.assertEqual(len(coefficients.axes[1].lines[0].get_ydata()), 4)
        coefficients.canvas.draw()
        backend = "mpmath-50" if use_mpmath else "numpy"
        directory = self.output_directory/backend
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/"galerkin-fields.pdf"
        path.unlink(missing_ok=True)
        figure.savefig(path)
        contents = path.read_bytes()
        self.assertTrue(contents.startswith(b"%PDF-"))
        self.assertGreater(len(contents), 1000)
        if not use_mpmath:
            # The same helpers also accept a captured collocation polynomial.
            physical = collocation_problem().bind(arithmetic)
            collocation = CollocationSolution(physical=physical, N=2, L=1,
                                             coefficients=["0.5", "-0.5", 0, 0])
            single = plot_coefficients(collocation)
            self.assertEqual(len(single.axes), 1)
            single.canvas.draw()
            existing, axes = plt.subplots(3)
            self.assertIs(plot_fields(collocation, ax=axes), existing)
            existing.canvas.draw()

    def test_numpy_galerkin_field_pdf_and_independent_coefficients(self):
        self.check_plots(False)

    def test_mpmath_galerkin_field_pdf_and_independent_coefficients(self):
        self.check_plots(True)

    def test_actual_mpmath_galerkin_ipython_inline_png(self):
        completed = subprocess.run(
            [sys.executable, "-B", "-c", INLINE_CHECK, str(self.output_directory/"inline")],
            cwd=Path(__file__).resolve().parent, text=True, capture_output=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["png_count"], 1)


if __name__ == "__main__":
    unittest.main()
