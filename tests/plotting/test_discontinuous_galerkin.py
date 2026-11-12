"""Layered plots show their interfaces without connecting numerical flux jumps."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.layered import preset_material
from overscreening_breakdown.notebook_examples import discontinuous_galerkin as guide


INLINE_CHECK = r"""
import atexit
import base64
import json
import os
from pathlib import Path
import sys
import tempfile

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="overscreening-layered-inline-", dir="/tmp") as temporary:
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
    source = (
        "%matplotlib inline\n"
        "from matplotlib import pyplot as plt\n"
        "from overscreening_breakdown.layered import preset_material\n"
        "from overscreening_breakdown.notebook_examples.discontinuous_galerkin import "
        "run_experiment, plot_material, plot_experiment\n"
        "material=preset_material(3)\n"
        "experiment=run_experiment(material,N=4,newton_atol='1e-10',verbosity=0,numerical={"
        "'quadrature_order':12,'quadrature_atol':'1e-8','quadrature_rtol':'1e-8'})\n"
        "assert experiment.solution.converged\n"
        "material_figure=plot_material(material,x_max='8',samples=12)\n"
        "field_figure=plot_experiment(experiment,x_max='8',samples=12)\n"
        "plt.show()\n"
    )
    try:
        with capture_output(display=True) as captured:
            result = shell.run_cell(source, store_history=False)
            result.raise_error()
        pngs = [item.data["image/png"] for item in captured.outputs if "image/png" in item.data]
        assert len(pngs) == 2, "Expected a material preview and a physical field figure"
        for index, png in enumerate(pngs, 1):
            data = base64.b64decode(png) if isinstance(png, str) else png
            assert data.startswith(b"\x89PNG\r\n\x1a\n")
            (output/("layered-inline-{}.png".format(index))).write_bytes(data)
        print(json.dumps({"png_count": len(pngs)}))
    finally:
        shell.atexit_operations()
        TerminalInteractiveShell.clear_instance()
"""


class DiscontinuousGalerkinPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material = preset_material(3)
        cls.experiment = guide.run_experiment(
            cls.material, N=8, newton_atol="1e-10", model="BSK", verbosity=0,
            numerical=dict(quadrature_order=20,
                           quadrature_atol="1e-8", quadrature_rtol="1e-8"),
        )

    def setUp(self):
        self.addCleanup(plt.close, "all")

    def save_and_check(self, figure, name):
        self.assertIsInstance(figure, Figure)
        figure.canvas.draw()
        root = os.environ.get("CI_ARTIFACT_DIR")
        with tempfile.TemporaryDirectory(prefix="overscreening-layered-plots-", dir="/tmp") as temporary:
            directory = Path(root).resolve()/"discontinuous-galerkin" if root else Path(temporary)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/(name+".pdf")
            figure.savefig(path)
            contents = path.read_bytes()
            self.assertTrue(contents.startswith(b"%PDF-"))
            self.assertGreater(len(contents), 1000)

    def assert_resolution_title(self, figure, N, unknown_count):
        self.assertIsNotNone(figure._suptitle)
        title = figure._suptitle.get_text()
        self.assertIn("N={}".format(N), title)
        self.assertIn("{} unknowns".format(unknown_count), title)

    def test_material_plot_marks_every_requested_interface(self):
        for count in (1, 2, 3, 5, 7, 10):
            with self.subTest(interfaces=count):
                material = preset_material(count)
                with patch.object(plt, "show", side_effect=AssertionError("caller owns display")):
                    figure = guide.plot_material(material, x_max="8")
                verticals = []
                for axis in figure.axes:
                    for line in axis.lines:
                        x = np.asarray(line.get_xdata(), dtype=float)
                        if len(x) == 2 and x[0] == x[-1]:
                            verticals.append(x[0])
                expected = [float(value) for value in material.bind(NumpyArithmetic()).interfaces]
                self.assertTrue(all(any(abs(point-value) < 1e-12 for value in verticals)
                                    for point in expected))
                figure.canvas.draw()
                plt.close(figure)

    def test_material_preview_accepts_integration_profile_without_a_basis_size(self):
        figure = guide.plot_material(self.material, accuracy="strict_integration")
        self.assertNotIn("N=", figure.axes[0].get_title())
        figure.canvas.draw()
        with self.assertRaisesRegex(ValueError, "strict_integration"):
            guide.plot_material(self.material, accuracy="refined")

    def test_physical_flux_curves_are_split_at_material_interfaces(self):
        self.assertTrue(self.experiment.solution.converged)
        before = tuple(self.experiment.solution.coefficients)
        with patch.object(plt, "show", side_effect=AssertionError("caller owns display")):
            figure = guide.plot_experiment(self.experiment, x_max="8")
        axes = [axis for axis in figure.axes
                if "flux" in (axis.get_ylabel()+" "+axis.get_title()).lower()]
        self.assertTrue(axes, "The physical flux panel must be clearly labelled")
        interfaces = [float(value) for value in self.experiment.solution.physical.epsilon.interfaces]
        curves = []
        for axis in axes:
            for line in axis.lines:
                x = np.asarray(line.get_xdata(), dtype=float)
                if len(x) < 3 or not np.all(np.isfinite(x)) or np.ptp(x) == 0:
                    continue
                curves.append(x)
                self.assertFalse(any(x.min() < point < x.max() for point in interfaces),
                                 "A plotted flux segment must not bridge a jump")
        self.assertGreaterEqual(len(curves), len(interfaces)+1)
        self.assert_resolution_title(figure, 8, 15)
        self.save_and_check(figure, "physical-fields")
        self.assertEqual(before, tuple(self.experiment.solution.coefficients))

    def test_convergence_figure_uses_stored_solver_information(self):
        before = self.experiment.solution.newton_result
        with patch.object(plt, "show", side_effect=AssertionError("caller owns display")):
            figure = guide.plot_convergence(self.experiment)
        self.assertTrue(figure.axes)
        self.assertTrue(any(axis.lines for axis in figure.axes))
        self.assert_resolution_title(figure, 8, 15)
        self.assertIn("Newton absolute residual tolerance = 1e-10",
                      figure.axes[0].get_title())
        self.assertEqual(figure.axes[0].get_ylabel(), "residual / stage target")
        requested = self.experiment.solution.arithmetic.scalar("1e-10")
        for quadrature in self.experiment.solution.quadrature_history:
            for stage in quadrature.newton_result.stages:
                self.assertEqual(stage.target_norm, requested)
                if stage.converged:
                    self.assertLessEqual(stage.final_norm, requested)
        self.save_and_check(figure, "convergence")
        self.assertIs(self.experiment.solution.newton_result, before)

    def test_convergence_tolerance_label_preserves_mpmath_exponent(self):
        result = SimpleNamespace(
            solution=SimpleNamespace(arithmetic=MpmathArithmetic(80), N=4,
                                     converged=False, quadrature_history=(), newton_result=None),
            model="manufactured_linear", settings={"numerical": {"newton_atol": "1e-400"}},
        )
        figure = guide.plot_convergence(result)
        title = figure.axes[0].get_title()
        self.assertIn("Newton absolute residual tolerance = 1.0e-400", title)
        self.assertIn("unverified", figure._suptitle.get_text())
        figure.canvas.draw()
        result.settings = {}
        missing = guide.plot_convergence(result)
        self.assertIn("was not recorded", missing.axes[0].get_title())

    def test_error_plot_forms_small_differences_in_native_arithmetic(self):
        arithmetic = MpmathArithmetic(80)
        coefficient = preset_material(1).coefficient.bind(arithmetic)
        solution = SimpleNamespace(arithmetic=arithmetic,
                                   physical=SimpleNamespace(epsilon=coefficient),
                                   N=4, unknown_count=7, converged=True)
        case = SimpleNamespace()
        expected = []
        for field, base, difference in (("u", "1", "1e-60"),
                                        ("w", "-2", "1e-61"),
                                        ("flux", "5", "1e-62")):
            exact = arithmetic.scalar(base)
            numerical = exact+arithmetic.scalar(difference)
            expected.append(float(abs(numerical-exact)))
            setattr(solution, field, lambda x, value=numerical: value)
            setattr(case, field, lambda x, a, value=exact: value)
        result = SimpleNamespace(solution=solution, case=case,
                                 model="manufactured_linear")
        figure = guide.plot_errors(result, samples=5)
        self.assertEqual(len(figure.axes), 3)
        self.assert_resolution_title(figure, 4, 7)
        for axis, difference in zip(figure.axes, expected):
            curves = [np.asarray(line.get_ydata(), dtype=float) for line in axis.lines
                      if len(line.get_ydata()) >= 3]
            self.assertTrue(curves)
            self.assertTrue(all(np.allclose(values, difference, rtol=1e-10, atol=0)
                                for values in curves))
        self.save_and_check(figure, "native-manufactured-errors")

        # An exact sample has no positive logarithmic ordinate. It should be
        # explained, rather than replaced with a manufactured error floor.
        for field in ("u", "w", "flux"):
            reference = getattr(case, field)
            setattr(solution, field, lambda x, f=reference: f(x, arithmetic))
        exact_figure = guide.plot_errors(result, samples=5)
        for axis in exact_figure.axes:
            self.assertTrue(any("exactly zero" in text.get_text().lower()
                                for text in axis.texts))
            for line in axis.lines:
                if len(line.get_ydata()) >= 3:
                    self.assertTrue(np.all(np.isnan(line.get_ydata())))
        exact_figure.canvas.draw()

    def test_material_and_fields_render_as_actual_ipython_inline_pngs(self):
        root = os.environ.get("CI_ARTIFACT_DIR")
        with tempfile.TemporaryDirectory(prefix="overscreening-layered-inline-output-", dir="/tmp") as temporary:
            directory = (Path(root).resolve()/"discontinuous-galerkin"/"inline"
                         if root else Path(temporary))
            completed = subprocess.run(
                [sys.executable, "-B", "-c", INLINE_CHECK, str(directory)],
                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["png_count"], 2)
            self.assertEqual(len(list(directory.glob("*.png"))), 2)

