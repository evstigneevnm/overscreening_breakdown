"""Bounded PDF and IPython inline plotting checks for the CI suite."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt

from manufactured_problems import problem_2
from manufactured_solutions_solver import linear_manufactured_solutions_solver


INLINE_CHECK = textwrap.dedent(
    r"""
    import atexit
    import base64
    import json
    import os
    from pathlib import Path
    import sys
    import tempfile

    output_directory = Path(sys.argv[1])
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="overscreening-inline-") as temporary:
        for variable, name in (
            ("IPYTHONDIR", "ipython"),
            ("MPLCONFIGDIR", "matplotlib"),
            ("JUPYTER_CONFIG_DIR", "jupyter-config"),
            ("JUPYTER_DATA_DIR", "jupyter-data"),
            ("JUPYTER_RUNTIME_DIR", "jupyter-runtime"),
        ):
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
                    "import spectral1d\n"
                    "import manufactured_solutions_solver\n"
                    "import matplotlib\n"
                    "assert matplotlib.RcParams._get is not dict.get\n"
                    "from matplotlib import pyplot as plt\n"
                    "plt.plot([0.0, 0.5, 1.0], [0.0, 1.0, 0.0])\n"
                    "plt.title('Overscreening compatibility check')\n"
                    "plt.show()\n",
                    store_history=False,
                )
                result.raise_error()
            pngs = [output.data["image/png"] for output in captured.outputs
                    if "image/png" in output.data]
            assert pngs, "Inline backend produced no image/png display bundle"
            for index, png in enumerate(pngs, start=1):
                data = base64.b64decode(png) if isinstance(png, str) else png
                assert data.startswith(b"\x89PNG\r\n\x1a\n"), "Invalid PNG output"
                (output_directory / f"inline-{index}.png").write_bytes(data)
            print(json.dumps({"png_count": len(pngs)}))
        finally:
            shell.atexit_operations()
            TerminalInteractiveShell.clear_instance()
    """
)


class PlottingLightTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)
        self.addCleanup(plt.close, "all")
        temporary = tempfile.TemporaryDirectory(prefix="overscreening-plots-")
        self.addCleanup(temporary.cleanup)
        artifact_root = os.environ.get("CI_ARTIFACT_DIR")
        self.output_directory = (
            Path(artifact_root).resolve() / "plotting"
            if artifact_root else Path(temporary.name)
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def check_manufactured_pdfs(self, use_mpmath):
        directory = self.output_directory / ("mpmath-50" if use_mpmath else "numpy")
        directory.mkdir(parents=True, exist_ok=True)
        # Retained local artifacts must not let a failed render reuse old PDFs.
        for previous_pdf in directory.glob("*.pdf"):
            previous_pdf.unlink()
        problem = problem_2()
        # Plot helpers require two resolutions and three mapping parameters.
        # Set them on the problem, since solve() reads its sweep configuration.
        problem._N_all = [8, 16]
        problem._L_all = [1, 2, 3]
        solver = linear_manufactured_solutions_solver(
            default_folder=str(directory), show_figures=False,
            use_mpmath=use_mpmath, prec=50,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            solver.solve(problem)

        errors = np.asarray(solver.get_all_L2_norms(), dtype=float)
        self.assertEqual(errors.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(errors)))
        expected_suffixes = ("L2_vs_L.pdf", "N_coeffs.pdf", "L_coeffs.pdf", "solutions.pdf")
        pdfs = list(directory.glob("*.pdf"))
        self.assertEqual(len(pdfs), len(expected_suffixes))
        for suffix in expected_suffixes:
            with self.subTest(backend=directory.name, figure=suffix):
                matching = [path for path in pdfs if path.name.endswith(suffix)]
                self.assertEqual(len(matching), 1)
                contents = matching[0].read_bytes()
                self.assertTrue(contents.startswith(b"%PDF-"))
                self.assertGreater(len(contents), 1000)

    def test_numpy_manufactured_pdfs(self):
        self.check_manufactured_pdfs(use_mpmath=False)

    def test_mpmath_manufactured_pdfs(self):
        self.check_manufactured_pdfs(use_mpmath=True)

    def test_ipython_inline_png_without_rcparams_patch(self):
        # Isolate the inline backend and IPython singleton from the PDF tests.
        completed = subprocess.run(
            [sys.executable, "-B", "-c", INLINE_CHECK,
             str(self.output_directory / "inline")],
            cwd=Path(__file__).resolve().parent,
            text=True, capture_output=True, timeout=90,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
        self.assertGreater(json.loads(completed.stdout)["png_count"], 0)


if __name__ == "__main__":
    unittest.main()
