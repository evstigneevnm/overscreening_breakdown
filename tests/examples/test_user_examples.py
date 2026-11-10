"""Run catalog-based examples without executing the retained research sweeps."""

from contextlib import redirect_stdout
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from overscreening_breakdown import load_solution
from examples import collocation, galerkin, restart_and_projection, variable_epsilon
from examples._common import sampled_error


ROOT = Path(__file__).resolve().parents[2]


class UserExampleTests(unittest.TestCase):
    def test_bounded_examples_return_native_solutions_for_numpy_and_mpmath(self):
        for module in (collocation, galerkin, variable_epsilon):
            for use_mpmath in (False, True):
                with self.subTest(example=module.__name__, mpmath=use_mpmath):
                    solution = module.run(use_mpmath=use_mpmath, precision=50)
                    self.assertTrue(solution.converged)
                    self.assertTrue(solution.nonlinear_converged)
                    self.assertEqual(solution.quadrature_converged,
                                     True if solution.solver_type == "galerkin" else None)
                    arithmetic = solution.arithmetic
                    bound = arithmetic.scalar("1e-30" if use_mpmath else "1e-8")
                    self.assertLess(sampled_error(solution, module.make_case()), bound)
                    value = solution.u(arithmetic.scalar("0.375"))
                    self.assertIsInstance(value, type(arithmetic.scalar(0)))
                    if use_mpmath:
                        self.assertNotIsInstance(value, (float, np.floating))
        for use_mpmath in (False, True):
            with self.subTest(coefficient="jump", mpmath=use_mpmath):
                solution = variable_epsilon.run(coefficient="jump", use_mpmath=use_mpmath)
                self.assertTrue(solution.converged)
                self.assertEqual(solution.physical.epsilon.interfaces, (1,))
                # Newton and quadrature convergence do not certify a small
                # approximation error for the finite global transmission basis.
                self.assertTrue(solution.arithmetic.isfinite(
                    sampled_error(solution, variable_epsilon.make_case("jump"))))

    def test_typed_restart_and_explicit_projection_example_in_both_arithmetics(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath), tempfile.TemporaryDirectory(dir="/tmp") as directory:
                with redirect_stdout(io.StringIO()):
                    source, restored = restart_and_projection.run(
                        directory, use_mpmath=use_mpmath, precision=50)
                self.assertEqual(source.solver_type, "collocation")
                self.assertEqual(restored.solver_type, "galerkin")
                self.assertTrue(restored.nonlinear_converged)
                self.assertTrue(restored.quadrature_converged)
                for name in ("collocation", "projected", "galerkin"):
                    self.assertTrue((Path(directory)/(name+".json")).is_file())
                case = collocation.make_case()
                projected = load_solution(Path(directory)/"projected.json",
                                          problem=case.make_problem(), solver="galerkin")
                self.assertFalse(projected.converged)
                self.assertTrue(projected.projection_info)
                self.assertLess(sampled_error(restored, case), restored.arithmetic.scalar(
                    "1e-30" if use_mpmath else "1e-8"))

    def test_all_user_scripts_have_module_and_off_repository_help(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        for name in ("collocation", "galerkin", "variable_epsilon", "restart_and_projection"):
            with tempfile.TemporaryDirectory(dir="/tmp") as directory:
                commands = (([sys.executable, "-B", str(ROOT/"examples"/(name+".py")), "--help"], directory),
                            ([sys.executable, "-B", "-m", "examples."+name, "--help"], str(ROOT)))
                for command, cwd in commands:
                    with self.subTest(example=name, command=command):
                        completed = subprocess.run(command, cwd=cwd, env=environment,
                                                   capture_output=True, text=True, timeout=30)
                        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
                        self.assertIn("--output-dir", completed.stdout)
                        self.assertIn("--backend", completed.stdout)

    def test_off_repository_cli_writes_typed_snapshot_and_actual_figures(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["MPLBACKEND"] = "Agg"
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            output = Path(directory)/"results"
            environment["MPLCONFIGDIR"] = str(Path(directory)/"matplotlib")
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT/"examples"/"galerkin.py"), "--backend", "mpmath",
                 "--precision", "50", "--output-dir", str(output)],
                cwd=directory, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            self.assertIn("nonlinear_converged=True", completed.stdout)
            self.assertIn("quadrature_converged=True", completed.stdout)
            path = output/"galerkin-mpmath.json"
            restored = load_solution(path, problem=galerkin.make_case().make_problem(), solver="galerkin")
            self.assertTrue(restored.converged)
            self.assertEqual(restored.arithmetic.dps, 50)
            for suffix in ("fields", "coefficients"):
                data = (output/("galerkin-mpmath-"+suffix+".pdf")).read_bytes()
                self.assertTrue(data.startswith(b"%PDF-"))
                self.assertGreater(len(data), 1000)

    def test_notebooks_have_clean_outputs_canonical_imports_and_no_directory_magic(self):
        notebooks = sorted((ROOT/"examples"/"notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 4)
        self.assertFalse(list((ROOT/"examples").glob("*.ipynb")))
        for path in notebooks:
            notebook = json.loads(path.read_text())
            sources = []
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell.get("outputs"), [])
                    self.assertIsNone(cell.get("execution_count"))
                    code = "".join(cell.get("source", []))
                    ast.parse("\n".join(line for line in code.splitlines()
                                        if not line.lstrip().startswith("%")))
                    sources.append(code)
            source = "\n".join(sources)
            with self.subTest(notebook=path.name):
                self.assertNotIn("%cd", source)
                self.assertNotIn("RcParams._get", source)
                self.assertNotIn("from spectral1d import", source)
                self.assertIn("overscreening_breakdown", source)
        execution = json.loads((ROOT/"examples"/"notebooks"/"execution_example.ipynb").read_text())
        setup = next("".join(cell["source"]) for cell in execution["cells"]
                     if cell["cell_type"] == "code" and "def run_collocation(" in "".join(cell["source"]))
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            namespace = {}
            with patch.dict(os.environ, {"OVERSCREENING_OUTPUT_DIR": directory}):
                exec(compile("\n".join(line for line in setup.splitlines()
                                       if not line.lstrip().startswith("%")),
                             "execution_example.ipynb setup", "exec"), namespace)
            problem = collocation.make_case().make_problem()
            source = namespace["run_collocation"](problem, 8, L=1, tolerance="1e-35",
                                                   use_mpmath=True, precision=50)
            mixed = namespace["solve"](problem, 4, solver="galerkin", use_mpmath=True,
                                         precision=50, tolerance="1e-35")
            self.assertTrue(source.converged)
            self.assertTrue(mixed.nonlinear_converged and mixed.quadrature_converged)
            self.assertIsInstance(mixed.u(1), type(mixed.arithmetic.scalar(0)))
            path = Path(directory)/"bounded-notebook-smoke.json"
            mixed.save(path)
            restored = namespace["load_solution"](path, problem=problem, solver="galerkin")
            self.assertEqual(tuple(restored.coefficients), tuple(mixed.coefficients))


if __name__ == "__main__":
    unittest.main()
