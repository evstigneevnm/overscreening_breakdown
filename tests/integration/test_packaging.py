"""Installed-package behavior and compatibility of historical import paths."""

import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch


MODULE_ALIASES = {
    "arithmetic": "arithmetic",
    "coefficients": "coefficients",
    "nonlinear_solver": "nonlinear_solver",
    "rational_mapping": "rational_mapping",
    "quadrature": "quadrature",
    "solvers": "solvers",
    "solution_io": "solution_io",
    "projection": "projection",
    "physical_problem": "problems.flux",
    "spectral1d": "collocation.spectral1d",
    "collocation_problem": "collocation.adapter",
    "collocation_solution": "collocation.solution",
    "galerkin": "galerkin.basis",
    "galerkin_assembly": "galerkin.assembly",
    "solutions": "galerkin.solution",
    "solution_plotting": "plotting",
    "manufactured_problems": "manufactured.legacy_problems",
    "manufactured_solutions_solver": "manufactured.legacy_runner",
    "problems_for_updated_testing": "manufactured.legacy_additional",
    "nonlinear_problem_on_segment": "problems.segment",
}


class PackagingTests(unittest.TestCase):
    def test_historical_modules_share_canonical_runtime_objects(self):
        package = importlib.import_module("overscreening_breakdown")
        package_directory = Path(package.__file__).resolve().parent
        for old_name, relative_name in MODULE_ALIASES.items():
            with self.subTest(module=old_name):
                historical = importlib.import_module(old_name)
                canonical = importlib.import_module("overscreening_breakdown."+relative_name)
                self.assertIs(historical, canonical)
                self.assertIn(package_directory, Path(canonical.__file__).resolve().parents)
        from overscreening_breakdown.problems.bsk import (
            overscreening_breakdown, overscreening_breakdown_segment,
        )
        self.assertIs(package.overscreening_breakdown, overscreening_breakdown)
        self.assertIs(package.overscreening_breakdown_segment, overscreening_breakdown_segment)

    def test_patching_historical_module_changes_the_canonical_solver(self):
        historical = importlib.import_module("spectral1d")
        canonical = importlib.import_module("overscreening_breakdown.collocation.spectral1d")
        original = canonical.solve_newton
        replacement = object()
        with patch("spectral1d.solve_newton", replacement):
            self.assertIs(canonical.solve_newton, replacement)
            self.assertIs(historical.solve_nonlinear_problem, canonical.solve_nonlinear_problem)
        self.assertIs(canonical.solve_newton, original)

    def test_historical_pickle_global_names_still_resolve(self):
        # Existing solution files contain numerical data. Also preserve class
        # lookup for user-created trusted pickles using the old module names.
        names = (
            ("spectral1d", "basis_functions", "collocation.spectral1d"),
            ("spectral1d", "solve_nonlinear_problem", "collocation.spectral1d"),
            ("overscreening_breakdown", "overscreening_breakdown", "problems.bsk"),
            ("overscreening_breakdown", "overscreening_breakdown_segment", "problems.bsk"),
            ("manufactured_problems", "problem_1", "manufactured.legacy_problems"),
            ("nonlinear_problem_on_segment", "nonlinear_problem_on_segment", "problems.segment"),
        )
        for old_module, name, new_module in names:
            with self.subTest(module=old_module, name=name):
                payload = "c{}\n{}\n.".format(old_module, name).encode("ascii")
                expected = getattr(importlib.import_module("overscreening_breakdown."+new_module), name)
                self.assertIs(pickle.loads(payload), expected)

    def test_distribution_metadata_is_installed_for_editable_and_wheel_modes(self):
        distribution = metadata.distribution("overscreening-breakdown")
        self.assertEqual(distribution.metadata["Name"].replace("_", "-").lower(),
                         "overscreening-breakdown")
        self.assertEqual(metadata.version("overscreening-breakdown"), distribution.version)
        self.assertTrue(distribution.version)
        self.assertTrue(Path(distribution.locate_file("")).exists())

    def test_installed_package_solves_outside_the_repository(self):
        script = textwrap.dedent("""
            import importlib
            import json
            from pathlib import Path
            import mpmath as mp
            import overscreening_breakdown as package
            import physical_problem
            import solvers

            assert physical_problem is importlib.import_module('overscreening_breakdown.problems.flux')
            assert solvers is importlib.import_module('overscreening_breakdown.solvers')
            original_bits = mp.mp.prec
            problem = physical_problem.FluxProblem(
                delta='2',
                f=lambda x, u, a: 2/(1+x)**3-96/(1+x)**5+u-1/(1+x),
                f_u=lambda x, u, a: a.scalar(1),
                boundary_conditions=(('1', None, None, '-6', None), (None,)*5),
            )
            result = solvers.solve(problem, 4, solver='galerkin', use_mpmath=True,
                                   precision=30, tolerance='1e-20', use_globalization=False)
            assert result.converged and result.newton_result.homotopy == 0
            assert abs(result.u(1)-result.arithmetic.scalar('0.5')) < result.arithmetic.scalar('1e-20')
            assert abs(result.w(1)+result.arithmetic.scalar('0.25')) < result.arithmetic.scalar('1e-20')
            assert mp.mp.prec == original_bits
            print(json.dumps({'backend': mp.libmp.BACKEND, 'package_file': package.__file__,
                              'working_directory': str(Path.cwd())}))
        """)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["MPLBACKEND"] = "Agg"
        import mpmath
        with tempfile.TemporaryDirectory(prefix="overscreening-package-", dir="/tmp") as directory:
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script], cwd=directory, env=environment,
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["backend"], mpmath.libmp.BACKEND)
            self.assertEqual(Path(report["working_directory"]), Path(directory))
            self.assertNotIn(Path(directory), Path(report["package_file"]).resolve().parents)


if __name__ == "__main__":
    unittest.main()
