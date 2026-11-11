"""Installed canonical API, lazy imports, and the breaking namespace cleanup."""

import importlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


PUBLIC_EXPORTS = {
    "BSKProblem": ("problems.bsk_problem", "BSKProblem"),
    "FluxProblem": ("problems.flux", "FluxProblem"),
    "BoundFluxProblem": ("problems.flux", "BoundFluxProblem"),
    "Coefficient": ("coefficients", "Coefficient"),
    "ConstantCoefficient": ("coefficients", "ConstantCoefficient"),
    "NumpyArithmetic": ("arithmetic", "NumpyArithmetic"),
    "MpmathArithmetic": ("arithmetic", "MpmathArithmetic"),
    "QuadratureFactory": ("quadrature", "QuadratureFactory"),
    "QuadraturePlan": ("quadrature", "QuadraturePlan"),
    "ProgressEvent": ("diagnostics", "ProgressEvent"),
    "ProgressCallbackError": ("diagnostics", "ProgressCallbackError"),
    "solve": ("solvers", "solve"),
    "load_solution": ("solvers", "load_solution"),
    "load_legacy_collocation": ("solvers", "load_legacy_collocation"),
    "project_collocation": ("projection", "project_collocation"),
    "save_solution": ("solution_io", "save_solution"),
    "CollocationSolution": ("collocation.solution", "CollocationSolution"),
    "GalerkinSolution": ("galerkin.solution", "GalerkinSolution"),
    "plot_fields": ("plotting", "plot_fields"),
    "plot_coefficients": ("plotting", "plot_coefficients"),
    "physical_l2_norm": ("norms", "physical_l2_norm"),
    "physical_h1_norm": ("norms", "physical_h1_norm"),
    "physical_l2_error": ("norms", "physical_l2_error"),
    "physical_h1_error": ("norms", "physical_h1_error"),
}

REMOVED_MODULES = (
    "arithmetic", "coefficients", "nonlinear_solver", "rational_mapping", "quadrature",
    "physical_problem", "spectral1d", "collocation_problem", "collocation_solution",
    "galerkin", "galerkin_assembly", "solutions", "solution_io", "projection",
    "solution_plotting", "solvers", "manufactured_problems", "manufactured_solutions_solver",
    "problems_for_updated_testing", "nonlinear_problem_on_segment",
)


class PackagingTests(unittest.TestCase):
    def test_public_exports_are_the_canonical_runtime_objects(self):
        package = importlib.import_module("overscreening_breakdown")
        for name, (module, attribute) in PUBLIC_EXPORTS.items():
            with self.subTest(name=name):
                canonical = importlib.import_module("overscreening_breakdown."+module)
                self.assertIs(getattr(package, name), getattr(canonical, attribute))
                self.assertIn(name, package.__all__)
                self.assertIn(name, dir(package))
        for removed in ("overscreening_breakdown", "overscreening_breakdown_segment"):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, package.__all__)
                with self.assertRaises(AttributeError):
                    getattr(package, removed)

    def test_importing_the_package_is_lazy(self):
        script = textwrap.dedent("""
            import sys
            import overscreening_breakdown as package
            for dependency in ('numpy', 'mpmath', 'scipy', 'matplotlib'):
                assert dependency not in sys.modules, dependency
            assert 'BSKProblem' in dir(package)
            _ = package.FluxProblem
            assert 'matplotlib.pyplot' not in sys.modules
            assert 'overscreening_breakdown.collocation.spectral1d' not in sys.modules
        """)
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd="/tmp",
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_distribution_does_not_install_the_removed_flat_modules(self):
        distribution = metadata.distribution("overscreening-breakdown")
        installed = {str(path) for path in distribution.files or ()}
        self.assertTrue(installed)
        for name in REMOVED_MODULES:
            with self.subTest(module=name):
                self.assertNotIn(name+".py", installed)
                # Other distributions can independently own a generic name.
                # This project must not supply it from either its src tree or
                # its canonical package directory.
                spec = importlib.util.find_spec(name)
                if spec is not None and spec.origin:
                    origin = Path(spec.origin).resolve()
                    source = Path(__file__).resolve().parents[2]/"src"
                    self.assertNotEqual(origin, source/(name+".py"))
                    self.assertNotIn("overscreening_breakdown", origin.parts)

    def test_distribution_metadata_is_installed_for_editable_and_wheel_modes(self):
        package = importlib.import_module("overscreening_breakdown")
        distribution = metadata.distribution("overscreening-breakdown")
        self.assertEqual(distribution.metadata["Name"].replace("_", "-").lower(),
                         "overscreening-breakdown")
        self.assertEqual(package.__version__, distribution.version)
        self.assertTrue(Path(distribution.locate_file("")).exists())

    def test_installed_public_api_solves_outside_the_repository(self):
        script = textwrap.dedent("""
            import json
            from pathlib import Path
            import mpmath as mp
            import overscreening_breakdown as package
            from overscreening_breakdown import FluxProblem, solve

            original_bits = mp.mp.prec
            problem = FluxProblem(
                delta='2',
                f=lambda x, u, a: 2/(1+x)**3-96/(1+x)**5+u-1/(1+x),
                f_u=lambda x, u, a: a.scalar(1),
                boundary_conditions=(('1', None, None, '-6', None), (None,)*5),
            )
            result = solve(problem, 4, solver='galerkin', use_mpmath=True,
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
