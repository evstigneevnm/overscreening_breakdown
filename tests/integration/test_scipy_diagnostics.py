"""Public SciPy diagnostics and a verified root despite stagnation status."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy import optimize

from overscreening_breakdown import load_solution, solve
from overscreening_breakdown.manufactured import rational_case


class ScipyDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)
        mp.mp.prec = 77

    def problem(self):
        return rational_case(nonlinear=True, beta="0.1", initial="exp").make_problem()

    def test_all_scipy_methods_return_native_verified_diagnostics(self):
        for method in ("root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                result = solve(self.problem(), 6, method=method, tolerance="1e-10",
                               use_globalization=False)
                self.assertTrue(result.converged)
                self.assertIsNone(result.newton_result)
                diagnostic = result.scipy_result
                self.assertEqual(diagnostic.method, method)
                self.assertTrue(diagnostic.converged)
                self.assertEqual(diagnostic.homotopy, 0)
                self.assertLessEqual(diagnostic.residual_norm, diagnostic.target_norm)
                self.assertGreater(diagnostic.nfev, 0)
                self.assertIsInstance(diagnostic.status, int)
                self.assertIsInstance(diagnostic.message, str)
                for value in (diagnostic.residual_norm, diagnostic.residual_max,
                              diagnostic.target_norm, diagnostic.homotopy):
                    self.assertIsInstance(value, np.float64)
                self.assertLess(abs(result.u(1)-0.5), 1e-8)
                self.assertEqual(mp.mp.prec, 77)

    def test_stagnation_status_preserves_the_full_manufactured_accuracy_checks(self):
        from tests.manufactured.test_manufactured_nonlinear_light import (
            NonlinearManufacturedLightTests,
        )

        actual_root = optimize.root
        solutions = []

        def stagnated(*args, **kwargs):
            # Keep the actual spatial problem and actual SciPy iterate. Only
            # the termination report changes, reproducing the status class
            # that must not override a freshly verified discrete residual.
            result = actual_root(*args, **kwargs)
            result.success = False
            result.status = 5
            result.message = "iteration is not making good progress"
            return result

        def observe_solution(*args, **kwargs):
            result = solve(*args, **kwargs)
            solutions.append(result)
            return result

        fixture = NonlinearManufacturedLightTests(
            "test_fourth_order_accuracy_for_every_method_and_precision")
        with patch("overscreening_breakdown.collocation.spectral1d.optimize.root",
                   side_effect=stagnated), patch(
                       "tests.manufactured.test_manufactured_nonlinear_light.solve",
                       side_effect=observe_solution):
            # Reuse every existing boundary, independent PDE, sampled-field,
            # mapped-L2 and precision-restoration assertion without loosening
            # any bound. This is the same N=24 fixture that failed on GitHub.
            fixture.check_case(2, 24, "root_no_jac", False, 15)
        self.assertEqual(len(solutions), 1)
        diagnostic = solutions[0].scipy_result
        self.assertFalse(diagnostic.scipy_success)
        self.assertEqual(diagnostic.status, 5)
        self.assertIn("not making good progress", diagnostic.message)
        self.assertTrue(diagnostic.converged)
        self.assertLessEqual(diagnostic.residual_norm, diagnostic.target_norm)
        self.assertEqual(mp.mp.prec, 77)

    def test_runtime_diagnostics_are_distinct_from_initializer_and_saved_data(self):
        problem = self.problem()
        result = solve(problem, 6, method="root_no_jac", initializer="fsolve",
                       tolerance="1e-10", use_globalization=False)
        self.assertTrue(result.converged)
        self.assertIsNotNone(result.initialization_result)
        self.assertIsNotNone(result.scipy_result)
        self.assertIsNone(result.newton_result)
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory)/"scipy-solution.json"
            result.save(path)
            contents = json.loads(path.read_text())
            self.assertNotIn("scipy_result", contents)
            restored = load_solution(path, problem=problem, solver="collocation")
        self.assertTrue(restored.converged)
        self.assertEqual(restored.method, "root_no_jac")
        self.assertEqual(tuple(restored.coefficients), tuple(result.coefficients))
        self.assertIsNone(restored.scipy_result)
        self.assertIsNone(restored.initialization_result)
        self.assertEqual(mp.mp.prec, 77)

    def test_newton_and_galerkin_do_not_fabricate_scipy_results(self):
        for solver, resolution in (("collocation", 6), ("galerkin", 4)):
            for use_mpmath in (False, True):
                with self.subTest(solver=solver, mpmath=use_mpmath):
                    result = solve(self.problem(), resolution, solver=solver,
                                   use_mpmath=use_mpmath, precision=30,
                                   tolerance="1e-18" if use_mpmath else "1e-10",
                                   use_globalization=False)
                    self.assertTrue(result.converged)
                    self.assertIsNone(result.scipy_result)
                    self.assertEqual(mp.mp.prec, 77)


if __name__ == "__main__":
    unittest.main()
