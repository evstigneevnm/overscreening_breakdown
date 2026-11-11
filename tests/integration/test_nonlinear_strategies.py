"""Public initialization/recovery wiring with native numerical diagnostics."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np

from overscreening_breakdown import load_solution, solve
from overscreening_breakdown.manufactured import rational_case
from overscreening_breakdown.nonlinear_solver import LinearSolveError, solve_newton


class NonlinearStrategyTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)
        # Check restoration of an exact, noncanonical global precision.
        mp.mp.prec = 77

    def problem(self, coefficient="constant"):
        return rational_case(coefficient, nonlinear=True, beta="0.1", initial="exp").make_problem()

    def options(self, solver, use_mpmath=False, precision=30):
        return dict(solver=solver, use_mpmath=use_mpmath, precision=precision,
                    tolerance="1e-{}".format(precision-12) if use_mpmath else "1e-10",
                    use_globalization=False)

    def fail_first_linear_solve(self):
        """Leave the real system, gradient, descent, and subsequent solves intact."""
        observed = {"failures": 0}

        def wrapped(system, **options):
            original = system.solve_linear

            def first_failure(matrix, rhs):
                if observed["failures"] == 0:
                    observed["failures"] += 1
                    raise LinearSolveError("forced first linear solve failure")
                return original(matrix, rhs)

            return solve_newton(replace(system, solve_linear=first_failure), **options)

        return wrapped, observed

    def patch_target(self, solver):
        return ("overscreening_breakdown.collocation.spectral1d.solve_newton"
                if solver == "collocation" else "overscreening_breakdown.solvers.solve_newton")

    def assert_native_diagnostics(self, solution):
        arithmetic = solution.arithmetic
        scalar_type = type(arithmetic.scalar(0))
        result = solution.newton_result
        values = list(result.state)+[result.homotopy, result.residual_norm]
        for stage in result.stages:
            values.extend((stage.homotopy, stage.initial_norm, stage.target_norm, stage.final_norm))
        for record in result.history:
            values.extend((record.homotopy, record.residual_norm, record.step))
            values.extend(value for value in (record.merit, record.gradient_norm) if value is not None)
        for value in values:
            self.assertIsInstance(value, scalar_type)
            if arithmetic.use_mpmath:
                self.assertNotIsInstance(value, (float, np.floating))

    def test_fsolve_initialization_works_with_every_numpy_collocation_method(self):
        for method in ("newton", "root", "root_no_jac", "nonlin_solve"):
            with self.subTest(method=method):
                result = solve(self.problem(), 6, method=method, initializer="fsolve",
                               initializer_max_evaluations=40, use_globalization=False)
                self.assertTrue(result.converged)
                initialization = result.initialization_result
                self.assertEqual(initialization.method, "fsolve")
                self.assertLessEqual(initialization.evaluations, 40)
                self.assertEqual(initialization.evaluations,
                                 initialization.function_evaluations+initialization.jacobian_evaluations)
                self.assertTrue(initialization.accepted)
                self.assertLess(initialization.final_norm, initialization.initial_norm)
                self.assertIsInstance(initialization.state, np.ndarray)
                self.assertEqual(initialization.state.dtype, np.float64)
                self.assertLess(abs(result.u(1)-0.5), 1e-8)
                self.assertEqual(mp.mp.prec, 77)

    def test_initializer_budget_and_scipy_status_do_not_claim_final_convergence(self):
        with patch("scipy.optimize.fsolve") as optimizer:
            limited = solve(self.problem(), 4, initializer="fsolve", initializer_max_evaluations=1,
                            max_iterations=0, use_globalization=False)
            optimizer.assert_not_called()
        self.assertFalse(limited.converged)
        self.assertFalse(limited.initialization_result.accepted)
        self.assertEqual(limited.initialization_result.evaluations, 1)
        self.assertEqual(limited.initialization_result.reason, "evaluation_limit")
        self.assertEqual(limited.newton_result.reason, "max_iterations")
        self.assertGreater(limited.newton_result.residual_norm, 0)

        def reported_success(function, initial, **options):
            # Returning an untouched state with status=1 cannot certify the PDE.
            return initial.copy(), {}, 1, "reported step convergence"

        with patch("scipy.optimize.fsolve", side_effect=reported_success):
            reported = solve(self.problem(), 4, initializer="fsolve", max_iterations=0,
                             use_globalization=False)
        self.assertTrue(reported.initialization_result.scipy_success)
        self.assertFalse(reported.initialization_result.accepted)
        self.assertFalse(reported.nonlinear_converged)
        self.assertFalse(reported.converged)
        self.assertEqual(mp.mp.prec, 77)

    def test_unsupported_initializer_and_scipy_recovery_requests_are_explicit_errors(self):
        requests = [dict(initializer="fsolve", use_mpmath=True, precision=120),
                    dict(initializer="fsolve", solver="galerkin"),
                    dict(initializer="fsolve", solver="galerkin", use_mpmath=True)]
        requests += [dict(method=method, newton_fallback="adjoint")
                     for method in ("root", "root_no_jac", "nonlin_solve")]
        with patch("scipy.optimize.fsolve") as optimizer:
            for options in requests:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    solve(self.problem(), 4, **options)
                self.assertEqual(mp.mp.prec, 77)
            optimizer.assert_not_called()

    def test_public_native_adjoint_recovers_after_a_real_linear_failure(self):
        for solver in ("collocation", "galerkin"):
            for use_mpmath, precision in ((False, 15), (True, 30), (True, 120)):
                with self.subTest(solver=solver, mpmath=use_mpmath, precision=precision):
                    options = self.options(solver, use_mpmath, precision)
                    resolution = 6 if solver == "collocation" else 4
                    wrapper, observed = self.fail_first_linear_solve()
                    with patch(self.patch_target(solver), side_effect=wrapper):
                        disabled = solve(self.problem(), resolution, **options)
                    self.assertEqual(observed["failures"], 1)
                    self.assertFalse(disabled.converged)
                    self.assertFalse(disabled.nonlinear_converged)
                    self.assertEqual(disabled.newton_result.reason, "linear_solve_failed")
                    self.assertEqual(disabled.newton_result.adjoint_steps, 0)
                    self.assertEqual(mp.mp.prec, 77)

                    wrapper, observed = self.fail_first_linear_solve()
                    with patch(self.patch_target(solver), side_effect=wrapper):
                        recovered = solve(self.problem(), resolution, newton_fallback="adjoint", **options)
                    self.assertEqual(observed["failures"], 1)
                    self.assertTrue(recovered.converged)
                    result = recovered.newton_result
                    self.assertEqual(result.homotopy, 0)
                    self.assertGreaterEqual(result.adjoint_steps, 1)
                    self.assertGreaterEqual(result.adjoint_evaluations, result.adjoint_steps)
                    self.assertEqual(result.adjoint_steps, sum(stage.adjoint_steps for stage in result.stages))
                    self.assertIn("linear_solve_failed", result.recovery_reasons)
                    recovery_records = [record for record in result.history if record.kind == "adjoint"]
                    self.assertEqual(len(recovery_records), result.adjoint_steps)
                    self.assertTrue(all(record.recovery_reason == "linear_solve_failed"
                                        for record in recovery_records))
                    self.assertTrue(all(record.homotopy == 0 for record in result.history))
                    self.assert_native_diagnostics(recovered)
                    arithmetic = recovered.arithmetic
                    error_bound = arithmetic.scalar(
                        "1e-{}".format(precision-15) if use_mpmath else "1e-8")
                    self.assertLess(abs(recovered.u(1)-arithmetic.scalar("0.5")), error_bound)
                    self.assertEqual(mp.mp.prec, 77)

    def test_galerkin_retains_recovery_diagnostics_on_each_refined_plan(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                wrapper, observed = self.fail_first_linear_solve()
                with patch(self.patch_target("galerkin"), side_effect=wrapper):
                    result = solve(self.problem("exponential"), 4,
                                   newton_fallback="adjoint", quadrature_order=5,
                                   quadrature_atol="1e-10", quadrature_rtol="1e-10",
                                   **self.options("galerkin", use_mpmath, 30))
                self.assertEqual(observed["failures"], 1)
                self.assertTrue(result.converged)
                self.assertGreater(len(result.quadrature_history), 1)
                self.assertGreaterEqual(result.quadrature_history[0].newton_result.adjoint_steps, 1)
                self.assertFalse(result.quadrature_history[0].quadrature_converged)
                self.assertIs(result.quadrature_history[-1].newton_result, result.newton_result)
                self.assertTrue(all(item.newton_result.converged and item.newton_result.homotopy == 0
                                    for item in result.quadrature_history))
                self.assertEqual(result.plan.order, result.quadrature_history[-1].order)
                self.assert_native_diagnostics(result)
                self.assertEqual(mp.mp.prec, 77)

    def test_default_controls_and_runtime_only_diagnostics_preserve_saved_format(self):
        for solver in ("collocation", "galerkin"):
            with self.subTest(solver=solver):
                problem = self.problem()
                default = solve(problem, 4, **self.options(solver))
                explicit = solve(problem, 4, initializer=None, newton_fallback=None,
                                 **self.options(solver))
                self.assertTrue(default.converged and explicit.converged)
                self.assertEqual(tuple(default.coefficients), tuple(explicit.coefficients))
                self.assertIsNone(default.initialization_result)
                self.assertEqual(default.newton_result.adjoint_steps, 0)
                self.assertTrue(all(record.kind == "newton" for record in default.newton_result.history))
                if solver == "collocation":
                    result = solve(problem, 4, initializer="fsolve", **self.options(solver))
                    self.assertIsNotNone(result.initialization_result)
                else:
                    wrapper, _ = self.fail_first_linear_solve()
                    with patch(self.patch_target(solver), side_effect=wrapper):
                        result = solve(problem, 4, newton_fallback="adjoint", **self.options(solver))
                    self.assertGreaterEqual(result.newton_result.adjoint_steps, 1)
                self.assertTrue(result.converged)
                with tempfile.TemporaryDirectory(dir="/tmp") as directory:
                    path = Path(directory)/"runtime-diagnostics.json"
                    result.save(path)
                    serialized = json.dumps(json.loads(path.read_text()))
                    self.assertNotIn("initialization_result", serialized)
                    self.assertNotIn("newton_result", serialized)
                    restored = load_solution(path, problem=problem, solver=solver)
                self.assertTrue(restored.converged)
                self.assertEqual(tuple(result.coefficients), tuple(restored.coefficients))
                self.assertIsNone(restored.initialization_result)
                self.assertIsNone(restored.newton_result)
                self.assertEqual(restored.quadrature_history, ())
                self.assertEqual(mp.mp.prec, 77)


if __name__ == "__main__":
    unittest.main()
