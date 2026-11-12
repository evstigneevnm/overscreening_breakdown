"""Shared physical sources recover through native references in both methods."""

from dataclasses import dataclass, replace
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
import pytest

from overscreening_breakdown import (
    BSKProblem, Coefficient, ContinuationOptions, FluxProblem, ScreenedLinearReference, solve,
)
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.manufactured import rational_case
from overscreening_breakdown.nonlinear_solver import LinearSolveError, solve_newton


@dataclass(frozen=True)
class UserSource:
    """Ordinary user callable with an independently manufactured exact field."""

    beta: str = "0.125"
    smooth_epsilon: bool = False

    def __call__(self, x, u, arithmetic):
        r = arithmetic.scalar(1)/(1+x)
        manufactured = 2*r**3-96*r**5
        if self.smooth_epsilon:
            manufactured += 3*r**4-120*r**6
        return manufactured+(u-r)+arithmetic.scalar(self.beta)*(u**3-r**3)

    def derivative(self, x, u, arithmetic):
        return arithmetic.scalar(1)+3*arithmetic.scalar(self.beta)*u*u


class ContinuationSolverTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)
        mp.mp.prec = 77

    def configurations(self):
        yield False, 15
        for digits in (15, 30, 50, 100, 120):
            yield True, digits

    def tolerance(self, use_mpmath, digits):
        return "1e-{}".format(max(10, digits-8)) if use_mpmath else "1e-11"

    def problem(self, smooth=False):
        source = UserSource(smooth_epsilon=smooth)
        coefficient = (Coefficient(lambda x, a: 1+a.scalar(1)/(1+x),
                                   lower_bound=1, upper_bound=2, wall_value=2)
                       if smooth else 1)
        return FluxProblem(
            delta="2", epsilon=coefficient, f=source, f_u=source.derivative,
            boundary_conditions=((1, None, None, -6, None),
                                 (0, None, None, None, None)),
            initial_guess=lambda x, a: a.exp(-x*x),
        )

    def fail_initial_physical_stage(self, every_refinement=False):
        observed = {"calls": 0, "failures": 0, "parameters": []}

        def wrapped(system, **options):
            observed["calls"] += 1
            observed["parameters"].append(options.get("parameter", system.scalar(0)))
            initial = observed["calls"] == 1
            if every_refinement:
                initial = options.get("stage", 1) == 1
            if initial:
                def fail(matrix, rhs):
                    observed["failures"] += 1
                    raise LinearSolveError("exercise reference recovery")
                system = replace(system, solve_linear=fail)
            return solve_newton(system, **options)

        return wrapped, observed

    def assert_reference_recovery(self, solution):
        result = solution.newton_result
        arithmetic = solution.arithmetic
        scalar_type = type(arithmetic.scalar(0))
        self.assertTrue(solution.converged)
        self.assertTrue(result.converged)
        self.assertEqual(result.homotopy, 0)
        self.assertIsNotNone(result.reference_result)
        self.assertTrue(result.reference_result.converged)
        self.assertLessEqual(result.reference_result.residual_norm,
                             result.reference_result.target_norm)
        self.assertEqual(result.stages[0].reason, "linear_solve_failed")
        self.assertTrue(any(stage.homotopy > 0 for stage in result.stages))
        self.assertEqual(result.stages[-1].homotopy, 0)
        self.assertTrue(result.stages[-1].converged)
        self.assertGreater(result.continuation_attempts, 0)
        self.assertEqual(solution.initialization_result, None)
        for value in (result.homotopy, result.residual_norm,
                      result.reference_result.residual_norm,
                      result.reference_result.strength,
                      *tuple(result.reference_result.state)):
            self.assertIsInstance(value, scalar_type)
            if arithmetic.use_mpmath:
                self.assertNotIsInstance(value, (float, np.floating))
        self.assertEqual(solution.physical.delta, 2)
        self.assertEqual(solution.physical.boundary_conditions,
                         ((1, None, None, -6, None), (0, None, None, None, None)))
        self.assertEqual(mp.mp.prec, 77)

    def assert_rational_fields(self, solution, tolerance):
        arithmetic = solution.arithmetic
        bound = 10000*arithmetic.scalar(tolerance)
        for text in ("0", "0.125", "0.75", "1", "3", "12"):
            x = arithmetic.scalar(text)
            exact = arithmetic.scalar(1)/(1+x)
            self.assertLess(abs(solution.u(x)-exact), bound)
            self.assertLess(abs(solution.w(x)+2*exact**3), bound)
        self.assertLess(abs(solution.u(arithmetic.inf)), bound)
        if solution.solver_type == "galerkin":
            self.assertTrue(solution.quadrature_converged)
            self.assertEqual(solution.u(arithmetic.inf), 0)
            self.assertEqual(solution.w(arithmetic.inf), 0)

    def test_user_callable_recovers_both_methods_at_every_supported_precision(self):
        problem = self.problem()
        self.assertIsInstance(problem.f, UserSource)
        self.assertFalse(hasattr(problem.f, "bind"))
        for solver, N in (("collocation", 6), ("galerkin", 4)):
            for use_mpmath, digits in self.configurations():
                with self.subTest(solver=solver, mpmath=use_mpmath, precision=digits):
                    wrapped, observed = self.fail_initial_physical_stage()
                    tolerance = self.tolerance(use_mpmath, digits)
                    with patch("overscreening_breakdown.continuation.solve_newton",
                               side_effect=wrapped):
                        result = solve(problem, N, solver=solver,
                                       use_mpmath=use_mpmath, precision=digits,
                                       tolerance=tolerance, max_iterations=20)
                    self.assertEqual(observed["failures"], 1)
                    self.assert_reference_recovery(result)
                    self.assert_rational_fields(result, tolerance)
                    self.assertTrue(result.physical.epsilon.is_identity)
                    # The same immutable physical callback remains reusable.
                    self.assertIsInstance(problem.f, UserSource)
                    self.assertEqual(problem.f.beta, "0.125")

    def test_variable_coefficient_recovery_retains_the_original_physical_problem(self):
        problem = self.problem(smooth=True)
        for use_mpmath, digits in self.configurations():
            with self.subTest(mpmath=use_mpmath, precision=digits):
                wrapped, observed = self.fail_initial_physical_stage()
                tolerance = self.tolerance(use_mpmath, digits)
                with patch("overscreening_breakdown.continuation.solve_newton",
                           side_effect=wrapped):
                    result = solve(problem, 4, solver="galerkin", use_mpmath=use_mpmath,
                                   precision=digits, tolerance=tolerance, max_iterations=20)
                self.assertEqual(observed["failures"], 1)
                self.assert_reference_recovery(result)
                self.assert_rational_fields(result, tolerance)
                a = result.arithmetic
                self.assertFalse(result.physical.epsilon.is_identity)
                self.assertEqual(result.physical.epsilon.wall_value, 2)
                self.assertEqual(result.physical.epsilon(a.scalar(1)), a.scalar("1.5"))

    def test_custom_reference_strength_reaches_the_same_physical_solution(self):
        for solver, N in (("collocation", 6), ("galerkin", 4)):
            with self.subTest(solver=solver):
                wrapped, observed = self.fail_initial_physical_stage()
                options = ContinuationOptions(reference=ScreenedLinearReference("0.375"))
                with patch("overscreening_breakdown.continuation.solve_newton",
                           side_effect=wrapped):
                    result = solve(self.problem(), N, solver=solver,
                                   use_mpmath=True, precision=50, tolerance="1e-35",
                                   continuation=options, max_iterations=20)
                self.assertEqual(observed["failures"], 1)
                self.assert_reference_recovery(result)
                self.assertEqual(result.newton_result.reference_result.strength,
                                 result.arithmetic.scalar("0.375"))
                self.assert_rational_fields(result, "1e-35")

    def test_quadrature_refinement_rebuilds_a_verified_reference_on_each_used_plan(self):
        case = rational_case("exponential", nonlinear=True)
        problem = replace(case.make_problem(), boundary_conditions=(
            (1, None, None, -6, None), (0, None, None, None, None)))
        native_reference = GalerkinAssembly.reference_operator
        for use_mpmath, digits in ((False, 15), (True, 30)):
            with self.subTest(mpmath=use_mpmath, precision=digits):
                plans = []

                def reference(assembly, strength="1"):
                    plans.append(assembly.plan)
                    return native_reference(assembly, strength)

                wrapped, observed = self.fail_initial_physical_stage(every_refinement=True)
                tolerance = self.tolerance(use_mpmath, digits)
                with patch.object(GalerkinAssembly, "reference_operator", reference), patch(
                    "overscreening_breakdown.continuation.solve_newton", side_effect=wrapped
                ):
                    result = solve(
                        problem, 4, solver="galerkin", use_mpmath=use_mpmath,
                        precision=digits, tolerance=tolerance,
                        quadrature_order=5, quadrature_atol="1e-9",
                        quadrature_rtol="1e-9", max_refinements=3,
                    )
                self.assertTrue(result.converged)
                self.assertTrue(result.quadrature_converged)
                self.assertGreater(len(result.quadrature_history), 1)
                self.assertEqual(len(plans), len(result.quadrature_history))
                self.assertEqual(observed["failures"], len(plans))
                self.assertEqual([plan.order for plan in plans],
                                 [item.order for item in result.quadrature_history])
                self.assertGreater(plans[-1].order, plans[0].order)
                self.assertEqual(result.plan, plans[-1])
                for item in result.quadrature_history:
                    self.assertTrue(item.newton_result.reference_result.converged)
                    self.assertTrue(any(stage.homotopy > 0
                                        for stage in item.newton_result.stages))
                for text in ("0.125", "1", "5"):
                    x = result.arithmetic.scalar(text)
                    self.assertLess(abs(result.u(x)-case.u(x, result.arithmetic)),
                                    result.arithmetic.scalar("1e-7"))
                self.assertEqual(mp.mp.prec, 77)


@pytest.mark.slow
class StiffBSKContinuationTests(unittest.TestCase):
    """Opt-in original notebook recovery; spatial accuracy is a separate study."""

    def test_original_unit_gaussian_recovers_for_both_methods(self):
        problem = BSKProblem(
            sigma=10, mu=1, gamma="0.5", delta=50,
            boundary_conditions=((100, None, None, 0, None),
                                 (0, None, None, None, None)),
            initial_guess=lambda x, a: a.exp(-x*x),
        )
        for solver in ("collocation", "galerkin"):
            with self.subTest(solver=solver):
                options = (dict(quadrature_order=128, subdivisions=4,
                                quadrature_atol="1e-8", quadrature_rtol="1e-8",
                                max_refinements=3) if solver == "galerkin" else {})
                result = solve(problem, 100, solver=solver, L=6, tolerance="1e-10",
                               max_iterations=20, initializer=None, **options)
                self.assertTrue(result.converged)
                self.assertTrue(result.nonlinear_converged)
                self.assertEqual(result.newton_result.homotopy, 0)
                reference = result.newton_result.reference_result
                if solver == "collocation":
                    self.assertIsNotNone(reference)
                if reference is not None:
                    self.assertTrue(reference.converged)
                self.assertLess(abs(result.u(result.arithmetic.inf)), 1e-8)
                if solver == "galerkin":
                    self.assertTrue(result.quadrature_converged)
