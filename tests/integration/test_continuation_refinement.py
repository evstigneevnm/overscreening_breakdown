"""Continuation references and budgets belong to the complete Galerkin solve."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from overscreening_breakdown import solve
from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.continuation import (
    AffineReference, ContinuationOptions, ResidualHomotopy,
)
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.manufactured import rational_case
from overscreening_breakdown.nonlinear_solver import LinearSolveError, solve_newton
from overscreening_breakdown.quadrature import QuadratureFactory


class ContinuationRefinementTests(unittest.TestCase):
    def problem(self, coefficient="exponential"):
        return rational_case(coefficient, nonlinear=True, initial="exp").make_problem()

    def options(self, use_mpmath):
        return dict(solver="galerkin", use_mpmath=use_mpmath, precision=30,
                    tolerance="1e-18" if use_mpmath else "1e-12",
                    quadrature_order=5, quadrature_atol="1e-10", quadrature_rtol="1e-10")

    def failures_per_plan(self):
        injected = set()

        def wrapped(system, **options):
            refinement = options["progress"]._context["refinement"]
            physical_attempt = options.get("parameter", 0) == 0 and options.get("stage", 1) == 1
            native_solve = system.solve_linear

            def fail_once(matrix, rhs):
                if physical_attempt and refinement not in injected:
                    injected.add(refinement)
                    raise LinearSolveError("forced first physical linear solve on this plan")
                return native_solve(matrix, rhs)

            return solve_newton(replace(system, solve_linear=fail_once), **options)

        return wrapped, injected

    def matvec(self, arithmetic, matrix, vector):
        return matrix*vector if arithmetic.use_mpmath else matrix @ vector

    def test_each_refined_plan_builds_and_verifies_its_own_reference(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                wrapped, injected = self.failures_per_plan()
                references = []
                native_reference = GalerkinAssembly.reference_operator

                def reference(assembly, strength="1"):
                    matrix, offset = native_reference(assembly, strength)
                    references.append((assembly, matrix, offset))
                    return matrix, offset

                with patch("overscreening_breakdown.continuation.solve_newton", side_effect=wrapped), \
                        patch.object(GalerkinAssembly, "reference_operator", reference):
                    result = solve(self.problem(), 4, **self.options(use_mpmath))
                self.assertTrue(result.converged)
                history = result.quadrature_history
                self.assertGreater(len(history), 1)
                self.assertEqual(injected, set(range(len(history))))
                self.assertEqual([assembly.plan.order for assembly, _, _ in references],
                                 [item.order for item in history])
                arithmetic = result.arithmetic
                for index, ((assembly, matrix, offset), item) in enumerate(zip(references, history)):
                    reference_result = item.newton_result.reference_result
                    self.assertTrue(reference_result.converged)
                    self.assertEqual(item.newton_result.stages[0].reason, "linear_solve_failed")
                    self.assertTrue(item.newton_result.converged)
                    self.assertEqual(item.newton_result.homotopy, 0)
                    self.assertIs(assembly.physical, result.physical)
                    residual = self.matvec(arithmetic, matrix, reference_result.state)+offset
                    self.assertLessEqual(arithmetic.norm(residual), reference_result.target_norm)
                    if index:
                        self.assertIsNot(assembly.plan, references[index-1][0].plan)
                        self.assertIsNot(matrix, references[index-1][1])
                        self.assertIsNot(reference_result.state,
                                         history[index-1].newton_result.reference_result.state)
                self.assertIs(history[-1].newton_result, result.newton_result)
                self.assertEqual(result.plan.order, history[-1].order)

    def test_newton_iteration_budget_is_shared_across_quadrature_plans(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                budget = 5
                result = solve(self.problem(), 4,
                               continuation=ContinuationOptions(max_total_iterations=budget),
                               **self.options(use_mpmath))
                self.assertFalse(result.converged)
                history = result.quadrature_history
                self.assertGreater(len(history), 1)
                self.assertEqual(sum(item.newton_result.iterations for item in history), budget)
                self.assertEqual(history[-1].reason, "max_total_iterations")
                self.assertTrue(all(item.newton_result.converged for item in history[:-1]))
                self.assertFalse(history[-1].nonlinear_converged)
                if not use_mpmath:
                    self.assertEqual(history[-1].newton_result.iterations, 0)

    def test_continuation_stage_budget_is_shared_across_quadrature_plans(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                wrapper, injected = self.failures_per_plan()
                budget = 4
                with patch("overscreening_breakdown.continuation.solve_newton", side_effect=wrapper):
                    result = solve(self.problem(), 4,
                                   continuation=ContinuationOptions(max_stages=budget),
                                   **self.options(use_mpmath))
                self.assertFalse(result.converged)
                history = result.quadrature_history
                self.assertGreater(len(history), 1)
                self.assertEqual(sum(item.newton_result.continuation_attempts for item in history), budget)
                self.assertEqual(history[-1].reason, "max_stages")
                self.assertTrue(history[0].newton_result.converged)
                self.assertTrue(history[-1].newton_result.reference_result.converged)
                self.assertGreater(history[-1].newton_result.homotopy, 0)
                self.assertEqual(injected, set(range(len(history))))

    def test_zero_remaining_budget_can_verify_an_exact_restart_without_reference(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                problem = self.problem("constant")
                options = self.options(use_mpmath)
                options["quadrature_order"] = 8
                first = solve(problem, 4, **options)
                self.assertTrue(first.converged)
                with patch.object(GalerkinAssembly, "reference_operator",
                                  side_effect=AssertionError("an exact restart needs no reference")):
                    result = solve(
                        problem, 4, initial_guess=first,
                        continuation=ContinuationOptions(max_total_iterations=0, max_stages=0),
                        **options,
                    )
                self.assertTrue(result.converged)
                self.assertTrue(result.quadrature_converged)
                self.assertEqual(result.newton_result.iterations, 0)
                self.assertEqual(result.newton_result.homotopy, 0)
                self.assertIsNone(result.newton_result.reference_result)

    def test_adjoint_merit_uses_the_full_intermediate_mixed_residual(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            with self.subTest(backend=arithmetic.backend):
                basis = GalerkinBasis(4, arithmetic)
                assembly = GalerkinAssembly(self.problem(), basis, QuadratureFactory(arithmetic).plan(8))
                physical = assembly.nonlinear_system()
                matrix, offset = assembly.reference_operator()
                path = ResidualHomotopy(physical, AffineReference(matrix, offset, physical))
                state = arithmetic.vector(["0.17", "-0.11", "0.08", "0.3", "0.2", "-0.1", "0.07"])
                direction = arithmetic.vector([1, -2, 3, -1, 2, -3, 1])
                q = arithmetic.scalar("0.375")
                system = path.nonlinear_system(state)
                residual = system.residual(state, q)
                gradient = system.merit_gradient(residual, system.jacobian(state, q))
                step = arithmetic.scalar("1e-15" if arithmetic.use_mpmath else "1e-5")

                def merit(candidate):
                    value = system.residual(candidate, q)
                    return arithmetic.dot(value, value)/2

                difference = (merit(state+step*direction)-merit(state-step*direction))/(2*step)
                observed = arithmetic.dot(gradient, direction)
                relative = abs(difference-observed)/max(arithmetic.scalar(1), abs(observed))
                self.assertLess(relative, arithmetic.scalar("1e-25" if arithmetic.use_mpmath else "1e-6"))
                self.assertLess(arithmetic.dot(gradient, system.adjoint_direction(gradient)), 0)


if __name__ == "__main__":
    unittest.main()
