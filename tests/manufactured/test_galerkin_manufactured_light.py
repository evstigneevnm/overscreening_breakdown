"""Common-interface mixed solves, with independent physical field diagnostics."""

import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown import BSKProblem, Coefficient, NumpyArithmetic, solve
from overscreening_breakdown.manufactured import rational_case, transmission_case
from overscreening_breakdown.quadrature import QuadratureFactory


class GalerkinManufacturedLightTests(unittest.TestCase):
    def configurations(self):
        yield False, 15
        for precision in (15, 30, 50, 100, 120):
            yield True, precision

    def tolerance(self, use_mpmath, precision):
        return "1e-{}".format(max(10, precision-8) if use_mpmath else 11)

    def check_rational_fields(self, solution, field_bound=None):
        arithmetic = solution.arithmetic
        tolerance = arithmetic.scalar(self.tolerance(arithmetic.use_mpmath, arithmetic.dps))
        bound = 1000*tolerance if field_bound is None else arithmetic.scalar(field_bound)
        case = rational_case()
        for text in ("0", "0.125", "0.75", "1", "3.5", "12"):
            x = arithmetic.scalar(text)
            u, w = solution.u(x), solution.w(x)
            self.assertLess(abs(u-case.u(x, arithmetic)), bound)
            self.assertLess(abs(w-case.w(x, arithmetic)), bound)
            self.assertIsInstance(u, type(arithmetic.scalar(0)))
            self.assertIsInstance(w, type(arithmetic.scalar(0)))
        self.assertEqual((solution.u(arithmetic.inf), solution.w(arithmetic.inf)), (0, 0))
        self.assertTrue(solution.nonlinear_converged)
        self.assertTrue(solution.quadrature_converged)
        self.assertEqual(solution.newton_result.homotopy, 0)

    def test_linear_exact_space_solutions_for_all_precisions_and_coefficients(self):
        for use_mpmath, precision in self.configurations():
            for coefficient in ("constant", "scaled", "smooth"):
                with self.subTest(mpmath=use_mpmath, dps=precision, epsilon=coefficient):
                    result = solve(
                        rational_case(coefficient).make_problem(), 4, solver="galerkin",
                        use_mpmath=use_mpmath, precision=precision, use_globalization=False,
                        tolerance=self.tolerance(use_mpmath, precision),
                    )
                    self.assertTrue(result.converged)
                    self.assertLessEqual(result.newton_result.iterations, 2)
                    self.check_rational_fields(result)

    def test_nonlinear_exact_space_solutions_for_all_precisions_and_coefficients(self):
        for use_mpmath, precision in self.configurations():
            for coefficient in ("constant", "scaled", "smooth"):
                with self.subTest(mpmath=use_mpmath, dps=precision, epsilon=coefficient):
                    result = solve(
                        rational_case(coefficient, nonlinear=True).make_problem(), 4,
                        solver="galerkin", use_mpmath=use_mpmath, precision=precision,
                        initial_guess=["0.125", "0.0625", "-0.03125", 0, 0, 0, 0],
                        tolerance=self.tolerance(use_mpmath, precision),
                    )
                    self.assertTrue(result.converged)
                    self.assertGreater(result.newton_result.iterations, 1)
                    self.check_rational_fields(result)

    def test_globalization_enabled_and_disabled_reach_the_physical_system(self):
        for use_mpmath, precision in ((False, 15), (True, 50)):
            options = dict(solver="galerkin", use_mpmath=use_mpmath, precision=precision,
                           tolerance=self.tolerance(use_mpmath, precision),
                           initial_guess=["0.125", 0, 0, 0, 0, 0, 0])
            problem = rational_case("smooth", nonlinear=True).make_problem()
            plain = solve(problem, 4, use_globalization=False, **options)
            globalized = solve(problem, 4, use_globalization=True, **options)
            self.assertTrue(plain.converged and globalized.converged)
            arithmetic = globalized.arithmetic
            self.assertLess(arithmetic.norm(arithmetic.vector(plain.coefficients)
                                            -globalized.coefficients),
                            100*arithmetic.scalar(options["tolerance"]))
            self.assertEqual(globalized.newton_result.homotopy, 0)

    def test_quadrature_refinement_is_independent_of_nonlinear_convergence(self):
        for use_mpmath, precision in ((False, 15), (True, 30)):
            problem = rational_case("exponential", nonlinear=True).make_problem()
            options = dict(solver="galerkin", use_mpmath=use_mpmath, precision=precision,
                           tolerance=self.tolerance(use_mpmath, precision), quadrature_order=5,
                           quadrature_atol="1e-10", quadrature_rtol="1e-10")
            coarse = solve(problem, 4, max_refinements=0, **options)
            self.assertTrue(coarse.nonlinear_converged)
            self.assertFalse(coarse.quadrature_converged)
            self.assertFalse(coarse.converged)
            refined = solve(problem, 4, initial_guess=coarse, max_refinements=3, **options)
            self.assertTrue(refined.converged)
            self.assertGreater(refined.plan.order, coarse.plan.order)
            self.assertEqual(refined.plan.order, refined.quadrature_history[-1].order)
            self.check_rational_fields(refined, field_bound="1e-8")

    def test_runnable_example_reuses_one_problem_for_numpy_and_mpmath(self):
        from examples.galerkin import make_case, run
        from examples._common import sampled_error

        case = make_case()
        problem = case.make_problem()
        original_bits = mp.mp.prec
        states = []
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                solution = run(problem=problem, use_mpmath=use_mpmath, precision=50)
                newton = solution.newton_result
                check = solution.quadrature_history[-1]
                error = sampled_error(solution, case)
                self.assertTrue(solution.converged)
                self.assertTrue(newton.converged)
                self.assertTrue(check.quadrature_converged)
                self.assertEqual(newton.homotopy, 0)
                self.assertLess(error, solution.arithmetic.scalar(
                    "1e-35" if use_mpmath else "1e-9"))
                self.assertEqual(mp.mp.prec, original_bits)
                self.assertEqual(solution.physical.delta, 2)
                self.assertEqual(solution.physical.boundary_conditions[0][0], 1)
                self.check_rational_fields(solution)
                states.append(np.asarray([float(value) for value in solution.coefficients]))
        np.testing.assert_allclose(states[0], states[1], rtol=1e-10, atol=1e-10)

    def test_bsk_problem_binds_and_solves_with_variable_epsilon(self):
        coefficient = Coefficient(lambda x, a: 1+a.scalar(1)/(1+x),
                                  lower_bound=1, upper_bound=2, wall_value=2)
        problem = BSKProblem(
            sigma=2, mu=0.1, gamma=0.5, delta=1, epsilon=coefficient,
            boundary_conditions=((0.05, None, None, 0, None), (None,)*5),
        )
        source = problem.bind(NumpyArithmetic()).physical_rhs(0.5, 0.03)
        previous_bits = mp.mp.prec
        samples = []
        for use_mpmath, precision in ((False, 15), (True, 30)):
            result = solve(
                problem, 12, solver="galerkin", use_mpmath=use_mpmath, precision=precision,
                tolerance=self.tolerance(use_mpmath, precision), quadrature_order=28,
                quadrature_atol="1e-9", quadrature_rtol="1e-9",
            )
            self.assertTrue(result.converged)
            self.assertEqual(result.physical.delta, 1)
            self.assertEqual(result.physical.epsilon.wall_value, 2)
            self.assertEqual(result.u(0), result.arithmetic.scalar(0.05))
            self.assertEqual((result.u(result.arithmetic.inf), result.w(result.arithmetic.inf)), (0, 0))
            self.assertAlmostEqual(float(result.physical.physical_rhs(0.5, 0.03)), source, places=14)
            values = []
            for text in ("0", "0.125", "1", "3", "8"):
                fields = (result.u(text), result.w(text))
                self.assertTrue(all(result.arithmetic.isfinite(value) for value in fields))
                values.extend(float(value) for value in fields)
            samples.append(np.asarray(values))
            self.assertEqual(mp.mp.prec, previous_bits)
        np.testing.assert_allclose(samples[0], samples[1], rtol=1e-8, atol=1e-10)
        self.assertEqual(problem.bind(NumpyArithmetic()).physical_rhs(0.5, 0.03), source)

    def test_transmission_solution_improves_in_physical_field_and_flux_norms(self):
        case = transmission_case()
        for use_mpmath, precision in ((False, 15), (True, 30)):
            errors = []
            for N in (8, 16):
                with self.subTest(mpmath=use_mpmath, N=N):
                    result = solve(
                        case.make_problem(), N, solver="galerkin", use_mpmath=use_mpmath,
                        precision=precision, tolerance=self.tolerance(use_mpmath, precision),
                        quadrature_order=2*N+4, use_globalization=False,
                        # The absolute residual comparison at float64 includes
                        # cancellation from the larger derivative matrices.
                        quadrature_atol=(self.tolerance(True, precision)
                                         if use_mpmath else "1e-9"),
                    )
                    self.assertTrue(result.converged)
                    arithmetic = result.arithmetic
                    plan = QuadratureFactory(arithmetic, interfaces=(1,)).plan(2*N+3)
                    errors.append(tuple(result.l2_error(
                        lambda x, field=field: getattr(case, field)(x, arithmetic),
                        field=field, plan=plan) for field in ("u", "w", "flux")))
            if len(errors) != 2:
                continue
            for (coarse, fine), bound in zip(zip(*errors), ("0.012", "0.008", "0.25")):
                self.assertLess(fine, arithmetic.scalar("0.8")*coarse)
                self.assertLess(fine, arithmetic.scalar(bound))
