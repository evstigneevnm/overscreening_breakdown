"""End-to-end collocation solves from unscaled physical FluxProblem data.

The exact fields lie in the discrete space, so the high-precision checks
measure arithmetic accuracy rather than a fixed spatial approximation error.
"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic
from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.problems.flux import FluxProblem
from overscreening_breakdown.collocation.spectral1d import solve_nonlinear_problem
from overscreening_breakdown.manufactured import rational_case, polynomial_segment_case


NUMPY_METHODS = ("newton", "root", "root_no_jac", "nonlin_solve")
MPMATH_PRECISIONS = (15, 30, 50, 100, 120)


def rational_solution(x):
    return 1/(1+x)


def rational_problem(**options):
    return replace(rational_case(nonlinear=True, beta="0.1", initial="exp").make_problem(),
                   **options)


def polynomial_solution(x):
    return (1-x/2)**3


def polynomial_problem(**options):
    return replace(polynomial_segment_case().make_problem(), **options)


class FluxCollocationTests(unittest.TestCase):
    def setUp(self):
        original_precision = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_precision)
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def configurations(self):
        for method in NUMPY_METHODS:
            yield method, False, 15
        for precision in MPMATH_PRECISIONS:
            yield "newton", True, precision

    def solver(self, problem=None, method="newton", use_mpmath=False,
               precision=50, globalization=False, domain=None):
        tolerance = "1e-{}".format(max(9, precision-10)) if use_mpmath else 1e-11
        return solve_nonlinear_problem(
            6, problem=problem, domain=None if domain is None else list(domain),
            L=1, tolerance=tolerance,
            use_method=method, use_mpmath=use_mpmath, prec=precision,
            use_globalization=globalization,
        )

    def check_solution(self, solver, problem, exact, use_mpmath, precision):
        coefficients = solver.solve_problem()
        self.assertTrue(solver.is_converged())
        scalar = mp.mpf if use_mpmath else np.float64
        bound = scalar("1e-{}".format(max(7, precision-14))) if use_mpmath else scalar("1e-8")
        self.assertLess(solver.residual_L2_norm(coefficients), bound)
        if use_mpmath:
            self.assertTrue(all(hasattr(value, "_mpf_") for value in coefficients))
        else:
            self.assertEqual(coefficients.dtype, np.dtype("float64"))

        # Native scalar coordinates and expectations keep this independent of
        # float64 sampling, including at 100 and 120 significant digits.
        samples = ("0", "0.1", "0.5", "1", "2")
        if problem.domain[1] == "inf":
            samples += ("5", "10")
        points = np.array([scalar(value) for value in samples], dtype=object if use_mpmath else float)
        actual = solver.solution_in_domain(coefficients, points.copy())
        self.assertLess(max(abs(value-exact(x)) for x, value in zip(points, actual)), bound)
        for boundary, endpoint, derivative in zip(
            problem.boundary_conditions, reversed(solver.basis_boundaries()),
            (solver.psi_with_derivative_value_at_right,
             solver.psi_with_derivative_value_at_left),
        ):
            for order, target in enumerate(boundary):
                if target is None:
                    continue
                trace = sum(
                    np.asarray(derivative(k, order) if order else solver.psi(k, endpoint)).item()*value
                    for k, value in enumerate(coefficients)
                )
                self.assertLess(abs(trace-scalar(target)), 10*bound)

        # Check the unscaled physical equation separately from the assembled
        # normalized residual and its convergence flag.
        physical = problem.bind(MpmathArithmetic(precision)) if use_mpmath else solver._solve_nonlinear_problem__problem.physical
        interior = solver.all_discrete_points_in_domain()
        values = solver.solution_in_domain(coefficients, interior.copy())
        second = solver.solution_with_derivative_in_domain(coefficients, interior.copy(), 2)
        fourth = solver.solution_with_derivative_in_domain(coefficients, interior.copy(), 4)
        physical_errors = [
            abs(d2-physical.delta**2*d4-physical.physical_rhs(x, u))
            for x, u, d2, d4 in zip(interior, values, second, fourth)
        ]
        self.assertLess(max(physical_errors), 10*bound)
        return coefficients

    def test_rational_half_line_for_every_method_and_precision(self):
        problem = rational_problem()
        for method, use_mpmath, precision in self.configurations():
            with self.subTest(method=method, mpmath=use_mpmath, dps=precision):
                solver = self.solver(problem, method, use_mpmath, precision)
                self.check_solution(solver, problem, rational_solution, use_mpmath, precision)

    def test_polynomial_segment_for_every_method_and_precision(self):
        problem = polynomial_problem()
        for method, use_mpmath, precision in self.configurations():
            with self.subTest(method=method, mpmath=use_mpmath, dps=precision):
                solver = self.solver(problem, method, use_mpmath, precision)
                self.check_solution(solver, problem, polynomial_solution, use_mpmath, precision)

    def test_set_problem_adapts_and_updates_the_domain(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                solver = self.solver(domain=(0, 3), use_mpmath=use_mpmath)
                problem = polynomial_problem()
                solver.set_problem(problem)
                self.check_solution(solver, problem, polynomial_solution, use_mpmath, 50)

    def test_second_order_problem_has_no_delta_division(self):
        for use_mpmath in (False, True):
            for segment in (False, True):
                with self.subTest(mpmath=use_mpmath, segment=segment):
                    if segment:
                        exact = lambda x: 1-x/2
                        problem = FluxProblem(
                            delta=0, domain=(0, 2),
                            f=lambda x, u, arithmetic: u-(1-x/2),
                            f_u=lambda x, u, arithmetic: arithmetic.scalar(1),
                            boundary_conditions=((1, None, None, None, None), (0, None, None, None, None)),
                            initial_guess=lambda x, arithmetic: arithmetic.scalar("0.8")*(1-x/2),
                        )
                    else:
                        exact = rational_solution
                        problem = rational_problem(
                            delta=0,
                            f=lambda x, u, arithmetic: 2/(1+x)**3+u-1/(1+x),
                            f_u=lambda x, u, arithmetic: arithmetic.scalar(1),
                            boundary_conditions=((1, None, None, None, None), (None,)*5),
                        )
                    solver = self.solver(problem, use_mpmath=use_mpmath)
                    self.check_solution(solver, problem, exact, use_mpmath, 50)

    def test_callbacks_only_receive_actual_interior_coordinates(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                calls = {name: [] for name in ("f", "f_u", "initial_guess")}
                original = rational_problem()

                def record(name, callback):
                    def checked(x, *arguments):
                        self.assertEqual(np.ndim(x), 0)
                        self.assertGreater(x, 0)
                        self.assertTrue(mp.isfinite(x))
                        calls[name].append(x)
                        return callback(x, *arguments)
                    return checked

                problem = rational_problem(
                    f=record("f", original.f),
                    f_u=record("f_u", original.f_u),
                    initial_guess=record("initial_guess", original.initial_guess),
                )
                solver = self.solver(problem, use_mpmath=use_mpmath)
                coefficients = solver.solve_problem()
                self.assertTrue(solver.is_converged())
                self.assertLess(solver.residual_L2_norm(coefficients), 1e-8)
                for name, coordinates in calls.items():
                    self.assertTrue(coordinates, name)

    def test_jacobian_direction_at_non_solution_and_nonzero_homotopy(self):
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                solver = self.solver(rational_problem(), use_mpmath=use_mpmath)
                scalar = mp.mpf if use_mpmath else np.float64
                adapter = solver._solve_nonlinear_problem__problem
                state = solver.function_expand(adapter.base_solution)
                direction_values = [scalar(k+1)/scalar(len(state)) for k in range(len(state))]
                direction = mp.matrix(direction_values) if use_mpmath else np.array(direction_values)
                state = state+scalar("0.03")*direction
                residual = solver._solve_nonlinear_problem__residual_operator
                jacobian = solver._solve_nonlinear_problem__linearization_operator
                step = scalar("1e-15" if use_mpmath else "1e-5")
                q = scalar("0.7")
                difference = (residual(state+step*direction, q)-residual(state-step*direction, q))/(2*step)
                matrix = jacobian(state, q)
                product = matrix*direction if use_mpmath else matrix @ direction
                norm = mp.norm if use_mpmath else np.linalg.norm
                error = norm(difference-product)/max(scalar(1), norm(product))
                self.assertLess(error, scalar("1e-27" if use_mpmath else "1e-7"))

    def test_globalization_option_reaches_the_physical_problem(self):
        for use_mpmath in (False, True):
            for globalization in (False, True):
                with self.subTest(mpmath=use_mpmath, globalization=globalization):
                    problem = rational_problem()
                    solver = self.solver(problem, use_mpmath=use_mpmath, globalization=globalization)
                    self.check_solution(solver, problem, rational_solution, use_mpmath, 50)
                    self.assertEqual(solver.get_newton_result().homotopy, 0)
                    self.assertTrue(solver.get_newton_result().stages[-1].converged)

    def test_solver_binding_keeps_source_and_physical_contexts_independent(self):
        delta = "2."+"0"*79+"1"
        problem = rational_problem(delta=delta)
        high_solver = self.solver(problem, use_mpmath=True, precision=120)
        high = high_solver._solve_nonlinear_problem__problem.physical
        high_source = high.physical_rhs("0.2", "0.3")
        low_solver = self.solver(problem, use_mpmath=True, precision=30)
        low = low_solver._solve_nonlinear_problem__problem.physical
        self.assertEqual(problem.delta, delta)
        self.assertEqual(problem.domain, (0, "inf"))
        self.assertEqual(problem.bind(MpmathArithmetic(30)).boundary_conditions[0],
                         (1, None, None, -6, None))
        self.assertIsNot(high.arithmetic, low.arithmetic)
        self.assertEqual(high.arithmetic.dps, 120)
        self.assertEqual(low.arithmetic.dps, 30)
        self.assertGreater(high.delta, 2)
        self.assertEqual(low.delta, 2)
        self.assertEqual(high.physical_rhs("0.2", "0.3"), high_source)

    def test_runnable_example_reuses_one_problem_in_both_arithmetics(self):
        from examples.collocation import make_case, run
        from examples._common import sampled_error

        case = make_case()
        problem = case.make_problem()
        before = (problem.delta, problem.epsilon, problem.domain,
                  problem.boundary_conditions, problem.f, problem.f_u,
                  problem.initial_guess)
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                solver = run(problem=problem, use_mpmath=use_mpmath, precision=50)
                error = sampled_error(solver, case)
                self.assertTrue(solver.converged)
                bound = mp.mpf("1e-30") if use_mpmath else 1e-8
                self.assertLess(error, bound)
                self.assertLess(solver.newton_result.residual_norm, bound)
                self.assertEqual(
                    (problem.delta, problem.epsilon, problem.domain,
                     problem.boundary_conditions, problem.f, problem.f_u,
                     problem.initial_guess), before,
                )

    def test_incompatible_problem_is_rejected_before_collocation_assembly(self):
        opaque = Coefficient(lambda x, arithmetic: arithmetic.scalar(1), lower_bound=1, upper_bound=1, wall_value=1)
        incompatible = (rational_problem(epsilon=2), rational_problem(epsilon=opaque), rational_problem().bind(MpmathArithmetic(50)))
        for problem in incompatible:
            for constructor in (True, False):
                with self.subTest(problem=type(problem).__name__, constructor=constructor):
                    with patch("overscreening_breakdown.collocation.spectral1d.collocation_discretization.bilinear_form", side_effect=AssertionError("assembled before validation")):
                        with self.assertRaises((TypeError, ValueError)):
                            if constructor:
                                self.solver(problem)
                            else:
                                self.solver(domain=(0, np.inf)).set_problem(problem)

    def test_physical_callback_failure_policy(self):
        def broken(*arguments):
            raise RuntimeError("physical callback defect")

        for callback in ("f", "f_u"):
            for use_mpmath in (False, True):
                with self.subTest(callback=callback, mpmath=use_mpmath):
                    solver = self.solver(rational_problem(**{callback: broken}), use_mpmath=use_mpmath)
                    with self.assertRaisesRegex(RuntimeError, "physical callback defect"):
                        solver.solve_problem()
                    self.assertFalse(solver.is_converged())
                with self.subTest(callback=callback, mpmath=use_mpmath, nonfinite=True):
                    solver = self.solver(
                        rational_problem(**{callback: lambda x, u, arithmetic: arithmetic.inf}),
                        use_mpmath=use_mpmath,
                    )
                    coefficients = solver.solve_problem()
                    self.assertTrue(all(mp.isfinite(value) for value in coefficients))
                    self.assertFalse(solver.is_converged())
                    self.assertEqual(
                        solver.get_newton_result().reason,
                        "nonfinite_residual" if callback == "f" else "nonfinite_jacobian",
                    )


if __name__ == "__main__":
    unittest.main()
