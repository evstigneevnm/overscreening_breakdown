"""Small analytic linear problems for CI; no plots or default parameter sweeps.

Run with ``python -m unittest test_manufactured_linear_light``.
"""

import unittest

import mpmath as mp
import numpy as np

from manufactured_problems import problem_2, problem_3, problem_5
from spectral1d import collocation_discretization


MPMATH_PRECISIONS = (15, 30, 50, 100, 120)
RESOLUTIONS = (12, 24)
SAMPLE_POINTS = (0, 0.125, 0.5, 1, 2, 4, 8)


class LinearManufacturedLightTests(unittest.TestCase):
    def setUp(self):
        original_precision = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_precision)

    def solve(self, problem_type, resolution, precision):
        problem = problem_type()
        discretization = collocation_discretization(
            resolution,
            domain=problem.get_domain(),
            L=2,
            use_mpmath=precision is not None,
            prec=precision if precision is not None else 15,
        )
        problem.set_use_mp_math(discretization.get_mpmath_ref())
        discretization.set_boundary_conditions(problem.get_boundary_conditions())

        matrix = discretization.bilinear_form(discretization.psi, 0)
        derivatives = (
            discretization.psi,
            discretization.dpsi,
            discretization.ddpsi,
            discretization.dddpsi,
            discretization.ddddpsi,
        )
        for weight, derivative in zip(problem.operator(), derivatives):
            if weight is not None:
                matrix += discretization.bilinear_form(derivative, weight)
        matrix = discretization.set_boundary_bilinear_form_tau_method(matrix)
        rhs = discretization.linear_functional(lambda x, t: problem.rhs_in_domain(x))
        rhs = discretization.linear_functional_boundary(rhs)
        coefficients = discretization.solve(matrix, rhs)

        if precision is None:
            self.assertEqual(matrix.dtype, np.dtype("float64"))
            self.assertEqual(coefficients.dtype, np.dtype("float64"))
        else:
            self.assertEqual(mp.mp.dps, precision)
            self.assertIsInstance(coefficients, mp.matrix)
            self.assertTrue(all(isinstance(value, mp.mpf) for value in coefficients))

        self.check_residual(discretization, problem, coefficients, precision)
        self.check_boundaries(discretization, problem, coefficients)
        samples = np.array(SAMPLE_POINTS, dtype=object if precision else float)
        actual = discretization.solution_in_domain(coefficients, samples)
        expected = problem.solution_in_domain(samples)
        sample_error = max(abs(value) for value in actual - expected)
        l2_error = discretization.L2_error(coefficients, problem.solution_in_domain)
        self.assertTrue(np.isfinite(l2_error))
        return l2_error, sample_error, np.asarray(actual, dtype=float)

    def check_residual(self, discretization, problem, coefficients, precision):
        # Evaluate the differential equation directly, independently of the
        # matrix and padded right-hand side passed to the linear solver.
        points = discretization.all_discrete_points_in_basis()
        physical_points = discretization.all_discrete_points_in_domain()
        lhs = np.zeros(len(points), dtype=object if precision else float)
        for order, weight in enumerate(problem.operator()):
            if weight is not None:
                lhs += weight * discretization.solution_with_derivative_in_basis(
                    coefficients, points, order
                )
        rhs = problem.rhs_in_domain(physical_points)
        scale = max(1, max(abs(value) for value in rhs))
        relative_residual = max(abs(value) for value in lhs - rhs) / scale
        # High derivatives amplify rounding. Keep guard digits while verifying
        # that mpmath actually computes beyond double precision at higher dps.
        tolerance = 1e-10 if precision is None else mp.power(10, -max(9, precision - 8))
        self.assertLess(relative_residual, tolerance)

    def check_boundaries(self, discretization, problem, coefficients):
        for physical_side, conditions in enumerate(problem.get_boundary_conditions()):
            for order, expected in enumerate(conditions):
                if expected is None:
                    continue
                if order == 0:
                    endpoint = discretization.basis_boundaries()[1 - physical_side]
                    actual = np.asarray(
                        discretization.solution_in_basis(coefficients, endpoint)
                    ).item()
                else:
                    derivative = (
                        discretization.psi_with_derivative_value_at_right
                        if physical_side == 0
                        else discretization.psi_with_derivative_value_at_left
                    )
                    actual = sum(
                        derivative(k, order) * value
                        for k, value in enumerate(coefficients)
                    )
                self.assertLess(abs(actual - expected), 1e-9)

    def check_problem(self, problem_type, fine_l2_bound, sample_bounds):
        numpy_samples = {}
        for precision in (None,) + MPMATH_PRECISIONS:
            errors = []
            for resolution, sample_bound in zip(RESOLUTIONS, sample_bounds):
                with self.subTest(
                    problem=problem_type.__name__, N=resolution, dps=precision
                ):
                    l2_error, sample_error, samples = self.solve(
                        problem_type, resolution, precision
                    )
                    self.assertLess(sample_error, sample_bound)
                    errors.append(l2_error)
                    if precision is None:
                        numpy_samples[resolution] = samples
                    else:
                        np.testing.assert_allclose(
                            samples, numpy_samples[resolution], rtol=1e-8, atol=1e-10
                        )
            if len(errors) == len(RESOLUTIONS):
                with self.subTest(problem=problem_type.__name__, dps=precision):
                    self.assertLess(errors[-1], fine_l2_bound)
                    self.assertLess(errors[-1], errors[0] / 10)

    def test_fourth_order_exponential(self):
        self.check_problem(problem_2, fine_l2_bound=2e-6, sample_bounds=(2e-3, 2e-6))

    def test_second_order_gaussian(self):
        self.check_problem(problem_3, fine_l2_bound=1e-4, sample_bounds=(2e-2, 2e-4))

    def test_fourth_order_rational(self):
        self.check_problem(problem_5, fine_l2_bound=1e-11, sample_bounds=(1e-7, 1e-11))


if __name__ == "__main__":
    unittest.main()
