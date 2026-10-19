"""Small nonlinear manufactured problems for the CI method/precision matrix.

Run with ``python -m unittest test_manufactured_nonlinear_light``. Spatial
accuracy is bounded separately from the arithmetic precision of the solve.
"""

from contextlib import redirect_stdout
import io
import unittest

import mpmath as mp
import numpy as np

from manufactured_problems import nonlinear_problem_1, nonlinear_problem_2
from spectral1d import solve_nonlinear_problem


MPMATH_PRECISIONS = (15, 30, 50, 100, 120)
NUMPY_METHODS = ("newton", "root", "root_no_jac", "nonlin_solve")
SAMPLE_POINTS = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])


class NonlinearManufacturedLightTests(unittest.TestCase):
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

    def check_boundaries(self, solver, problem, coefficients, bound):
        # Mapping the half-line reverses the two basis endpoints.
        for boundary, endpoint, derivative in zip(
            problem.get_boundary_conditions(),
            reversed(solver.basis_boundaries()),
            (
                solver.psi_with_derivative_value_at_right,
                solver.psi_with_derivative_value_at_left,
            ),
        ):
            for order, target in enumerate(boundary):
                if target is None:
                    continue
                actual = sum(
                    np.asarray(
                        derivative(k, order) if order else solver.psi(k, endpoint)
                    ).item()
                    * coefficient
                    for k, coefficient in enumerate(coefficients)
                )
                self.assertLess(abs(actual - target), bound)

    def check_case(self, problem_class, size, method, use_mpmath, precision):
        problem = problem_class()
        # Leave guard digits for cancellation in the fourth-order operator.
        tolerance = 10.0 ** -max(10, precision - 10) if use_mpmath else 1e-11
        solver = solve_nonlinear_problem(
            size,
            problem=problem,
            L=2,
            tolerance=tolerance,
            use_method=method,
            use_globalization=False,
            use_mpmath=use_mpmath,
            prec=precision,
        )
        coefficients = solver.solve_problem()
        self.assertTrue(solver.is_converged())
        self.assertEqual(
            len(coefficients),
            size + sum(
                value is not None
                for boundary in problem.get_boundary_conditions()
                for value in boundary
            ),
        )
        if use_mpmath:
            self.assertTrue(all(isinstance(value, mp.mpf) for value in coefficients))
            arithmetic_bound = mp.mpf("1e-" + str(max(8, precision - 12)))
        else:
            self.assertEqual(coefficients.dtype, np.dtype("float64"))
            arithmetic_bound = 1e-8
        self.assertLess(solver.residual_L2_norm(coefficients), arithmetic_bound)

        # Re-evaluate the differential equation independently of the solver's
        # assembled matrix and its convergence flag, with globalization q=0.
        points = solver.all_discrete_points_in_domain()
        basis_points = solver.all_discrete_points_in_basis()
        values = solver.solution_in_basis(coefficients, basis_points)
        lhs = np.zeros(size, dtype=object if use_mpmath else float)
        for order, factor in enumerate(problem.left_hand_side()):
            if factor is not None:
                lhs = lhs + factor * solver.solution_with_derivative_in_basis(
                    coefficients, basis_points, order
                )
        rhs = problem.right_hand_side(points, values, 0)
        self.assertLess(max(abs(value) for value in lhs - rhs), arithmetic_bound)
        self.check_boundaries(solver, problem, coefficients, arithmetic_bound)

        samples = solver.solution_in_domain(coefficients, SAMPLE_POINTS)
        expected = problem.solution_in_domain(SAMPLE_POINTS)
        sample_bound = 5e-4 if size == 12 else 1e-5
        self.assertLess(max(abs(value) for value in samples - expected), sample_bound)
        error = solver.L2_error(coefficients, problem.solution_in_domain)
        self.assertTrue(np.isfinite(error))
        self.assertLess(error, 2e-4 if size == 12 else 1e-5)
        return error, np.asarray(samples, dtype=float)

    def test_second_order_accuracy_refines_for_every_method_and_precision(self):
        reference_samples = {}
        for method, use_mpmath, precision in self.configurations():
            with self.subTest(method=method, mpmath=use_mpmath, dps=precision):
                errors = {}
                for size in (12, 24):
                    errors[size], samples = self.check_case(
                        nonlinear_problem_1, size, method, use_mpmath, precision
                    )
                    if size not in reference_samples:
                        reference_samples[size] = samples
                    else:
                        np.testing.assert_allclose(
                            samples, reference_samples[size], rtol=1e-8, atol=1e-9
                        )
                self.assertLess(errors[24], errors[12] / 10)

    def test_fourth_order_accuracy_for_every_method_and_precision(self):
        reference_samples = None
        for method, use_mpmath, precision in self.configurations():
            with self.subTest(method=method, mpmath=use_mpmath, dps=precision):
                _, samples = self.check_case(
                    nonlinear_problem_2, 24, method, use_mpmath, precision
                )
                if reference_samples is None:
                    reference_samples = samples
                else:
                    np.testing.assert_allclose(
                        samples, reference_samples, rtol=1e-8, atol=1e-9
                    )


if __name__ == "__main__":
    unittest.main()
