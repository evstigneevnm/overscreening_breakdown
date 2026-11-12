"""Native affine references and boundary layouts for collocation continuation."""

import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.collocation.adapter import CollocationProblemAdapter
from overscreening_breakdown.collocation.spectral1d import solve_nonlinear_problem
from overscreening_breakdown.problems.flux import FluxProblem


class CollocationReferenceTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)

    def configurations(self):
        yield False, 15
        for precision in (15, 30, 50, 100, 120):
            yield True, precision

    def kernel(self, *, use_mpmath=False, precision=50, **options):
        arguments = dict(
            delta="2.125", f=lambda x, u, a: u,
            f_u=lambda x, u, a: a.scalar(1),
            boundary_conditions=((1, None, None, -6, None),
                                 (0, None, None, None, None)),
        )
        arguments.update(options)
        arithmetic = MpmathArithmetic(precision) if use_mpmath else NumpyArithmetic()
        adapter = CollocationProblemAdapter(FluxProblem(**arguments).bind(arithmetic))
        return solve_nonlinear_problem(
            6, problem=adapter, L=2, use_mpmath=use_mpmath, prec=precision,
            use_globalization=False, newton_fallback="adjoint",
        )

    def scalar(self, use_mpmath):
        return mp.mpf if use_mpmath else np.float64

    def vector(self, values, use_mpmath):
        values = list(values)
        return mp.matrix(values) if use_mpmath else np.asarray(values, dtype=np.float64)

    def product(self, matrix, state, use_mpmath):
        return matrix*state if use_mpmath else matrix @ state

    def norm(self, values, use_mpmath):
        return mp.norm(values) if use_mpmath else np.linalg.norm(values)

    def tolerance(self, use_mpmath, precision):
        scalar = self.scalar(use_mpmath)
        return scalar("1e-{}".format(max(9, precision-7) if use_mpmath else 9))

    def wall_derivative(self, coefficients, derivative, precision):
        # Differentiate the represented rational polynomial in physical x;
        # the angular derivative formulas are singular at the mapped wall.
        oracle = mp.mp.clone()
        oracle.dps = max(40, precision+20)
        values = [oracle.mpf(value) for value in coefficients]

        def field(x):
            y = (x-2)/(x+2)
            return oracle.fsum(value*oracle.chebyt(k, y)
                               for k, value in enumerate(values))

        return oracle.diff(field, oracle.mpf(0), derivative)

    def test_reference_matches_an_independently_supplied_linear_physical_source(self):
        for use_mpmath, precision in self.configurations():
            with self.subTest(mpmath=use_mpmath, dps=precision):
                scalar = self.scalar(use_mpmath)
                kernel = self.kernel(
                    use_mpmath=use_mpmath, precision=precision,
                    f=lambda x, u, a: a.scalar("0.375")*u,
                    f_u=lambda x, u, a: a.scalar("0.375"),
                )
                state = self.vector((scalar(k-3)/11 for k in range(kernel._M)),
                                    use_mpmath)
                system = kernel.nonlinear_system(state)
                matrix, offset = kernel.reference_operator("0.375")
                affine = self.product(matrix, state, use_mpmath)+offset
                physical = system.residual(state, scalar(0))
                scale = max(scalar(1), self.norm(physical, use_mpmath))
                self.assertLess(self.norm(affine-physical, use_mpmath),
                                self.tolerance(use_mpmath, precision)*scale)
                self.assertEqual((len(offset), matrix.rows if use_mpmath else matrix.shape[0]),
                                 (9, 9))

    def test_reference_construction_does_not_call_physical_source_or_initial_guess(self):
        def forbidden(*arguments):
            raise AssertionError("reference assembly must not evaluate a physical callback")

        for use_mpmath, precision in self.configurations():
            with self.subTest(mpmath=use_mpmath, dps=precision):
                scalar = self.scalar(use_mpmath)
                kernel = self.kernel(use_mpmath=use_mpmath, precision=precision,
                                     f=forbidden, f_u=forbidden, initial_guess=forbidden)
                matrix, offset = kernel.reference_operator()
                coefficients = kernel.solve(matrix, -offset)
                residual = self.product(matrix, coefficients, use_mpmath)+offset
                bound = self.tolerance(use_mpmath, precision)
                self.assertLess(self.norm(residual, use_mpmath), bound)
                self.assertLess(abs(kernel.solution_in_domain(coefficients, scalar(0))[0]-1),
                                bound)
                self.assertLess(abs(kernel.solution_in_domain(coefficients, scalar("inf"))[0]),
                                bound)
                self.assertLess(abs(self.wall_derivative(coefficients, 3, precision)+6),
                                bound)
                if use_mpmath:
                    self.assertTrue(all(isinstance(value, mp.mpf) for value in coefficients))
                else:
                    self.assertEqual(coefficients.dtype, np.dtype("float64"))

    def test_boundary_count_controls_reference_shape_and_row_coordinates(self):
        cases = (
            (0, (0, "inf"), ((1, None, None, None, None), (None,)*5), (0,)),
            (0, (0, "inf"), ((1, None, None, None, None),
                            (0, None, None, None, None)), ("inf", 0)),
            (2, (0, "inf"), ((1, None, None, -6, None),
                            (0, None, None, None, None)), ("inf", 0, 0)),
            (2, (0, 2), ((1, 0, None, None, None),
                        (0, 0, None, None, None)), (2, 2, 0, 0)),
        )
        for use_mpmath in (False, True):
            scalar = self.scalar(use_mpmath)
            for delta, domain, boundaries, endpoints in cases:
                with self.subTest(mpmath=use_mpmath, delta=delta, domain=domain):
                    kernel = self.kernel(use_mpmath=use_mpmath, delta=delta,
                                         domain=domain, boundary_conditions=boundaries)
                    count = 6+len(endpoints)
                    matrix, offset = kernel.reference_operator()
                    self.assertEqual(kernel._M, count)
                    self.assertEqual(len(offset), count)
                    self.assertEqual(matrix.rows if use_mpmath else matrix.shape[0], count)
                    points = kernel.all_discrete_points_in_domain_with_bounds()
                    self.assertEqual(points.shape, (count,))
                    rows = (0,)+tuple(6+j for j in range(1, len(endpoints)))
                    for row, endpoint in zip(rows, endpoints):
                        self.assertEqual(points[row], scalar(endpoint))
                    coefficients = kernel.solve(matrix, -offset)
                    system = kernel.nonlinear_system(coefficients)
                    self.assertEqual(len(system.initial_guess()), count)
                    self.assertLess(self.norm(system.residual(coefficients, scalar(0)),
                                              use_mpmath),
                                    self.tolerance(use_mpmath, 50))

    def test_adjoint_geometry_preserves_all_three_explicit_half_line_constraints(self):
        for use_mpmath, precision in self.configurations():
            with self.subTest(mpmath=use_mpmath, dps=precision):
                scalar = self.scalar(use_mpmath)
                kernel = self.kernel(use_mpmath=use_mpmath, precision=precision)
                state = self.vector((scalar(0) for _ in range(kernel._M)), use_mpmath)
                system = kernel.nonlinear_system(state)
                gradient = self.vector((scalar(k+1)/7 for k in range(kernel._M)),
                                       use_mpmath)
                direction = system.adjoint_direction(gradient)
                bound = 100*self.tolerance(use_mpmath, precision)
                self.assertEqual(len(direction), kernel._M)
                self.assertLess(abs(self.wall_derivative(direction, 0, precision)), bound)
                self.assertLess(abs(self.wall_derivative(direction, 3, precision)), bound)
                self.assertLess(abs(sum(direction)), bound)

    def test_reference_strength_retains_more_than_float64_precision(self):
        kernel = self.kernel(use_mpmath=True, precision=120)
        base, _ = kernel.reference_operator("1")
        perturbed, _ = kernel.reference_operator("1."+"0"*79+"1")
        observed = perturbed[1, 0]-base[1, 0]
        expected = mp.mpf("1e-80")/mp.mpf("2.125")**2
        self.assertGreater(observed, 0)
        self.assertLess(abs(observed-expected), mp.mpf("1e-115"))

    def test_physical_system_rejects_nonzero_continuation_parameters(self):
        kernel = self.kernel()
        system = kernel.nonlinear_system(np.zeros(kernel._M))
        for callback in (system.residual, system.jacobian):
            with self.assertRaisesRegex(ValueError, "requires q=0"):
                callback(np.zeros(kernel._M), 0.25)

    def test_reference_strength_validation(self):
        for use_mpmath in (False, True):
            kernel = self.kernel(use_mpmath=use_mpmath)
            for invalid in (0, -1, "nan", "inf", "-inf"):
                with self.subTest(mpmath=use_mpmath, strength=invalid):
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        kernel.reference_operator(invalid)


if __name__ == "__main__":
    unittest.main()
