"""The continuation reference is a solved mixed problem in native arithmetic."""

import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.continuation import AffineReference, ResidualHomotopy
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.problems.flux import FluxProblem
from overscreening_breakdown.quadrature import QuadratureFactory


def unused_source(x, u, arithmetic):
    raise AssertionError("the reference must not evaluate the physical source")


class GalerkinReferenceTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for precision in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(precision)

    def tolerance(self, arithmetic):
        return arithmetic.scalar("3e-11" if not arithmetic.use_mpmath
                                 else "3e-{}".format(max(10, arithmetic.dps-5)))

    def assembly(self, arithmetic, *, epsilon=1, wall="1", wall_third="0.125",
                 length="1", order=5, source=unused_source, source_u=unused_source):
        problem = FluxProblem(
            delta=2, epsilon=epsilon, f=source, f_u=source_u,
            boundary_conditions=((wall, None, None, wall_third, None), (None,)*5),
        )
        basis = GalerkinBasis(4, arithmetic, L=length)
        plan = QuadratureFactory(arithmetic, L=length,
                                 interfaces=getattr(epsilon, "interfaces", ())).plan(order)
        return GalerkinAssembly(problem, basis, plan)

    def matvec(self, arithmetic, matrix, vector):
        return matrix*vector if arithmetic.use_mpmath else matrix @ vector

    def homotopy(self, assembly, strength="1"):
        system = assembly.nonlinear_system()
        matrix, offset = assembly.reference_operator(strength)
        return ResidualHomotopy(system, AffineReference(matrix, offset, system))

    def assert_native(self, arithmetic, values):
        for value in values:
            if arithmetic.use_mpmath:
                self.assertTrue(hasattr(value, "_mpf_"))
                self.assertNotIsInstance(value, (float, np.floating))
            else:
                self.assertIsInstance(value, np.float64)

    def test_reference_moments_include_the_reaction_lift_and_natural_wall_load(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                assembly = self.assembly(arithmetic, epsilon=3)
                strength = arithmetic.scalar("0.375")
                matrix, offset = assembly.reference_operator(strength)
                m = assembly.basis.n_u
                # Q_0=2/(1+x), V_0=4x/(1+x)^2 and lift=1/(1+x).
                # Their physical pairings are (Q_0,V_0)=4 and (lift,Q_0)=2.
                self.assertLess(abs(matrix[m, 0]-4*strength), self.tolerance(arithmetic))
                self.assertLess(abs(offset[m]-(2*strength-1)), self.tolerance(arithmetic))
                self.assertLess(abs(offset[0]+arithmetic.scalar(2)/3),
                                self.tolerance(arithmetic))
                self.assertLess(abs(matrix[m, m]-arithmetic.scalar(28)/3),
                                self.tolerance(arithmetic))
                self.assert_native(arithmetic, offset)
                self.assert_native(arithmetic, (matrix[i, j] for i in range(7) for j in range(7)))

    def test_reference_solve_finds_both_fields_and_retains_all_boundary_data(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                coefficient = Coefficient(
                    lambda x, a: a.scalar(3 if x < 1 else 300),
                    lower_bound=3, upper_bound=300, wall_value=3, interfaces=(1,),
                )
                assembly = self.assembly(arithmetic, epsilon=coefficient)
                matrix, offset = assembly.reference_operator()
                state = arithmetic.solve(matrix, -offset)
                residual = self.matvec(arithmetic, matrix, state)+offset
                self.assertLess(arithmetic.norm(residual), self.tolerance(arithmetic))
                m = assembly.basis.n_u
                self.assertGreater(arithmetic.norm(arithmetic.vector(tuple(state)[m:])), 0)
                # The constitutive block is the physical one, with its lift.
                first = tuple(assembly.linear_residual(state))[:m]
                self.assertLess(arithmetic.norm(arithmetic.vector(first)),
                                self.tolerance(arithmetic))
                self.assertEqual(assembly.basis.evaluate_fields(state, -1, u_b=1)[0], 1)
                self.assertEqual(assembly.basis.evaluate_fields(state, 1, u_b=1), (0, 0))
                self.assertFalse(assembly.physical.epsilon.is_identity)
                self.assertEqual(assembly.physical.epsilon.wall_value, 3)
                with self.assertRaisesRegex(AssertionError, "physical source"):
                    assembly.residual(state)

    def test_reference_matches_an_independent_constant_coefficient_physical_problem(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                coefficient = Coefficient(
                    lambda x, a: 3+a.scalar(9)/(1+x),
                    lower_bound=3, upper_bound=12, wall_value=12,
                )
                physical = self.assembly(arithmetic, epsilon=coefficient)
                strength = arithmetic.scalar("0.625")
                reference = self.assembly(
                    arithmetic, epsilon=12,
                    source=lambda x, u, a: 12*strength*u,
                    source_u=lambda x, u, a: 12*strength,
                )
                matrix, offset = physical.reference_operator(strength)
                state = arithmetic.vector(["0.1", "-0.2", "0.05", "0.3", "0.1", "-0.1", "0.2"])
                self.assertLess(arithmetic.norm(self.matvec(arithmetic, matrix, state)+offset
                                                -reference.residual(state)),
                                self.tolerance(arithmetic))
                self.assertLess(arithmetic.norm(matrix-reference.jacobian(state)),
                                self.tolerance(arithmetic))

    def test_reference_is_independent_of_physical_contrast_and_source_scale(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(120)):
            calls = []

            def coefficient(x, a):
                calls.append(x)
                return a.scalar("1e300") if x < 1 else a.scalar("1e298")

            epsilon = Coefficient(coefficient, lower_bound="1e298", upper_bound="1e300",
                                  wall_value="1e300", interfaces=(1,))
            first = self.assembly(arithmetic, epsilon=epsilon, wall="1e100")
            second = self.assembly(arithmetic, epsilon=1, wall="1e100")
            before = tuple(calls)
            matrix, offset = first.reference_operator()
            other_matrix, other_offset = second.reference_operator()
            self.assertEqual(tuple(calls), before)
            self.assertLess(arithmetic.norm(matrix-other_matrix), self.tolerance(arithmetic))
            scale = arithmetic.norm(other_offset)
            self.assertLess(arithmetic.norm(offset-other_offset)/scale, self.tolerance(arithmetic))
            self.assertTrue(all(arithmetic.isfinite(value) for value in offset))

    def test_reference_rebuilds_in_the_current_plan_and_returns_independent_arrays(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            assembly = self.assembly(arithmetic)
            matrix, offset = assembly.reference_operator()
            refined = assembly.refined()
            new_matrix, new_offset = refined.reference_operator()
            self.assertIs(refined.physical, assembly.physical)
            self.assertGreater(len(refined.plan.nodes), len(assembly.plan.nodes))
            self.assertLess(arithmetic.norm(new_matrix-matrix), self.tolerance(arithmetic))
            self.assertLess(arithmetic.norm(new_offset-offset), self.tolerance(arithmetic))
            matrix[0, 0] += 99
            offset[0] += 99
            self.assertNotEqual(matrix[0, 0], assembly.reference_operator()[0][0, 0])
            self.assertNotEqual(offset[0], assembly.reference_operator()[1][0])

    def test_reference_precision_is_owned_and_retains_decimal_strength(self):
        arithmetic = MpmathArithmetic(120)
        assembly = self.assembly(arithmetic)
        previous = mp.mp.prec
        try:
            mp.mp.dps = 17
            matrix, offset = assembly.reference_operator("1."+"0"*79+"1")
            baseline, base_offset = assembly.reference_operator()
            expected = arithmetic.scalar("1e-80")
            self.assertLess(abs((matrix[3, 0]-baseline[3, 0])/4-expected),
                            arithmetic.scalar("1e-117"))
            self.assertLess(abs((offset[3]-base_offset[3])/2-expected),
                            arithmetic.scalar("1e-117"))
            self.assertEqual(arithmetic.dps, 120)
            arithmetic.context.dps = 130
            with self.assertRaises(RuntimeError):
                assembly.reference_operator()
        finally:
            mp.mp.prec = previous

    def test_reference_strength_must_be_positive_and_finite(self):
        assembly = self.assembly(NumpyArithmetic())
        for strength in (0, -1, "nan", "inf"):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                assembly.reference_operator(strength)

    def test_reference_endpoint_does_not_evaluate_physical_callbacks(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            assembly = self.assembly(arithmetic)
            path = self.homotopy(assembly)
            state = arithmetic.zeros(7)
            self.assertEqual(tuple(path.residual(state, 1)), tuple(path.reference.offset))
            self.assertEqual(arithmetic.norm(path.jacobian(state, 1)-path.reference.matrix), 0)
            with self.assertRaisesRegex(AssertionError, "physical source"):
                path.residual(state, 0)

    def test_full_residual_blend_matches_the_intermediate_physical_flux_problem(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                q = arithmetic.scalar("0.375")
                strength = arithmetic.scalar("0.625")

                def epsilon(x, a):
                    return a.scalar(3 if x < 1 else 9)

                def source(x, u, a):
                    return 3*(u+u**3-a.exp(-x))

                def source_u(x, u, a):
                    return 3*(1+3*u*u)

                physical = self.assembly(
                    arithmetic,
                    epsilon=Coefficient(epsilon, lower_bound=3, upper_bound=9,
                                        wall_value=3, interfaces=(1,)),
                    source=source, source_u=source_u,
                )
                intermediate = self.assembly(
                    arithmetic,
                    epsilon=Coefficient(
                        lambda x, a: (1-q)*epsilon(x, a)+3*q,
                        lower_bound=3, upper_bound=(1-q)*9+3*q,
                        wall_value=3, interfaces=(1,),
                    ),
                    source=lambda x, u, a: (1-q)*source(x, u, a)+q*3*strength*u,
                    source_u=lambda x, u, a: (1-q)*source_u(x, u, a)+q*3*strength,
                )
                path = self.homotopy(physical, strength)
                state = arithmetic.vector(["0.17", "-0.11", "0.08", "0.3", "0.2", "-0.1", "0.07"])
                self.assertEqual(tuple(path.residual(state, 0)), tuple(physical.residual(state)))
                self.assertEqual(arithmetic.norm(path.jacobian(state, 0)
                                                -physical.jacobian(state)), 0)
                self.assertLess(arithmetic.norm(path.residual(state, q)
                                                -intermediate.residual(state)),
                                self.tolerance(arithmetic))
                self.assertLess(arithmetic.norm(path.jacobian(state, q)
                                                -intermediate.jacobian(state)),
                                self.tolerance(arithmetic))
                for row in range(physical.basis.n_u):
                    self.assertLess(abs(path.residual(state, q)[row]
                                        -physical.residual(state)[row]),
                                    self.tolerance(arithmetic))
                self.assertEqual(tuple(physical.boundary_vector), tuple(intermediate.boundary_vector))

                direction = arithmetic.vector([1, -2, 3, -1, 2, -3, 1])
                digits = arithmetic.dps if arithmetic.use_mpmath else 15
                step = arithmetic.scalar("1e-{}".format(max(4, digits//3)))
                difference = (path.residual(state+step*direction, q)
                              -path.residual(state-step*direction, q))/(2*step)
                observed = self.matvec(arithmetic, path.jacobian(state, q), direction)
                relative = arithmetic.norm(difference-observed)/max(arithmetic.scalar(1),
                                                                     arithmetic.norm(observed))
                bound = arithmetic.scalar("3e-{}".format(max(7, 2*(digits//3)-4)))
                self.assertLess(relative, bound)


if __name__ == "__main__":
    unittest.main()
