"""Merit derivatives, homogeneous trace geometry and native adjoint products."""

import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.collocation.globalization import CollocationMeritGeometry
from overscreening_breakdown.continuation import AffineReference, ResidualHomotopy
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.manufactured import rational_case
from overscreening_breakdown.quadrature import QuadratureFactory


class AdjointGeometryTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for digits in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(digits)

    def tolerance(self, arithmetic):
        return arithmetic.scalar("2e-12" if not arithmetic.use_mpmath else
                                 "2e-{}".format(max(11, arithmetic.dps-5)))

    def assert_native(self, arithmetic, values):
        for value in values:
            if arithmetic.use_mpmath:
                self.assertTrue(hasattr(value, "_mpf_"))
                self.assertNotIsInstance(value, (float, np.floating))
            else:
                self.assertIsInstance(value, np.float64)

    def assert_directional_derivative(self, arithmetic, merit, state, direction, gradient):
        exponent = arithmetic.dps//3 if arithmetic.use_mpmath else 5
        h = arithmetic.scalar("1e-{}".format(exponent))
        difference = (merit(state+h*direction)-merit(state-h*direction))/(2*h)
        derivative = arithmetic.dot(gradient, direction)
        tolerance = arithmetic.scalar("3e-{}".format(max(8, 2*exponent-2)))
        self.assertLess(abs(difference-derivative), tolerance*(1+abs(derivative)))

    def test_native_dot_and_transpose_satisfy_the_adjoint_identity(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                matrix = arithmetic.matrix([["1.25", "-2", "0.125"], ["3", "0.5", "-1"]])
                vector = arithmetic.vector(["0.5", "-0.75", "1.25"])
                dual = arithmetic.vector(["-0.25", "0.875"])
                image = arithmetic.vector([
                    arithmetic.scalar("1.25")*vector[0]-2*vector[1]+vector[2]/8,
                    3*vector[0]+vector[1]/2-vector[2],
                ])
                adjoint = arithmetic.transpose_matvec(matrix, dual)
                self.assertLess(abs(arithmetic.dot(image, dual)-arithmetic.dot(vector, adjoint)),
                                self.tolerance(arithmetic))
                self.assert_native(arithmetic, (*adjoint, arithmetic.dot(vector, vector)))
                self.assertEqual(arithmetic.dot([], []), 0)

    def test_native_products_reject_mismatched_and_nonvector_shapes(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(30)):
            with self.subTest(backend=arithmetic.backend):
                with self.assertRaises(ValueError):
                    arithmetic.dot([1], [1, 2])
                with self.assertRaises(ValueError):
                    arithmetic.dot(arithmetic.matrix([[1, 2], [3, 4]]), [1, 2, 3, 4])
                with self.assertRaises(ValueError):
                    arithmetic.transpose_matvec([[1, 2]], [1, 2])
                with self.assertRaises(TypeError):
                    arithmetic.dot([1j], [1])

    def test_collocation_rms_merit_gradient_matches_an_independent_derivative(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                scalar = arithmetic.scalar
                geometry = CollocationMeritGeometry(arithmetic, (), 3)
                state = arithmetic.vector(["0.25", "-0.5", "0.75"])
                direction = arithmetic.vector(["0.125", "0.25", "-0.375"])
                q = scalar("0.625")

                def residual(z):
                    return arithmetic.vector([z[0]**2+z[1]*z[2]+q,
                                              arithmetic.sin(z[0])+2*z[1]-q*z[2],
                                              z[0]*z[1]+z[2]**3-q])

                jacobian = arithmetic.matrix([
                    [2*state[0], state[2], state[1]],
                    [arithmetic.cos(state[0]), 2, -q],
                    [state[1], state[0], 3*state[2]**2],
                ])
                gradient = geometry.merit_gradient(residual(state), jacobian)
                self.assert_directional_derivative(
                    arithmetic, lambda z: arithmetic.dot(residual(z), residual(z))/6,
                    state, direction, gradient)
                self.assert_native(arithmetic, gradient)

    def test_normalized_projection_preserves_nonzero_traces_and_is_descent(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                big, small = arithmetic.scalar("1e200"), arithmetic.scalar("1e-200")
                rows = arithmetic.matrix([[big, 2*big, 0, 0], [0, 0, small, -small]])
                geometry = CollocationMeritGeometry(arithmetic, rows, 4)
                gradient = arithmetic.vector([1, -2, 3, 4])
                direction = geometry.adjoint_direction(gradient)
                state = arithmetic.vector([2, 1, -1, 0])
                candidate = state+arithmetic.scalar("0.125")*direction
                for row in ((1, 2, 0, 0), (0, 0, 1, -1)):
                    self.assertLess(abs(arithmetic.dot(row, direction)), self.tolerance(arithmetic))
                    self.assertLess(abs(arithmetic.dot(row, candidate)-arithmetic.dot(row, state)),
                                    self.tolerance(arithmetic))
                self.assertLess(geometry.dot(gradient, direction), 0)
                self.assertLess(abs(geometry.dot(gradient, direction)+geometry.dot(direction, direction)),
                                100*self.tolerance(arithmetic))
                self.assert_native(arithmetic, direction)

    def test_actual_collocation_tau_rows_define_the_homogeneous_trace_space(self):
        from overscreening_breakdown.collocation.spectral1d import collocation_discretization

        original_bits = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_bits)
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                discretization = collocation_discretization(
                    6, domain=[0, np.inf], L=1, use_mpmath=arithmetic.use_mpmath,
                    prec=arithmetic.dps)
                discretization.set_boundary_conditions(
                    [[1, None, None, -6, None], [0, None, None, None, None]])
                tau = discretization.set_boundary_bilinear_form_tau_method(
                    discretization.bilinear_form(discretization.psi, 0))
                size = tau.rows if arithmetic.use_mpmath else tau.shape[0]
                rows = [tuple(arithmetic.scalar(tau[i, j]) for j in range(size))
                        for i in range(size) if any(tau[i, j] != 0 for j in range(size))]
                self.assertEqual(len(rows), 3)
                geometry = CollocationMeritGeometry(arithmetic, rows, size)
                gradient = arithmetic.vector(arithmetic.scalar(index+1)/(index+2)
                                             for index in range(size))
                direction = geometry.adjoint_direction(gradient)
                for row in rows:
                    scale = max(abs(value) for value in row)
                    scaled = tuple(value/scale for value in row)
                    self.assertLess(abs(arithmetic.dot(scaled, direction)),
                                    100*self.tolerance(arithmetic))
                self.assertLess(geometry.dot(gradient, direction), 0)

    def test_full_trace_space_and_empty_trace_space_have_expected_stationarity(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                gradient = arithmetic.vector([3, -4])
                fixed = CollocationMeritGeometry(arithmetic, [[1, 0], [0, 1]], 2)
                free = CollocationMeritGeometry(arithmetic, (), 2)
                self.assertEqual(arithmetic.norm(fixed.adjoint_direction(gradient)), 0)
                self.assertEqual(arithmetic.norm(free.adjoint_direction(gradient)+gradient), 0)

    def test_geometry_preserves_sub_double_information_and_context_isolation(self):
        original_bits = mp.mp.prec
        high, low = MpmathArithmetic(120), MpmathArithmetic(30)
        correction = high.scalar("1e-80")
        perturbed = CollocationMeritGeometry(high, [[1, 1+correction, 0]], 3)
        plain = CollocationMeritGeometry(high, [[1, 1, 0]], 3)
        low_geometry = CollocationMeritGeometry(low, [[1, 1+correction, 0]], 3)
        gradient = high.vector([1, -1, 2])
        difference = high.norm(perturbed.adjoint_direction(gradient)-plain.adjoint_direction(gradient))
        self.assertGreater(difference, high.scalar("1e-81"))
        self.assertLess(difference, high.scalar("1e-78"))
        self.assertEqual(low.norm(low_geometry.adjoint_direction([1, -1, 2])
                                  +low.vector([1, -1, 2])), 0)
        self.assertEqual(mp.mp.prec, original_bits)
        high.context.prec += 1
        for callback in (lambda: perturbed.adjoint_direction(gradient),
                         lambda: perturbed.dot(gradient, gradient),
                         lambda: perturbed.merit_gradient(gradient, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
            with self.assertRaises(RuntimeError):
                callback()

    def test_mixed_merit_uses_both_independent_fields_and_conforming_directions(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                basis = GalerkinBasis(4, arithmetic, L=1)
                plan = QuadratureFactory(arithmetic).plan(8)
                assembly = GalerkinAssembly(rational_case("smooth", nonlinear=True).make_problem(), basis, plan)
                state = arithmetic.vector(["0.125", "-0.0625", "0.03125", "-0.3", "0.4", "-0.1", "0.05"])
                variation = arithmetic.vector(["0.1", "-0.2", "0.05", "0.25", "-0.125", "0.1", "-0.05"])
                q = arithmetic.scalar("0.75")
                physical = assembly.nonlinear_system(state)
                matrix, offset = assembly.reference_operator()
                system = ResidualHomotopy(
                    physical, AffineReference(matrix, offset, physical)).nonlinear_system(state)
                gradient = system.merit_gradient(system.residual(state, q), system.jacobian(state, q))
                self.assertGreater(arithmetic.norm(arithmetic.vector(tuple(gradient)[basis.n_u:])), 0)
                self.assert_directional_derivative(
                    arithmetic, lambda z: system.residual_norm(system.residual(z, q))**2/2,
                    state, variation, gradient)
                direction = system.adjoint_direction(gradient)
                self.assertEqual(arithmetic.norm(direction+gradient), 0)
                candidate = state+arithmetic.scalar("0.03125")*direction
                self.assertEqual(basis.evaluate_fields(candidate, y=-1, u_b=1)[0], 1)
                self.assertEqual(basis.evaluate_fields(candidate, y=1, u_b=1), (0, 0))
                self.assert_native(arithmetic, gradient)


if __name__ == "__main__":
    unittest.main()
