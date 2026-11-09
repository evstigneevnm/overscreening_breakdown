"""Small physical mixed problems solved through the shared Newton driver.

The rational fixtures have no spatial approximation error at N=4.  Interface
cases are separate refinement studies: their global basis does not enforce
pointwise flux continuity at a finite resolution.
"""

from dataclasses import replace
import unittest

import mpmath as mp
import numpy as np

from arithmetic import MpmathArithmetic, NumpyArithmetic
from coefficients import Coefficient
from galerkin import GalerkinBasis
from galerkin_assembly import GalerkinAssembly
from nonlinear_solver import LinearSolveError, solve_newton
from physical_problem import FluxProblem
from quadrature import QuadratureFactory


def rational_problem(coefficient="constant", nonlinear=False):
    """Manufacture u=1/(1+x), w=-2/(1+x)^3, with a positive reaction."""
    constant = 3 if coefficient == "scaled" else 1

    def epsilon_and_derivative(x, arithmetic):
        reciprocal = arithmetic.scalar(1)/(1+x)
        if coefficient == "smooth":
            return 1+reciprocal, -reciprocal**2
        if coefficient == "exponential":
            exponential = arithmetic.exp(-x)
            return 1+exponential, -exponential
        return arithmetic.scalar(constant), arithmetic.scalar(0)

    def source(x, u, arithmetic):
        reciprocal = arithmetic.scalar(1)/(1+x)
        epsilon, derivative = epsilon_and_derivative(x, arithmetic)
        flux = -reciprocal**2+24*reciprocal**4
        flux_derivative = 2*reciprocal**3-96*reciprocal**5
        value = derivative*flux+epsilon*flux_derivative+(u-reciprocal)
        if nonlinear:
            value += arithmetic.scalar("0.125")*(u**3-reciprocal**3)
        return value

    def source_u(x, u, arithmetic):
        return (1+arithmetic.scalar("0.375")*u*u if nonlinear
                else arithmetic.scalar(1))

    epsilon = (Coefficient(
        lambda x, arithmetic: epsilon_and_derivative(x, arithmetic)[0],
        lower_bound=1, upper_bound=2, wall_value=2,
    ) if coefficient in ("smooth", "exponential") else constant)
    return FluxProblem(
        delta=2, epsilon=epsilon, f=source, f_u=source_u,
        boundary_conditions=((1, None, None, -6, None), (None,)*5),
    )


def transmission_fields(x, arithmetic):
    """Independent piecewise rational u, w, u', w' and continuous flux."""
    reciprocal = arithmetic.scalar(1)/(1+x)
    coefficients = ((arithmetic.scalar(1),) if x < 1 else
                    (arithmetic.scalar(17)/12, -arithmetic.scalar(5)/2,
                     arithmetic.scalar(5), -arithmetic.scalar(10)/3))
    zero = arithmetic.scalar(0)
    u = sum((coefficient*reciprocal**k for k, coefficient in
             enumerate(coefficients, 1)), zero)
    u_x = sum((-k*coefficient*reciprocal**(k+1) for k, coefficient in
               enumerate(coefficients, 1)), zero)
    w = sum((-k*(k+1)*coefficient*reciprocal**(k+2) for k, coefficient in
             enumerate(coefficients, 1)), zero)
    w_x = sum((k*(k+1)*(k+2)*coefficient*reciprocal**(k+3)
               for k, coefficient in enumerate(coefficients, 1)), zero)
    epsilon = arithmetic.scalar(1 if x < 1 else 2)
    return u, w, u_x, w_x, epsilon*(u_x+4*w_x)


def transmission_problem():
    def source(x, u, arithmetic):
        reciprocal = arithmetic.scalar(1)/(1+x)
        coefficients = ((arithmetic.scalar(1),) if x < 1 else
                        (arithmetic.scalar(17)/12, -arithmetic.scalar(5)/2,
                         arithmetic.scalar(5), -arithmetic.scalar(10)/3))
        epsilon = arithmetic.scalar(1 if x < 1 else 2)
        flux_derivative = epsilon*sum((
            coefficient*(k*(k+1)*reciprocal**(k+2)
                         -4*k*(k+1)*(k+2)*(k+3)*reciprocal**(k+4))
            for k, coefficient in enumerate(coefficients, 1)), arithmetic.scalar(0))
        exact = transmission_fields(x, arithmetic)[0]
        return flux_derivative+(u-exact)+arithmetic.scalar("0.125")*(u**3-exact**3)

    return FluxProblem(
        delta=2,
        epsilon=Coefficient(lambda x, arithmetic: arithmetic.scalar(1 if x < 1 else 2),
                            lower_bound=1, upper_bound=2, wall_value=1, interfaces=(1,)),
        f=source,
        f_u=lambda x, u, arithmetic: 1+arithmetic.scalar("0.375")*u*u,
        boundary_conditions=((1, None, None, -6, None), (None,)*5),
    )


class GalerkinManufacturedLightTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for precision in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(precision)

    def tolerance(self, arithmetic):
        digits = max(10, arithmetic.dps-8) if arithmetic.use_mpmath else 11
        return arithmetic.scalar("1e-{}".format(digits))

    def assembly(self, problem, arithmetic, N=4, order=None):
        basis = GalerkinBasis(N, arithmetic, L=1)
        interfaces = problem.bind(arithmetic).epsilon.interfaces
        plan = QuadratureFactory(arithmetic, L=1, interfaces=interfaces).plan(
            2*N if order is None else order)
        return GalerkinAssembly(problem, basis, plan)

    def solve(self, assembly, initial=None, globalization=False):
        result = solve_newton(
            assembly.nonlinear_system(initial), atol=self.tolerance(assembly.arithmetic),
            max_iterations=30, use_globalization=globalization,
        )
        self.assertTrue(result.converged, result.reason)
        self.assertEqual(result.homotopy, 0)
        self.assertLessEqual(assembly.arithmetic.norm(assembly.residual(result.state)),
                             self.tolerance(assembly.arithmetic))
        return result

    def check_rational_fields(self, assembly, state):
        arithmetic = assembly.arithmetic
        bound = 1000*self.tolerance(arithmetic)
        for coordinate in ("0", "0.125", "0.75", "1", "3.5", "12"):
            x = arithmetic.scalar(coordinate)
            u, w = assembly.basis.evaluate_fields(
                state, assembly.basis.mapping.to_reference(x), u_b=1)
            self.assertLess(abs(u-1/(1+x)), bound)
            self.assertLess(abs(w+2/(1+x)**3), bound)
            if arithmetic.use_mpmath:
                self.assertIsInstance(u, type(arithmetic.scalar(0)))
                self.assertIsInstance(w, type(arithmetic.scalar(0)))
            else:
                self.assertIsInstance(u, np.float64)
                self.assertIsInstance(w, np.float64)
        self.assertEqual(assembly.basis.evaluate_fields(state, 1, u_b=1), (0, 0))

    def test_linear_exact_space_solutions_for_all_precisions_and_coefficients(self):
        for arithmetic in self.arithmetics():
            for coefficient in ("constant", "scaled", "smooth"):
                with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps,
                                  epsilon=coefficient):
                    assembly = self.assembly(rational_problem(coefficient), arithmetic)
                    result = self.solve(assembly)
                    self.assertLessEqual(result.iterations, 2)
                    self.check_rational_fields(assembly, result.state)

    def test_nonlinear_exact_space_solutions_for_all_precisions_and_coefficients(self):
        for arithmetic in self.arithmetics():
            for coefficient in ("constant", "scaled", "smooth"):
                with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps,
                                  epsilon=coefficient):
                    assembly = self.assembly(rational_problem(coefficient, nonlinear=True), arithmetic)
                    initial = arithmetic.vector(["0.125", "0.0625", "-0.03125", 0, 0, 0, 0])
                    result = self.solve(assembly, initial, globalization=True)
                    self.assertGreater(result.iterations, 1)
                    self.check_rational_fields(assembly, result.state)

    def test_globalization_enabled_and_disabled_reach_the_physical_system(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            assembly = self.assembly(rational_problem("smooth", nonlinear=True), arithmetic)
            initial = arithmetic.vector(["0.125", 0, 0, 0, 0, 0, 0])
            plain = self.solve(assembly, initial, globalization=False)
            globalized = self.solve(assembly, initial, globalization=True)
            self.assertLess(arithmetic.norm(plain.state-globalized.state),
                            100*self.tolerance(arithmetic))

    def test_continuation_recovers_a_linear_failure_and_preserves_the_wall_lift(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            assembly = self.assembly(rational_problem("smooth", nonlinear=True), arithmetic)
            system = assembly.nonlinear_system()
            calls = [0]

            def fail_first(matrix, rhs):
                calls[0] += 1
                if calls[0] == 1:
                    raise LinearSolveError("forced first physical step failure")
                return system.solve_linear(matrix, rhs)

            accepted = []
            result = solve_newton(
                replace(system, solve_linear=fail_first),
                atol=self.tolerance(arithmetic), max_iterations=30,
                use_globalization=True, homotopy_cutoff="0.25",
                callback=lambda state, record: accepted.append(state),
            )
            self.assertTrue(result.converged, result.reason)
            self.assertEqual(result.homotopy, 0)
            self.assertEqual(result.stages[0].reason, "linear_solve_failed")
            self.assertTrue(any(stage.homotopy > 0 for stage in result.stages))
            self.assertTrue(accepted)
            for state in accepted:
                self.assertEqual(assembly.basis.evaluate_fields(state, -1, u_b=1)[0], 1)
            self.check_rational_fields(assembly, result.state)

    def test_quadrature_refinement_is_independent_of_nonlinear_convergence(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(30)):
            with self.subTest(backend=arithmetic.backend):
                coarse = self.assembly(rational_problem("exponential", nonlinear=True),
                                       arithmetic, order=5)
                result = self.solve(coarse)
                check = coarse.check_refinement(result.state, atol="1e-10", rtol="1e-10")
                self.assertFalse(check.converged)
                self.assertGreater(check.residual_difference, check.residual_target)
                # The same accepted fields initialize the newly assembled system.
                fine = check.refined_assembly
                fine_result = self.solve(fine, result.state)
                self.assertLess(arithmetic.norm(fine.residual(fine_result.state)),
                                self.tolerance(arithmetic))

    def test_runnable_example_reuses_one_problem_for_numpy_and_mpmath(self):
        from examples.flux_problem_galerkin import make_problem, solve_example

        problem = make_problem()
        precision = mp.mp.prec
        states = []
        for use_mpmath in (False, True):
            with self.subTest(mpmath=use_mpmath):
                assembly, result, check, error = solve_example(
                    problem, use_mpmath=use_mpmath, precision=50)
                self.assertTrue(result.converged)
                self.assertEqual(result.homotopy, 0)
                self.assertTrue(check.converged)
                self.assertLess(error, assembly.arithmetic.scalar(
                    "1e-35" if use_mpmath else "1e-9"))
                self.assertEqual(mp.mp.prec, precision)
                self.assertEqual(problem.delta, "2")
                self.assertEqual(problem.boundary_conditions[0][0], "1")
                self.check_rational_fields(assembly, result.state)
                states.append(np.array([float(value) for value in result.state]))
        np.testing.assert_allclose(states[0], states[1], rtol=1e-10, atol=1e-10)

    def test_existing_bsk_problem_binds_and_solves_with_variable_epsilon(self):
        from overscreening_breakdown import overscreening_breakdown

        epsilon = Coefficient(lambda x, arithmetic: 1+arithmetic.scalar(1)/(1+x),
                              lower_bound=1, upper_bound=2, wall_value=2)
        problem = overscreening_breakdown(epsilon=epsilon)
        problem.set_parameters(sigma=2, mu=0.1, gamma=0.5, delta=1,
                               u0=0.05, u0xxx=0)
        source_before = np.asarray(problem.physical_rhs(0.5, 0.03)).item()
        precision = mp.mp.prec
        samples = []
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(30)):
            with self.subTest(backend=arithmetic.backend):
                assembly = self.assembly(problem, arithmetic, N=12, order=28)
                result = self.solve(assembly, globalization=True)
                self.assertEqual(assembly.physical.delta, 1)
                self.assertEqual(assembly.epsilon_b, 2)
                self.assertAlmostEqual(float(assembly.physical.physical_rhs(
                    arithmetic.scalar(0.5), arithmetic.scalar(0.03))), source_before, places=14)
                self.assertEqual(assembly.basis.evaluate_fields(
                    result.state, -1, u_b=assembly.lift.u_b)[0], arithmetic.scalar(0.05))
                self.assertEqual(assembly.basis.evaluate_fields(
                    result.state, 1, u_b=assembly.lift.u_b), (0, 0))
                values = []
                for coordinate in ("0", "0.125", "1", "3", "8"):
                    y = assembly.basis.mapping.to_reference(arithmetic.scalar(coordinate))
                    fields = assembly.basis.evaluate_fields(result.state, y, u_b=assembly.lift.u_b)
                    self.assertTrue(all(arithmetic.isfinite(value) for value in fields))
                    values.extend(float(value) for value in fields)
                samples.append(np.array(values))
                self.assertEqual(mp.mp.prec, precision)
        np.testing.assert_allclose(samples[0], samples[1], rtol=1e-8, atol=1e-10)
        self.assertEqual(np.asarray(problem.physical_rhs(0.5, 0.03)).item(), source_before)

    def test_transmission_solution_improves_in_physical_field_and_flux_norms(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(30)):
            errors = []
            for N in (8, 16):
                with self.subTest(backend=arithmetic.backend, N=N):
                    assembly = self.assembly(transmission_problem(), arithmetic, N=N,
                                             order=2*N+4)
                    result = self.solve(assembly)
                    a, b = assembly.basis.split_coefficients(result.state)
                    # An independent plan checks physical L2(dx) errors.
                    validation = QuadratureFactory(arithmetic, interfaces=(1,)).plan(2*N+3)
                    squares = [arithmetic.scalar(0)]*3
                    for y, x, weight in zip(validation.nodes, validation.physical_nodes,
                                            validation.physical_weights):
                        u, w = assembly.basis.evaluate_fields(result.state, y, u_b=1)
                        u_x = assembly.lift.dx(y)+sum((entry*value for entry, value in
                                                     zip(a, assembly.basis.v_dx(y))), arithmetic.scalar(0))
                        w_x = sum((entry*value for entry, value in zip(b, assembly.basis.q_dx(y))),
                                  arithmetic.scalar(0))
                        epsilon = arithmetic.scalar(1 if x < 1 else 2)
                        flux = epsilon*(u_x+4*w_x)
                        exact_u, exact_w, _, _, exact_flux = transmission_fields(x, arithmetic)
                        for index, difference in enumerate((u-exact_u, w-exact_w, flux-exact_flux)):
                            squares[index] += weight*difference*difference
                    errors.append(tuple(arithmetic.sqrt(value) for value in squares))
            # Bounds are calibrated for this C2 potential and H1 second field;
            # smooth-problem spectral rates would not apply at the interface.
            for (coarse, fine), absolute_bound in zip(
                    zip(*errors), ("0.012", "0.008", "0.25")):
                self.assertLess(fine, arithmetic.scalar("0.8")*coarse)
                self.assertLess(fine, arithmetic.scalar(absolute_bound))


if __name__ == "__main__":
    unittest.main()
