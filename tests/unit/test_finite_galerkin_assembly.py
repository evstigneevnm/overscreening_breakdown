"""Finite weak boundary loads, exact polynomial pairs and shared Newton links."""

from dataclasses import replace
import unittest

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.continuation import AffineReference, ContinuationOptions, solve_continuation
from overscreening_breakdown.galerkin.finite_assembly import FiniteGalerkinAssembly
from overscreening_breakdown.galerkin.finite_basis import FiniteGalerkinBasis
from overscreening_breakdown.nonlinear_solver import LinearSolveError, solve_newton
from overscreening_breakdown.problems.flux import FluxProblem
from overscreening_breakdown.quadrature import QuadratureFactory


def zero_source(x, u, a):
    return a.scalar(0)


def quartic_values(x, a, R="2"):
    R = a.scalar(R)
    t = x/R
    return (1+2*t-t*t+a.scalar(1)/2*t**3+t**4/3,
            (2-2*t+a.scalar(3)/2*t*t+a.scalar(4)/3*t**3)/R,
            (-2+3*t+4*t*t)/R**2, (3+8*t)/R**3, a.scalar(8)/R**4)


def quartic_problem(*, epsilon=1, nonlinear=True):
    def source(x, u, a):
        values = quartic_values(x, a)
        eps = 1+x/2 if epsilon == "ramp" else a.scalar(epsilon)
        eps_dx = a.scalar(1)/2 if epsilon == "ramp" else a.scalar(0)
        forcing = eps_dx*(values[1]-4*values[3])+eps*(values[2]-4*values[4])
        reaction = (u-values[0])
        return forcing+reaction+(a.scalar("0.125")*(u**3-values[0]**3) if nonlinear else 0)

    coefficient = (Coefficient(lambda x, a: 1+x/2, lower_bound=1, upper_bound=2, wall_value=1)
                   if epsilon == "ramp" else epsilon)
    return FluxProblem(
        delta=2, epsilon=coefficient, domain=(0, 2), f=source,
        f_u=lambda x, u, a: 1+(a.scalar("0.375")*u*u if nonlinear else 0),
        boundary_conditions=((1, None, None, "0.375", None), (0, 0, None, None, None)),
    )


def exact_problem(a, **options):
    # Bind the endpoint rationals in the intended arithmetic; decimal truncation
    # in a factory must not determine the exact-space precision ceiling.
    problem = quartic_problem(**options)
    return replace(problem, boundary_conditions=((a.scalar(1), None, None, a.scalar(3)/8, None),
                                                (a.scalar(17)/6, a.scalar(17)/12, None, None, None)))


class FiniteGalerkinAssemblyTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for digits in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(digits)

    def tolerance(self, a):
        return a.scalar("1e-10" if not a.use_mpmath else "1e-{}".format(max(9, a.dps-5)))

    def assemble(self, a, problem=None, N=4, order=12, interfaces=None):
        if problem is None:
            problem = exact_problem(a)
        basis = FiniteGalerkinBasis(N, a, domain=problem.domain)
        splits = problem.bind(a).epsilon.interfaces if interfaces is None else interfaces
        plan = QuadratureFactory(a, domain=problem.domain, interfaces=splits).plan(order)
        return FiniteGalerkinAssembly(problem, basis, plan)

    def exact_state(self, a):
        return a.vector((a.scalar(-3)/32, a.scalar(-7)/48, a.scalar(-1)/96,
                         a.scalar(-1)/4, a.scalar(-7)/8, a.scalar(-1)/8, 0, 0))

    def matvec(self, matrix, vector):
        return matrix*vector if hasattr(matrix, "rows") else matrix.dot(vector)

    def test_exact_quartic_both_blocks_all_precisions_and_nonzero_right_curvature(self):
        for a in self.arithmetics():
            for epsilon in (1, 3, "ramp"):
                with self.subTest(backend=a.backend, digits=a.dps, epsilon=epsilon):
                    assembly = self.assemble(a, exact_problem(a, epsilon=epsilon))
                    state = self.exact_state(a)
                    self.assertLess(a.norm(assembly.residual(state)), self.tolerance(a))
                    self.assertEqual(len(assembly.residual(state)), 8)
                    u, w = assembly.basis.evaluate_fields(state, 1, 1, a.scalar(17)/6)
                    self.assertLess(abs(u-a.scalar(17)/6), self.tolerance(a))
                    self.assertLess(abs(w+a.scalar(5)/4), self.tolerance(a))
                    self.assertTrue(all(isinstance(value, type(a.scalar(0))) for value in assembly.residual(state)))

    def test_lowest_matrix_modes_match_independent_polynomial_moments(self):
        for a in self.arithmetics():
            with self.subTest(backend=a.backend, digits=a.dps):
                p = FluxProblem(delta=2, domain=(0, 3), f=zero_source, f_u=zero_source,
                                boundary_conditions=((0, None, None, 0, None), (0, 0, None, None, None)))
                assembly = self.assemble(a, p, N=2, order=4)
                R = a.scalar(3)
                targets = ((0, 0, 0), (1, 0, -a.scalar(16)/(3*R)),
                           (0, 1, -R), (0, 2, -R/3), (0, 3, R/3),
                           (2, 1, R), (2, 2, -16/R-R/3), (3, 3, -128/(3*R)-R/15))
                matrix = assembly.linear_matrix
                for row, column, target in targets:
                    self.assertLess(abs(matrix[row, column]-target), self.tolerance(a))

    def test_both_natural_boundary_loads_and_two_end_lift_have_correct_signs(self):
        for a in self.arithmetics():
            with self.subTest(backend=a.backend, digits=a.dps):
                p = FluxProblem(delta=2, epsilon=3, domain=(0, 3), f=zero_source, f_u=zero_source,
                                boundary_conditions=((2, None, None, 7, None), (5, 11, None, None, None)))
                assembly = self.assemble(a, p, N=2, order=4)
                residual = assembly.linear_residual(a.zeros(4))
                for value, target in zip(residual, (-20, -20, -56, 56)):
                    self.assertLess(abs(value-target), self.tolerance(a))

    def test_jacobian_direction_at_non_solution_state_uses_distinct_test_spaces(self):
        for a in self.arithmetics():
            with self.subTest(backend=a.backend, digits=a.dps):
                assembly = self.assemble(a, exact_problem(a, epsilon="ramp"))
                state = self.exact_state(a)+a.vector(a.scalar(k+1)/100 for k in range(8))
                direction = a.vector(a.scalar((-1)**k*(k+1))/10 for k in range(8))
                h = a.scalar("0.125")
                difference = (assembly.residual(state-2*h*direction)-8*assembly.residual(state-h*direction)
                              +8*assembly.residual(state+h*direction)-assembly.residual(state+2*h*direction))/(12*h)
                product = self.matvec(assembly.jacobian(state), direction)
                self.assertLess(a.norm(product-difference), 20*self.tolerance(a))
                with self.assertRaises(ValueError):
                    assembly.residual(state, q=a.scalar("0.5"))

    def test_shared_newton_and_actual_reference_recovery_preserve_finite_geometry(self):
        a = MpmathArithmetic(30)
        assembly = self.assemble(a, exact_problem(a, epsilon="ramp"))
        direct = solve_newton(assembly.nonlinear_system(), atol="1e-22", max_iterations=12)
        self.assertTrue(direct.converged)
        self.assertLess(a.norm(direct.state-self.exact_state(a)), a.scalar("1e-20"))
        system, calls = assembly.nonlinear_system(), []
        original = system.solve_linear

        def fail_once(matrix, rhs):
            calls.append(1)
            if len(calls) == 1:
                raise LinearSolveError("controlled initial failure")
            return original(matrix, rhs)

        result = solve_continuation(
            replace(system, solve_linear=fail_once), reference_factory=assembly.reference_operator,
            options=ContinuationOptions(initial_step="0.5", max_step="0.5"),
            atol="1e-22", max_iterations=12,
        )
        self.assertTrue(result.converged)
        self.assertTrue(result.reference_result.converged)
        self.assertTrue(any(0 < stage.homotopy < 1 for stage in result.stages))
        self.assertEqual(result.homotopy, 0)
        self.assertEqual(assembly.physical.domain, (a.scalar(0), a.scalar(2)))
        self.assertLess(a.norm(assembly.residual(result.state)), a.scalar("1e-22"))

    def test_reference_operator_matches_a_full_constant_epsilon_finite_problem(self):
        a = MpmathArithmetic(50)
        coefficient = Coefficient(lambda x, arithmetic: 3+x/2,
                                  lower_bound=3, upper_bound=4, wall_value=3)

        def forbidden_source(x, u, arithmetic):
            raise AssertionError("reference assembly must not evaluate the physical source")

        p = replace(exact_problem(a), epsilon=coefficient, f=forbidden_source, f_u=forbidden_source)
        variable = self.assemble(a, p)
        reference_problem = replace(p, epsilon=3, f=lambda x, u, arithmetic: 6*u,
                                    f_u=lambda x, u, arithmetic: arithmetic.scalar(6))
        constant = self.assemble(a, reference_problem)
        matrix, offset = variable.reference_operator(2)
        state = a.vector(a.scalar(k)/10 for k in range(8))
        self.assertLess(a.norm(self.matvec(matrix, state)+offset-constant.residual(state)), a.scalar("1e-45"))
        reference = AffineReference(matrix, offset, variable.nonlinear_system())
        self.assertLess(a.norm(reference.residual(reference.solve())), a.scalar("1e-45"))

    def test_quadrature_refinement_retains_snapshot_and_both_finite_blocks(self):
        a = MpmathArithmetic(30)
        coefficient = Coefficient(lambda x, arithmetic: 1+arithmetic.scalar("0.1")*arithmetic.exp(-x),
                                  lower_bound=1, upper_bound="1.1", wall_value="1.1")
        p = FluxProblem(delta=2, epsilon=coefficient, domain=(0, 2), f=zero_source, f_u=zero_source,
                        boundary_conditions=((2, None, None, 0, None), (1, "0.25", None, None, None)))
        assembly = self.assemble(a, p, order=5)
        check = assembly.check_refinement(a.vector(range(8)), atol="1e-25", rtol=0)
        self.assertFalse(check.converged)
        self.assertIsInstance(check.refined_assembly, FiniteGalerkinAssembly)
        self.assertIs(check.refined_assembly.physical, assembly.physical)
        self.assertEqual(check.refined_assembly.plan.domain, assembly.physical.domain)
        self.assertEqual(check.refined_assembly.unknown_count, 8)

    def test_rejects_unsplit_interfaces_and_unsupported_boundary_rows(self):
        a = NumpyArithmetic()
        coefficient = Coefficient(lambda x, arithmetic: arithmetic.scalar(1 if x < .7 else 2),
                                  lower_bound=1, upper_bound=2, wall_value=1, interfaces=("0.7",))
        p = replace(exact_problem(a), epsilon=coefficient)
        with self.assertRaisesRegex(ValueError, "split every"):
            self.assemble(a, p, interfaces=())
        self.assemble(a, p)
        for boundaries in (((1, None, None, None, None), (0, 0, None, None, None)),
                           ((1, None, None, 0, None), (0, None, None, None, None)),
                           ((1, 0, None, 0, None), (0, 0, None, None, None)),
                           ((1, None, None, 0, None), (0, 0, 0, None, None))):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                self.assemble(a, replace(p, boundary_conditions=boundaries))
        with self.assertRaises(ValueError):
            self.assemble(a, replace(p, delta=0))
