"""Mixed weak assembly, physical normalization, and analytic linearization.

The exact rational fixture lies in the N=4 mixed space.  Independent polynomial
moments and physical identities detect wrong signs, Jacobian transposes, wall
loads and float64 leakage without reproducing the assembly loops.
"""

from dataclasses import FrozenInstanceError, replace
import unittest

import mpmath as mp
import numpy as np

from arithmetic import MpmathArithmetic, NumpyArithmetic
from coefficients import Coefficient
from galerkin import GalerkinBasis
from galerkin_assembly import GalerkinAssembly
from nonlinear_solver import solve_newton
from physical_problem import FluxProblem
from quadrature import QuadratureFactory


def zero_source(x, u, arithmetic):
    return arithmetic.scalar(0)


def rational_problem(a=1, b=0, nonlinear=True):
    """u=s, w=-2s³, epsilon=a+b*s, s=1/(1+x), delta=2."""
    def epsilon(x, arithmetic):
        return arithmetic.scalar(a)+arithmetic.scalar(b)/(1+x)

    def source(x, u, arithmetic):
        s = arithmetic.scalar(1)/(1+x)
        first, second = arithmetic.scalar(a), arithmetic.scalar(b)
        flux_derivative = 2*first*s**3+3*second*s**4-96*first*s**5-120*second*s**6
        return (flux_derivative+(u-s)+arithmetic.scalar("0.125")*(u**3-s**3)
                if nonlinear else flux_derivative)

    def source_u(x, u, arithmetic):
        return 1+arithmetic.scalar("0.375")*u*u if nonlinear else arithmetic.scalar(0)

    coefficient = (a if b == 0 else Coefficient(
        epsilon, lower_bound=a, upper_bound=a+b, wall_value=a+b))
    return FluxProblem(
        delta=2, epsilon=coefficient, f=source, f_u=source_u,
        boundary_conditions=((1, None, None, -6, None), (None,)*5),
    )


class GalerkinAssemblyTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for precision in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(precision)

    def tolerance(self, arithmetic):
        return arithmetic.scalar("3e-11" if not arithmetic.use_mpmath
                                 else "3e-{}".format(max(10, arithmetic.dps-5)))

    def problem(self, **options):
        arguments = dict(
            delta=2, f=zero_source, f_u=zero_source,
            boundary_conditions=((0, None, None, 0, None), (None,)*5),
        )
        arguments.update(options)
        return FluxProblem(**arguments)

    def assembly(self, arithmetic, problem=None, N=4, L=1, order=None,
                 interfaces=None):
        problem = self.problem() if problem is None else problem
        if interfaces is None:
            interfaces = problem.bind(arithmetic).epsilon.interfaces
        basis = GalerkinBasis(N, arithmetic, L=L)
        plan = QuadratureFactory(arithmetic, L=L, interfaces=interfaces).plan(
            N+1 if order is None else order)
        return GalerkinAssembly(problem, basis, plan)

    def exact_state(self, arithmetic):
        return arithmetic.vector([0, 0, 0, "-0.375", "0.5", "-0.125", 0])

    def assert_native(self, arithmetic, values):
        for value in values:
            if arithmetic.use_mpmath:
                self.assertTrue(hasattr(value, "_mpf_"))
                self.assertNotIsInstance(value, (float, np.floating))
            else:
                self.assertIsInstance(value, np.float64)

    def test_exact_N4_pair_satisfies_both_weak_blocks_in_every_precision(self):
        for arithmetic in self.arithmetics():
            for a, b in ((1, 0), (3, 0), (1, 1)):
                with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps,
                                  epsilon=(a, b)):
                    assembly = self.assembly(arithmetic, rational_problem(a, b))
                    state = self.exact_state(arithmetic)
                    residual = assembly.residual(state)
                    self.assertEqual(len(residual), 7)
                    self.assertLess(arithmetic.norm(residual), self.tolerance(arithmetic))
                    self.assert_native(arithmetic, residual)
                    self.assert_native(arithmetic, (assembly.jacobian(state)[i, j]
                                                    for i in range(7) for j in range(7)))

    def test_lowest_modes_have_independent_exact_physical_moments(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                assembly = self.assembly(arithmetic, L="1.5")
                length = arithmetic.scalar("1.5")
                matrix = assembly.linear_matrix
                n_u = assembly.basis.n_u
                expected = ((0, 0, 32/(15*length)),
                            (0, n_u, -4*length),
                            (n_u, n_u, 16/(3*length)+4*length))
                for row, col, target in expected:
                    self.assertLess(abs(matrix[row, col]-target), self.tolerance(arithmetic))

    def test_source_derivative_cross_block_has_the_weak_sign_and_transpose(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                problem = self.problem(
                    epsilon=3, f=lambda x, u, a: 6*u,
                    f_u=lambda x, u, a: a.scalar(6),
                )
                assembly = self.assembly(arithmetic, problem)
                matrix = assembly.jacobian(arithmetic.zeros(7))
                n_u = assembly.basis.n_u
                for i in range(n_u):
                    for j in range(assembly.basis.n_w):
                        self.assertLess(abs(matrix[n_u+j, i]+2*matrix[i, n_u+j]),
                                        self.tolerance(arithmetic))

    def test_affine_lift_and_variable_coefficient_coupling_have_correct_signs(self):
        for arithmetic in self.arithmetics():
            for variable in (False, True):
                with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps,
                                  variable=variable):
                    epsilon = Coefficient(
                        lambda x, a: 1+a.scalar(1)/(1+x), lower_bound=1,
                        upper_bound=2, wall_value=2,
                    ) if variable else 3
                    problem = self.problem(
                        epsilon=epsilon,
                        boundary_conditions=((1, None, None, 0, None), (None,)*5),
                    )
                    assembly = self.assembly(arithmetic, problem)
                    residual = assembly.linear_residual(arithmetic.zeros(7))
                    self.assertLess(abs(residual[0]+arithmetic.scalar(2)/3),
                                    self.tolerance(arithmetic))
                    target = -arithmetic.scalar(1)/12 if variable else arithmetic.scalar(0)
                    self.assertLess(abs(residual[3]-target), self.tolerance(arithmetic))
                    if variable:
                        self.assertLess(abs(assembly.linear_matrix[3, 0]-arithmetic.scalar(1)/15),
                                        self.tolerance(arithmetic))

    def test_natural_wall_load_uses_physical_q_b_and_normalized_trace(self):
        for arithmetic in self.arithmetics():
            for epsilon in (1, 3):
                with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps,
                                  epsilon=epsilon):
                    problem = self.problem(
                        epsilon=epsilon,
                        boundary_conditions=((0, None, None, "1.25", None), (None,)*5),
                    )
                    assembly = self.assembly(arithmetic, problem)
                    residual = assembly.residual(arithmetic.zeros(7))
                    for value in tuple(residual)[:3]:
                        self.assertEqual(value, 0)
                    for k in range(4):
                        # -delta²*q_b*Q_k(wall), Q_k(wall)=2*(-1)^k.
                        self.assertEqual(residual[3+k], -10*(-1)**k)

    def test_rescaling_epsilon_and_physical_source_preserves_normalized_system(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                def problem(scale):
                    coefficient = Coefficient(
                        lambda x, a: scale*(1+a.scalar(1)/(1+x)),
                        lower_bound=scale, upper_bound=2*scale, wall_value=2*scale,
                    )
                    return self.problem(
                        epsilon=coefficient,
                        f=lambda x, u, a: scale*(a.exp(-x)+u+u**3),
                        f_u=lambda x, u, a: scale*(1+3*u*u),
                        boundary_conditions=(("1.5", None, None, "0.25", None), (None,)*5),
                    )
                first = self.assembly(arithmetic, problem(1))
                second = self.assembly(arithmetic, problem(7))
                state = arithmetic.vector(["0.1", "-0.03", "0.02", "0.5", "0.1", "-0.2", "0.1"])
                for parameter in (0, arithmetic.scalar("0.375")):
                    self.assertLess(arithmetic.norm(first.residual(state, parameter)
                                                    -second.residual(state, parameter)),
                                    self.tolerance(arithmetic))
                    self.assertLess(arithmetic.norm(first.jacobian(state, parameter)
                                                    -second.jacobian(state, parameter)),
                                    self.tolerance(arithmetic))

    def test_nonlinear_jacobian_direction_at_non_solution_and_nonzero_homotopy(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                assembly = self.assembly(arithmetic, rational_problem(1, 1))
                state = arithmetic.vector(["0.17", "-0.11", "0.08", "0.3", "0.2", "-0.1", "0.07"])
                direction = arithmetic.vector([1, -2, 3, -1, 2, -3, 1])
                digits = arithmetic.dps if arithmetic.use_mpmath else 15
                step = arithmetic.scalar("1e-{}".format(max(4, digits//3)))
                parameter = arithmetic.scalar("0.375")
                difference = (assembly.residual(state+step*direction, parameter)
                              -assembly.residual(state-step*direction, parameter))/(2*step)
                matrix = assembly.jacobian(state, parameter)
                observed = matrix*direction if arithmetic.use_mpmath else matrix @ direction
                relative = arithmetic.norm(difference-observed)/max(arithmetic.scalar(1), arithmetic.norm(observed))
                bound = arithmetic.scalar("3e-{}".format(max(7, 2*(digits//3)-4)))
                self.assertLess(relative, bound)
                # Homotopy acts on the physical source, leaving the first
                # equation and fixed operator blocks unchanged.
                physical = assembly.residual(state, 0)
                damped = assembly.residual(state, parameter)
                for i in range(3):
                    self.assertEqual(physical[i], damped[i])
                self.assertGreater(arithmetic.norm(physical-damped), 0)

    def test_fixed_operator_and_sources_reuse_only_interior_interface_nodes(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            calls = {name: [] for name in ("epsilon", "f", "f_u")}

            def coefficient(x, a):
                self.assertNotEqual(x, 0)
                self.assertNotEqual(x, 1)
                self.assertTrue(a.isfinite(x))
                calls["epsilon"].append(x)
                return a.scalar(1 if x < 1 else 2)

            def source(x, u, a):
                self.assertNotIn(x, (0, 1))
                self.assertTrue(a.isfinite(x))
                calls["f"].append(x)
                return u+u**3

            def source_u(x, u, a):
                calls["f_u"].append(x)
                return 1+3*u*u

            description = Coefficient(coefficient, lower_bound=1, upper_bound=2,
                                      wall_value=1, interfaces=(1,))
            problem = self.problem(epsilon=description, f=source, f_u=source_u)
            assembly = self.assembly(arithmetic, problem)
            fixed_calls = len(calls["epsilon"])
            self.assertEqual(fixed_calls, len(assembly.plan.nodes))
            state = arithmetic.zeros(7)
            assembly.residual(state)
            assembly.jacobian(state)
            assembly.residual(state, arithmetic.scalar("0.25"))
            self.assertEqual(len(calls["epsilon"]), fixed_calls)
            for name in ("f", "f_u"):
                self.assertTrue(calls[name])
                self.assertEqual(set(calls[name]), set(assembly.plan.physical_nodes))

    def test_linear_system_recovers_exact_rational_fields(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(100)):
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                assembly = self.assembly(arithmetic, rational_problem(nonlinear=False))
                rhs = -assembly.residual(arithmetic.zeros(7))
                state = arithmetic.solve(assembly.linear_matrix, rhs)
                self.assertLess(arithmetic.norm(state-self.exact_state(arithmetic)),
                                self.tolerance(arithmetic))
                for coordinate in ("0", "0.125", "1", "4.75"):
                    x = arithmetic.scalar(coordinate)
                    u, w = assembly.basis.evaluate_fields(
                        state, assembly.basis.mapping.to_reference(x), u_b=1)
                    self.assertLess(abs(u-arithmetic.scalar(1)/(1+x)), self.tolerance(arithmetic))
                    self.assertLess(abs(w+arithmetic.scalar(2)/(1+x)**3), self.tolerance(arithmetic))

    def test_returned_linear_arrays_are_independent_copies(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            assembly = self.assembly(arithmetic, rational_problem())
            state = self.exact_state(arithmetic)
            expected = assembly.residual(state)
            matrix = assembly.linear_matrix
            matrix[0, 0] += 99
            for name in ("lift_vector", "boundary_vector", "affine_vector"):
                vector = getattr(assembly, name)
                vector[0] += 99
                self.assertNotEqual(vector[0], getattr(assembly, name)[0])
            self.assertLess(arithmetic.norm(assembly.residual(state)-expected), self.tolerance(arithmetic))
            self.assertNotEqual(matrix[0, 0], assembly.linear_matrix[0, 0])
            for name in ("basis", "plan", "physical", "epsilon_b", "delta_squared"):
                with self.assertRaises(FrozenInstanceError):
                    setattr(assembly, name, None)

    def test_assembly_rejects_unsupported_problems_and_already_bound_views(self):
        arithmetic = NumpyArithmetic()
        basis = GalerkinBasis(4, arithmetic)
        plan = QuadratureFactory(arithmetic).plan(5)
        invalid = (
            self.problem(delta=0), self.problem(domain=(0, 5)),
            self.problem(boundary_conditions=((None, None, None, 0, None), (None,)*5)),
            self.problem(boundary_conditions=((0, None, None, None, None), (None,)*5)),
            self.problem(boundary_conditions=((0, 0, None, 0, None), (None,)*5)),
            self.problem(boundary_conditions=((0, None, None, 0, None), (1, None, None, None, None))),
            self.problem(boundary_conditions=((0, None, None, 0, None), (0, 0, None, None, None))),
        )
        for problem in invalid:
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                GalerkinAssembly(problem, basis, plan)
        with self.assertRaises(TypeError):
            GalerkinAssembly(self.problem().bind(arithmetic), basis, plan)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            for delta in ("1e200", "1e-200"):
                with self.subTest(delta=delta), self.assertRaises(ValueError):
                    GalerkinAssembly(self.problem(delta=delta), basis, plan)
            for wall, interior in (("1e300", "1e-300"), ("1e-300", "1e300")):
                description = Coefficient(
                    lambda x, a, wall=wall, interior=interior: a.scalar(wall if x < 1 else interior),
                    lower_bound="1e-300", upper_bound="1e300", wall_value=wall, interfaces=(1,),
                )
                split_plan = QuadratureFactory(arithmetic, interfaces=(1,)).plan(5)
                with self.subTest(wall=wall, interior=interior), self.assertRaises(ValueError):
                    GalerkinAssembly(self.problem(epsilon=description), basis, split_plan)

    def test_assembly_rejects_incompatible_or_underintegrated_plans(self):
        arithmetic = MpmathArithmetic(50)
        basis = GalerkinBasis(4, arithmetic)
        problem = self.problem(epsilon=Coefficient(
            lambda x, a: a.scalar(1 if x < 1 else 2),
            lower_bound=1, upper_bound=2, wall_value=1, interfaces=(1,),
        ))
        for plan in (QuadratureFactory(arithmetic).plan(5),
                     QuadratureFactory(arithmetic, interfaces=(1,)).plan(4),
                     QuadratureFactory(arithmetic, L=2, interfaces=(1,)).plan(5),
                     QuadratureFactory(MpmathArithmetic(50), interfaces=(1,)).plan(5)):
            with self.assertRaises(ValueError):
                GalerkinAssembly(problem, basis, plan)
        extra = QuadratureFactory(arithmetic, interfaces=(1, 3)).plan(5)
        GalerkinAssembly(problem, basis, extra).residual(arithmetic.zeros(7))

    def test_replaced_plans_cannot_drop_terms_or_corrupt_positive_physical_quadrature(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            basis = GalerkinBasis(4, arithmetic)
            plan = QuadratureFactory(arithmetic, interfaces=(1,)).plan(5)
            problem = self.problem()
            malformed = []
            for field in ("nodes", "physical_nodes", "weights", "physical_weights", "panels"):
                malformed.append((field+" mutable", replace(
                    plan, **{field: list(getattr(plan, field))})))
            malformed.append(("mutable panel pairs", replace(
                plan, panels=tuple(list(panel) for panel in plan.panels))))
            for field in ("weights", "physical_weights"):
                weights = getattr(plan, field)
                for value in (arithmetic.scalar(0), -weights[0], arithmetic.scalar("nan")):
                    malformed.append((field+" invalid sign or finiteness", replace(
                        plan, **{field: (value,)+weights[1:]})))
                malformed.append((field+" truncated", replace(plan, **{field: weights[:-1]})))
            malformed.extend((
                ("reference weights disagree with physical weights", replace(
                    plan, weights=(2*plan.weights[0],)+plan.weights[1:])),
                ("scaled physical weights disagree with the mapping", replace(
                    plan, physical_weights=tuple(2*value for value in plan.physical_weights))),
                ("positive increasing physical nodes disagree with the mapping", replace(
                    plan, physical_nodes=(plan.physical_nodes[0]/2,)+plan.physical_nodes[1:])),
                ("reference endpoint is sampled", replace(
                    plan, nodes=(arithmetic.scalar(-1),)+plan.nodes[1:])),
                ("overlapping panels", replace(
                    plan, panels=((arithmetic.scalar(-1), arithmetic.scalar("0.25")),
                                  (arithmetic.scalar(0), arithmetic.scalar(1))))),
            ))
            for reason, invalid in malformed:
                with self.subTest(backend=arithmetic.backend, reason=reason):
                    with self.assertRaises(ValueError):
                        GalerkinAssembly(problem, basis, invalid)

    def test_contexts_remain_isolated_and_owned_precision_changes_invalidate_assembly(self):
        low_arithmetic, high_arithmetic = MpmathArithmetic(30), MpmathArithmetic(120)
        problem = rational_problem(1, 1)
        low = self.assembly(low_arithmetic, problem)
        high = self.assembly(high_arithmetic, problem)
        low_state, high_state = self.exact_state(low_arithmetic), self.exact_state(high_arithmetic)
        expected = high.residual(high_state)
        previous = mp.mp.prec
        try:
            mp.mp.dps = 17
            low_arithmetic.context.dps = 50
            for operation in (low.check_precision, lambda: low.residual(low_state),
                              lambda: low.jacobian(low_state), lambda: low.linear_matrix):
                with self.assertRaises(RuntimeError):
                    operation()
            self.assertEqual(high_arithmetic.norm(high.residual(high_state)-expected), 0)
            self.assertEqual(high_arithmetic.dps, 120)
        finally:
            mp.mp.prec = previous

    def test_decimal_wall_and_delta_preserve_sub_float64_corrections(self):
        arithmetic = MpmathArithmetic(120)
        decimal = "1."+"0"*79+"1"
        problem = self.problem(
            delta=decimal,
            boundary_conditions=((0, None, None, decimal, None), (None,)*5),
        )
        assembly = self.assembly(arithmetic, problem)
        residual = assembly.residual(arithmetic.zeros(7))
        # R_2[0] = -2*delta²*q_b = -2*(1+eta)^3.
        correction = (residual[3]+2)/(-6)
        self.assertLess(abs(correction-arithmetic.scalar("1e-80")), arithmetic.scalar("1e-118"))

    def test_refinement_compares_fixed_state_without_solving(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            assembly = self.assembly(arithmetic, rational_problem(), order=5)
            state = arithmetic.vector(["0.1", "0.2", "-0.1", "0.3", "-0.2", "0.1", "0.2"])
            before = tuple(state)
            refined = assembly.refined()
            self.assertIsNot(refined, assembly)
            self.assertIs(refined.basis, assembly.basis)
            self.assertIs(refined.physical, assembly.physical)
            self.assertGreater(len(refined.plan.nodes), len(assembly.plan.nodes))
            estimate = assembly.check_refinement(state, atol=0, rtol=0, q="0.375")
            self.assertFalse(estimate.converged)
            self.assertGreater(estimate.residual_difference, 0)
            self.assertGreater(estimate.jacobian_difference, 0)
            self.assertIsNot(estimate.refined_assembly, assembly)
            self.assertEqual(tuple(state), before)
            exact_linear = self.assembly(arithmetic)
            estimate = exact_linear.check_refinement(
                state, atol=self.tolerance(arithmetic), rtol=self.tolerance(arithmetic))
            self.assertTrue(estimate.converged)

    def test_invalid_states_parameters_and_unexpected_callbacks_fail_explicitly(self):
        arithmetic = NumpyArithmetic()
        assembly = self.assembly(arithmetic)
        for invalid in ([0]*6, [0]*8, [0, 0, 0, 0, 0, 0, "nan"]):
            for operation in (assembly.residual, assembly.jacobian, assembly.linear_residual):
                with self.assertRaises(ValueError):
                    operation(invalid)
        for parameter in ("nan", "inf", -1):
            for operation in (assembly.residual, assembly.jacobian):
                with self.assertRaises(ValueError):
                    operation(arithmetic.zeros(7), parameter)

        def broken(x, u, a):
            raise RuntimeError("user source failed")

        broken_assembly = self.assembly(arithmetic, self.problem(f=broken))
        with self.assertRaisesRegex(RuntimeError, "user source failed"):
            broken_assembly.residual(arithmetic.zeros(7))
        nonfinite_source = self.assembly(arithmetic, self.problem(f=lambda x, u, a: a.inf))
        with self.assertRaises(OverflowError):
            nonfinite_source.residual(arithmetic.zeros(7))
        result = solve_newton(nonfinite_source.nonlinear_system(), use_globalization=False)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "nonfinite_residual")
        nonfinite_derivative = self.assembly(arithmetic, self.problem(
            f=lambda x, u, a: a.exp(-x), f_u=lambda x, u, a: a.inf))
        with self.assertRaises(OverflowError):
            nonfinite_derivative.jacobian(arithmetic.zeros(7))
        result = solve_newton(nonfinite_derivative.nonlinear_system(), use_globalization=False)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "nonfinite_jacobian")


if __name__ == "__main__":
    unittest.main()
