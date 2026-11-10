"""Independent mixed fields and unweighted physical L2/H1 diagnostics."""

from dataclasses import FrozenInstanceError, replace
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.problems.flux import FluxProblem
from overscreening_breakdown.quadrature import QuadratureFactory
from overscreening_breakdown.galerkin.solution import (GalerkinSolution, physical_h1_error, physical_h1_norm,
                       physical_l2_error, physical_l2_norm)


class GalerkinSolutionTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for precision in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(precision)

    def tolerance(self, arithmetic):
        return arithmetic.scalar("3e-12" if not arithmetic.use_mpmath
                                 else "3e-{}".format(max(12, arithmetic.dps-4)))

    def problem(self, **options):
        arguments = dict(
            delta=2, f=lambda x, u, a: a.scalar(0), f_u=lambda x, u, a: a.scalar(0),
            epsilon=Coefficient(lambda x, a: 1+a.scalar(1)/(1+x),
                                lower_bound=1, upper_bound=2, wall_value=2),
            boundary_conditions=((1, None, None, -6, None), (None,)*5),
        )
        arguments.update(options)
        return FluxProblem(**arguments)

    def solution(self, arithmetic, problem=None, coefficients=None, **options):
        problem = self.problem() if problem is None else problem
        basis = GalerkinBasis(4, arithmetic)
        if coefficients is None:
            coefficients = arithmetic.vector([0, 0, 0, "-0.375", "0.5", "-0.125", 0])
        return GalerkinSolution(basis, problem.bind(arithmetic), coefficients, **options)

    def assert_native(self, arithmetic, value):
        if arithmetic.use_mpmath:
            self.assertIsInstance(value, type(arithmetic.scalar(0)))
            self.assertNotIsInstance(value, (float, np.floating))
        else:
            self.assertIsInstance(value, np.float64)

    def test_exact_independent_fields_derivatives_and_physical_flux(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                solution = self.solution(arithmetic)
                self.assertEqual(solution.solver_type, "galerkin")
                self.assertEqual((solution.N, solution.L, solution.unknown_count), (4, 1, 7))
                self.assertEqual(solution.domain, (0, arithmetic.inf))
                for text in ("0", "0.125", "1", "4", "10"):
                    x = arithmetic.scalar(text)
                    s = arithmetic.scalar(1)/(1+x)
                    for evaluate, expected in ((solution.u, s), (solution.w, -2*s**3),
                                               (solution.u_dx, -s*s), (solution.w_dx, 6*s**4),
                                               (solution.flux, (1+s)*(-s*s+24*s**4))):
                        value = evaluate(x)
                        self.assert_native(arithmetic, value)
                        self.assertLess(abs(value-expected), self.tolerance(arithmetic))
                self.assertEqual(solution.u(0), 1)
                self.assertEqual(solution.w(0), -2)
                self.assertEqual(solution.u_dx(0), -1)
                self.assertEqual(solution.w_dx(0), 6)
                for evaluate in (solution.u, solution.w, solution.u_dx, solution.w_dx, solution.flux):
                    self.assertEqual(evaluate(arithmetic.inf), 0)

    def test_w_is_independent_of_u_and_flux_uses_both_fields(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            first = self.solution(arithmetic)
            changed = first.coefficients
            changed[3] += 1
            second = self.solution(arithmetic, coefficients=changed)
            x = arithmetic.scalar("0.5")
            self.assertEqual(first.u(x), second.u(x))
            self.assertEqual(first.u_dx(x), second.u_dx(x))
            self.assertNotEqual(first.w(x), second.w(x))
            expected = first.physical.epsilon(x)*4*(second.w_dx(x)-first.w_dx(x))
            self.assertLess(abs((second.flux(x)-first.flux(x))-expected), self.tolerance(arithmetic))

    def test_array_scalar_empty_and_zero_dimensional_evaluations_preserve_types(self):
        for arithmetic in self.arithmetics():
            solution = self.solution(arithmetic)
            dtype = object if arithmetic.use_mpmath else np.float64
            points = np.array([[arithmetic.scalar(0), arithmetic.scalar("0.25")],
                               [arithmetic.scalar(1), arithmetic.inf]], dtype=dtype)
            before = points.copy()
            for evaluate in (solution.u, solution.w, solution.u_dx, solution.w_dx, solution.flux):
                observed = evaluate(points)
                self.assertEqual(observed.shape, points.shape)
                self.assertEqual(observed.dtype, np.dtype(dtype))
                for index in np.ndindex(points.shape):
                    self.assertEqual(observed[index], evaluate(points[index]))
                    self.assert_native(arithmetic, observed[index])
                self.assert_native(arithmetic, evaluate(np.array(arithmetic.scalar("0.25"))))
                empty = evaluate(np.empty((2, 0, 3)))
                self.assertEqual(empty.shape, (2, 0, 3))
                self.assertEqual(empty.dtype, np.dtype(dtype))
            np.testing.assert_array_equal(points, before)

    def test_far_tail_keeps_decaying_modes_after_y_rounds_to_one(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                solution = self.solution(arithmetic)
                x = arithmetic.scalar("1e100")
                s = arithmetic.scalar(1)/(1+x)
                for evaluate, expected in ((solution.u, s), (solution.w, -2*s**3),
                                           (solution.u_dx, -s*s)):
                    value = evaluate(x)
                    self.assertNotEqual(value, 0)
                    self.assertLess(abs(value/expected-1), self.tolerance(arithmetic))

    def test_native_coefficients_are_snapshots_and_properties_return_copies(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            coefficients = arithmetic.vector([0, 0, 0, "-0.375", "0.5", "-0.125", 0])
            info = [["source", "collocation"], ["source_N", 8]]
            solution = self.solution(arithmetic, coefficients=coefficients,
                                     projection_info=info, quadrature_history=["diagnostic"])
            expected = solution.w("0.5")
            coefficients[3] = 99
            info[0][1] = "changed"
            self.assertEqual(solution.w("0.5"), expected)
            self.assertEqual(solution.projection_info[0], ("source", "collocation"))
            self.assertEqual(solution.quadrature_history, ("diagnostic",))
            for name, index in (("coefficients", 3), ("u_coefficients", 0), ("w_coefficients", 0)):
                copied = getattr(solution, name)
                copied[index] += 9
                self.assertNotEqual(copied[index], getattr(solution, name)[index])
            for attribute in ("basis", "physical", "converged", "_entries", "projection_info"):
                with self.assertRaises(FrozenInstanceError):
                    setattr(solution, attribute, None)
            self.assertFalse(solution.converged)
            self.assertFalse(solution.nonlinear_converged)
            self.assertFalse(solution.quadrature_converged)
            for flags in (dict(converged=True),
                          dict(converged=True, nonlinear_converged=True),
                          dict(method="projection", nonlinear_converged=True),
                          dict(method="projection", quadrature_converged=True)):
                with self.assertRaises(ValueError):
                    self.solution(arithmetic, **flags)
            checked = self.solution(arithmetic, converged=True, nonlinear_converged=True,
                                    quadrature_converged=True)
            self.assertTrue(checked.converged)

    def test_physical_derivatives_agree_with_independent_rational_differentiation(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                coefficients = arithmetic.vector(["0.25", "-0.125", 0,
                                                  "0.5", "-0.25", "0.125", 0])
                solution = self.solution(arithmetic, coefficients=coefficients)
                oracle = mp.mp.clone()
                oracle.dps = arithmetic.dps+20

                def u(x):
                    y = (x-1)/(x+1)
                    return 1/(1+x)+(1-y*y)*(oracle.mpf("0.25")-oracle.mpf("0.125")*y)

                def w(x):
                    y = (x-1)/(x+1)
                    return (1-y)*(oracle.mpf("0.5")-oracle.mpf("0.25")*y
                                  +oracle.mpf("0.125")*(2*y*y-1))

                for text in ("0.125", "0.75", "4", "1000"):
                    coordinate = arithmetic.scalar(text)
                    for evaluate, reference in ((solution.u_dx, u), (solution.w_dx, w)):
                        expected = arithmetic.scalar(oracle.diff(reference, oracle.mpf(text)))
                        self.assertLess(abs(evaluate(coordinate)-expected), self.tolerance(arithmetic))

    def test_physical_norms_have_correct_analytic_values_for_both_fields(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                solution = self.solution(arithmetic)
                plan = QuadratureFactory(arithmetic).plan(8)
                expected = (("u", arithmetic.scalar(1), arithmetic.sqrt(arithmetic.scalar(4)/3)),
                            ("w", arithmetic.sqrt(arithmetic.scalar(4)/5),
                             arithmetic.sqrt(arithmetic.scalar(208)/35)))
                for name, l2, h1 in expected:
                    self.assertLess(abs(solution.l2_norm(name, plan=plan)-l2), self.tolerance(arithmetic))
                    self.assertLess(abs(solution.h1_norm(name, plan=plan)-h1), self.tolerance(arithmetic))
                self.assertGreater(solution.l2_norm("flux", plan=plan), 0)
                with self.assertRaises(ValueError):
                    solution.h1_norm("flux", plan=plan)
                with self.assertRaises(ValueError):
                    solution.l2_norm("unknown", plan=plan)

    def test_physical_norms_scale_with_mapping_length_and_use_dx(self):
        for arithmetic in self.arithmetics():
            for raw_length in ("0.5", "3"):
                length = arithmetic.scalar(raw_length)
                plan = QuadratureFactory(arithmetic, L=raw_length).plan(4)
                function = lambda x: length/(x+length)
                derivative = lambda x: -length/(x+length)**2
                observed_l2 = physical_l2_norm(function, plan)
                observed_h1 = physical_h1_norm(function, derivative, plan)
                self.assert_native(arithmetic, observed_l2)
                self.assertLess(abs(observed_l2-arithmetic.sqrt(length)), self.tolerance(arithmetic))
                self.assertLess(abs(observed_h1-arithmetic.sqrt(length+1/(3*length))),
                                self.tolerance(arithmetic))

    def test_physical_errors_use_independent_reference_fields_and_derivatives(self):
        for arithmetic in self.arithmetics():
            solution = self.solution(arithmetic)
            plan = QuadratureFactory(arithmetic).plan(8)
            reference = lambda x: arithmetic.scalar(1)/(1+x)
            reference_dx = lambda x: -arithmetic.scalar(1)/(1+x)**2
            self.assertLess(solution.l2_error(reference, plan=plan), self.tolerance(arithmetic))
            self.assertLess(solution.h1_error(reference, reference_dx, plan=plan), self.tolerance(arithmetic))
            half = lambda x: reference(x)/2
            half_dx = lambda x: reference_dx(x)/2
            self.assertLess(abs(physical_l2_error(solution.u, half, plan)-arithmetic.scalar("0.5")),
                            self.tolerance(arithmetic))
            expected = arithmetic.sqrt(arithmetic.scalar(1)/3)
            self.assertLess(abs(physical_h1_error(solution.u, solution.u_dx, half, half_dx, plan)-expected),
                            self.tolerance(arithmetic))
            w = lambda x: -arithmetic.scalar(2)/(1+x)**3
            expected = arithmetic.sqrt(arithmetic.scalar(36)/7)
            self.assertLess(abs(solution.h1_error(w, lambda x: arithmetic.scalar(0), "w", plan=plan)-expected),
                            self.tolerance(arithmetic))

    def test_high_precision_wall_parameters_and_coefficients_are_preserved(self):
        arithmetic = MpmathArithmetic(120)
        decimal = "1."+"0"*79+"1"
        problem = self.problem(boundary_conditions=((decimal, None, None, -6, None), (None,)*5))
        coefficients = arithmetic.vector(["1e-80", 0, 0, "1e-80", 0, 0, 0])
        solution = self.solution(arithmetic, problem, coefficients)
        self.assertLess(abs((solution.u(0)-1)-arithmetic.scalar("1e-80")), arithmetic.scalar("1e-118"))
        self.assertEqual(solution.w(1), arithmetic.scalar("1e-80"))
        original_precision = mp.mp.prec
        before = solution.u(arithmetic.scalar("0.375"))
        try:
            mp.mp.dps = 17
            self.solution(MpmathArithmetic(30))
            self.assertEqual(solution.u(arithmetic.scalar("0.375")), before)
            self.assertEqual(arithmetic.dps, 120)
        finally:
            mp.mp.prec = original_precision

    def test_stale_contexts_and_stale_bound_parameters_are_rejected(self):
        arithmetic = MpmathArithmetic(30)
        problem = self.problem()
        physical = problem.bind(arithmetic)
        solution = self.solution(arithmetic, problem)
        arithmetic.context.dps = 50
        for operation in (solution.check_precision, lambda: solution.u(0), lambda: solution.w(0),
                          lambda: solution.coefficients, lambda: solution.flux(1)):
            with self.assertRaises(RuntimeError):
                operation()
        fresh_basis = GalerkinBasis(4, arithmetic)
        with self.assertRaises(RuntimeError):
            GalerkinSolution(fresh_basis, physical, [0]*7)

    def test_norm_plans_must_match_owner_mapping_and_interfaces(self):
        arithmetic = MpmathArithmetic(50)
        coefficient = Coefficient(lambda x, a: a.scalar(1 if x < 1 else 2),
                                  lower_bound=1, upper_bound=2, wall_value=1, interfaces=(1,))
        solution = self.solution(arithmetic, self.problem(epsilon=coefficient))
        for plan in (QuadratureFactory(arithmetic).plan(8),
                     QuadratureFactory(arithmetic, L=2, interfaces=(1,)).plan(8),
                     QuadratureFactory(MpmathArithmetic(50), interfaces=(1,)).plan(8)):
            with self.assertRaises(ValueError):
                solution.l2_norm(plan=plan)
        compatible = QuadratureFactory(arithmetic, interfaces=(1,)).plan(8)
        self.assertGreater(solution.l2_norm(plan=compatible), 0)
        with self.assertRaises(ValueError):
            GalerkinSolution(solution.basis, self.problem().bind(MpmathArithmetic(50)), [0]*7)

    def test_nonfinite_callbacks_and_malformed_physical_norm_plans_fail_explicitly(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            plan = QuadratureFactory(arithmetic).plan(4)
            for invalid in ("inf", "-inf", "nan"):
                with self.assertRaises(OverflowError):
                    physical_l2_norm(lambda x: arithmetic.scalar(invalid), plan)
            with self.assertRaises(TypeError):
                physical_l2_norm(lambda x: [1], plan)
            with self.assertRaises(TypeError):
                physical_h1_norm(lambda x: 1, None, plan)
            with self.assertRaises(TypeError):
                physical_l2_norm(lambda x: 1, None)
            for invalid in (replace(plan, physical_weights=plan.physical_weights[:-1]),
                            replace(plan, physical_weights=(arithmetic.scalar(0),)+plan.physical_weights[1:]),
                            replace(plan, physical_weights=(-plan.physical_weights[0],)+plan.physical_weights[1:])):
                with self.assertRaises(ValueError):
                    physical_l2_norm(lambda x: 1, invalid)

    def test_invalid_coordinates_and_coefficients_are_rejected(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            solution = self.solution(arithmetic)
            for invalid in (-1, "nan", "-inf"):
                for evaluate in (solution.u, solution.w, solution.u_dx, solution.w_dx, solution.flux):
                    with self.assertRaises(ValueError):
                        evaluate(invalid)
            for invalid in (1j, [1j]):
                with self.assertRaises(TypeError):
                    solution.u(invalid)
            for invalid in ([0]*6, [0]*8, [0, 0, 0, 0, 0, 0, "nan"]):
                with self.assertRaises(ValueError):
                    self.solution(arithmetic, coefficients=invalid)


if __name__ == "__main__":
    unittest.main()
