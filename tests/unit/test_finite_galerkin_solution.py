"""Native finite mixed fields, endpoint lifts, derivatives, and physical norms."""

import unittest

import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.galerkin.finite_basis import FiniteGalerkinBasis
from overscreening_breakdown.galerkin.finite_solution import FiniteGalerkinSolution
from overscreening_breakdown.galerkin.solution import GalerkinSolution
from overscreening_breakdown.plotting import plot_coefficients, plot_fields
from overscreening_breakdown.problems.flux import FluxProblem
from overscreening_breakdown.quadrature import QuadratureFactory


class FiniteGalerkinSolutionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)

    def problem(self, *, R="2", epsilon=1, boundaries=None):
        if boundaries is None:
            boundaries = (("1", None, None, "0", None), ("3", "0", None, None, None))
        return FluxProblem(delta="2", epsilon=epsilon, domain=(0, R),
                           f=lambda x, u, a: u, f_u=lambda x, u, a: a.scalar(1),
                           boundary_conditions=boundaries)

    def fields(self, arithmetic):
        epsilon = Coefficient(lambda x, a: 1+x/3, lower_bound="1", upper_bound="2", wall_value="1")
        problem = self.problem(epsilon=epsilon)
        basis = FiniteGalerkinBasis(3, arithmetic, domain=(0, "2"))
        plan = QuadratureFactory(arithmetic, domain=(0, "2")).plan(8)
        return FiniteGalerkinSolution(basis, problem.bind(arithmetic),
                                      ["0.25", "-0.125", "1", "2", "-0.5", "0.25"], plan=plan)

    def close(self, a, actual, expected):
        tolerance = a.scalar("1e-{}".format(max(10, a.dps-8)))
        self.assertLessEqual(abs(actual-expected), tolerance*max(1, abs(expected)))

    def test_independent_polynomials_and_derivatives_obey_finite_geometry(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(30), MpmathArithmetic(120)):
            with self.subTest(mpmath=a.use_mpmath, digits=a.dps):
                result = self.fields(a)
                self.assertEqual((result.N, result.unknown_count), (3, 6))
                self.assertEqual((len(result.u_coefficients), len(result.w_coefficients)), (2, 4))
                self.assertEqual(result.domain, (0, 2))
                self.assertEqual(result.length, 2)
                self.assertFalse(hasattr(result, "L"))
                for text in ("0", "0.125", "1", "1.875", "2"):
                    x = a.scalar(text)
                    y = x-1
                    u = 1+x+(1-y*y)*(a.scalar(1)/4-y/8)
                    u_dx = 1-2*y*(a.scalar(1)/4-y/8)-(1-y*y)/8
                    w = a.scalar(3)/2+a.scalar(5)*y/4-y*y+y**3
                    w_dx = a.scalar(5)/4-2*y+3*y*y
                    for function, expected in ((result.u, u), (result.w, w),
                                               (result.u_dx, u_dx), (result.w_dx, w_dx),
                                               (result.flux, (1+x/3)*(u_dx+4*w_dx))):
                        value = function(x)
                        self.assertIsInstance(value, type(a.scalar(0)))
                        self.close(a, value, expected)
                self.assertEqual((result.u(0), result.u(2)), (1, 3))

    def test_array_shapes_and_coefficient_copies_preserve_native_fields(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            result = self.fields(a)
            points = np.array([["0", "0.5"], ["1", "2"]], dtype=object)
            values = result.u(points)
            self.assertEqual(values.shape, points.shape)
            self.assertEqual(values.dtype, np.dtype(object if a.use_mpmath else np.float64))
            self.assertEqual(values[1, 1], 3)
            original = result.u("0.5")
            copy = result.coefficients
            copy[0] = 999
            self.assertEqual(result.u("0.5"), original)
            self.assertEqual(result.u(np.array("0.5", dtype=object)), original)
            for x in ("-0.1", "2.1", "inf", "nan"):
                with self.subTest(x=x), self.assertRaises(ValueError):
                    result.flux(x)

    def test_physical_norms_integrate_in_dx_and_keep_w_independent(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50), MpmathArithmetic(120)):
            boundaries = ((0, None, None, 0, None), (3, 1, None, None, None))
            problem = self.problem(R="3", boundaries=boundaries)
            basis = FiniteGalerkinBasis(2, a, domain=(0, "3"))
            plan = QuadratureFactory(a, domain=(0, "3")).plan(10)
            result = FiniteGalerkinSolution(basis, problem.bind(a), [0, 2, 0, 0], plan=plan)
            self.close(a, result.u(1), a.scalar(1))
            self.assertEqual(result.w(1), 2)
            self.close(a, result.l2_norm(plan=plan), a.scalar(3))
            self.close(a, result.h1_norm(plan=plan), a.sqrt(a.scalar(12)))
            self.close(a, result.l2_norm("w", plan=plan), a.sqrt(a.scalar(12)))
            self.close(a, result.h1_norm("w", plan=plan), a.sqrt(a.scalar(12)))
            self.close(a, result.l2_norm("flux", plan=plan), a.sqrt(a.scalar(3)))
            self.close(a, result.l2_error(lambda x: x/2, plan=plan), a.scalar(3)/2)
            self.close(a, result.h1_error(lambda x: x, lambda x: a.scalar(1), plan=plan), a.scalar(0))
            with self.assertRaises(ValueError):
                result.h1_norm("flux", plan=plan)

    def test_diagnostics_reject_other_domains_owners_and_missing_interface_splits(self):
        a = NumpyArithmetic()
        epsilon = Coefficient(lambda x, ar: ar.scalar(1 if x < 1 else 2),
                               lower_bound=1, upper_bound=2, wall_value=1, interfaces=(1,))
        basis = FiniteGalerkinBasis(3, a, domain=(0, "2"))
        result = FiniteGalerkinSolution(basis, self.problem(epsilon=epsilon).bind(a), [0]*6)
        plans = (QuadratureFactory(a, domain=(0, "3")).plan(8),
                 QuadratureFactory(a, L="1", interfaces=(1,)).plan(8),
                 QuadratureFactory(NumpyArithmetic(), domain=(0, "2"), interfaces=(1,)).plan(8),
                 QuadratureFactory(a, domain=(0, "2")).plan(8))
        for plan in plans:
            with self.subTest(mapping=plan.mapping.mapping_type), self.assertRaises(ValueError):
                result.l2_norm(plan=plan)
        valid = QuadratureFactory(a, domain=(0, "2"), interfaces=(1,)).plan(8)
        self.assertGreater(result.l2_norm(plan=valid), 0)
        half_problem = FluxProblem(delta=1, f=lambda x, u, ar: u, f_u=lambda x, u, ar: ar.scalar(1),
                                    boundary_conditions=((1, None, None, 0, None), (None,)*5))
        half = GalerkinSolution(GalerkinBasis(3, a), half_problem.bind(a), [0]*5)
        with self.assertRaises(ValueError):
            half.l2_norm(plan=valid)

    def test_constructor_and_precision_guards_reject_incompatible_fields(self):
        a = MpmathArithmetic(50)
        result = self.fields(a)
        with self.assertRaises(ValueError):
            FiniteGalerkinSolution(result.basis, self.problem(R="3").bind(a), [0]*6)
        with self.assertRaises(ValueError):
            FiniteGalerkinSolution(result.basis, self.problem().bind(MpmathArithmetic(50)), [0]*6)
        with self.assertRaises(ValueError):
            FiniteGalerkinSolution(result.basis, result.physical, [0]*5)
        with self.assertRaises(ValueError):
            FiniteGalerkinSolution(result.basis, result.physical, [0]*6, converged=True)
        a.context.dps += 10
        with self.assertRaises(RuntimeError):
            result.u(1)

    def test_common_plots_use_the_finite_domain_without_a_mapping_length_alias(self):
        result = self.fields(NumpyArithmetic())
        before = tuple(result.coefficients)
        fields = plot_fields(result)
        self.assertEqual(len(fields.axes), 3)
        self.assertEqual((fields.axes[0].lines[0].get_xdata()[0],
                          fields.axes[0].lines[0].get_xdata()[-1]), (0, 2))
        coefficients = plot_coefficients(result)
        self.assertEqual([len(axis.lines[0].get_xdata()) for axis in coefficients.axes], [2, 4])
        fields.canvas.draw()
        coefficients.canvas.draw()
        self.assertEqual(tuple(result.coefficients), before)
