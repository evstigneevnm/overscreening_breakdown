"""Bounded Galerkin solves for genuine multilayer transmission problems."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import mpmath as mp

from overscreening_breakdown import solve
from overscreening_breakdown.layered import SUPPORTED_INTERFACE_COUNTS, preset_material
from overscreening_breakdown.manufactured.transmission import layered_transmission_case
from overscreening_breakdown.nonlinear_solver import LinearSolveError, solve_newton
from overscreening_breakdown.quadrature import QuadratureFactory


class LayeredTransmissionSolveTests(unittest.TestCase):
    def options(self, *, use_mpmath=False, digits=30, N=8):
        return dict(solver="galerkin", L="1", use_mpmath=use_mpmath, precision=digits,
                    tolerance="1e-{}".format(digits-8) if use_mpmath else "1e-10",
                    quadrature_order=max(20, 2*N+4), quadrature_atol="1e-8",
                    quadrature_rtol="1e-8", max_refinements=2)

    def check_outcome(self, result, count):
        self.assertTrue(result.converged)
        self.assertTrue(result.nonlinear_converged)
        self.assertTrue(result.quadrature_converged)
        self.assertEqual(result.newton_result.homotopy, 0)
        self.assertEqual(len(result.physical.epsilon.interfaces), count)
        self.assertEqual(result.u(0), 1)
        self.assertEqual((result.u(result.arithmetic.inf), result.w(result.arithmetic.inf)), (0, 0))
        for text in ("0.125", "1", "3", "8"):
            x = result.arithmetic.scalar(text)
            for value in (result.u(x), result.w(x), result.flux(x)):
                self.assertTrue(result.arithmetic.isfinite(value))
                self.assertIsInstance(value, type(result.arithmetic.scalar(0)))

    def test_every_interface_count_supports_linear_and_nonlinear_material_families(self):
        for kind in ("constant", "graded"):
            for count in SUPPORTED_INTERFACE_COUNTS:
                for nonlinear in (False, True):
                    with self.subTest(kind=kind, interfaces=count, nonlinear=nonlinear):
                        case = layered_transmission_case(preset_material(count, kind=kind),
                                                         nonlinear=nonlinear)
                        result = solve(case.make_problem(), 8, **self.options())
                        self.check_outcome(result, count)
                        if not nonlinear:
                            self.assertLessEqual(result.newton_result.iterations, 2)

    def test_representative_multiple_interface_solutions_keep_mpmath_arithmetic(self):
        original = mp.mp.prec
        for digits in (30, 50):
            for count in (1, 3, 10):
                with self.subTest(digits=digits, interfaces=count):
                    case = layered_transmission_case(preset_material(count, kind="graded"))
                    result = solve(case.make_problem(), 8,
                                   **self.options(use_mpmath=True, digits=digits))
                    self.check_outcome(result, count)
                    self.assertEqual(result.arithmetic.dps, digits)
                    self.assertEqual(mp.mp.prec, original)
                    # These are discrete residual checks, not a claim that a
                    # global N=8 approximation has 30/50 accurate field digits.
                    self.assertLessEqual(result.newton_result.residual_norm,
                                         result.arithmetic.scalar("1e-{}".format(digits-8)))

    def test_refinement_reduces_independently_integrated_field_and_flux_errors(self):
        for count in (1, 3, 10):
            with self.subTest(interfaces=count):
                material = preset_material(count, kind="graded")
                case = layered_transmission_case(material)
                errors = []
                for N in (8, 16):
                    result = solve(case.make_problem(), N, **self.options(N=N))
                    self.check_outcome(result, count)
                    a = result.arithmetic
                    # The diagnostic rule differs from both assembly rules.
                    plan = QuadratureFactory(a, L="1", interfaces=material.bind(a).interfaces).plan(2*N+9)
                    errors.append(tuple(result.l2_error(
                        lambda x, field=field: getattr(case, field)(x, a), field=field, plan=plan)
                                        for field in ("u", "w", "flux")))
                for coarse, fine in zip(errors[0], errors[1]):
                    self.assertGreater(coarse, 0)
                    self.assertLess(fine, coarse)

    def test_reference_recovery_keeps_the_physical_multilayer_problem(self):
        material = preset_material(3, kind="graded")
        case = layered_transmission_case(material)
        calls = [0]

        def initial_failure(system, **options):
            calls[0] += 1
            if calls[0] == 1:
                def fail(matrix, rhs):
                    raise LinearSolveError("exercise multilayer reference recovery")
                system = replace(system, solve_linear=fail)
            return solve_newton(system, **options)

        with patch("overscreening_breakdown.continuation.solve_newton", side_effect=initial_failure):
            result = solve(case.make_problem(), 8, **self.options(use_mpmath=True))
        self.check_outcome(result, 3)
        self.assertTrue(result.newton_result.reference_result.converged)
        self.assertEqual(result.newton_result.stages[0].reason, "linear_solve_failed")
        self.assertTrue(any(stage.homotopy > 0 for stage in result.newton_result.stages))
        a = result.arithmetic
        bound = material.bind(a)
        for point in (a.scalar("0.25"), a.scalar("1.125"), a.scalar(6)):
            self.assertEqual(result.physical.epsilon(point), bound.epsilon(point))
        self.assertEqual(result.physical.boundary_conditions,
                         ((1, None, None, -6, None), (0, None, None, None, None)))

