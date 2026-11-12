"""Explicit problem objects remain the source of Galerkin physics."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp

from overscreening_breakdown import FluxProblem, load_solution
from overscreening_breakdown.arithmetic import MpmathArithmetic
from overscreening_breakdown.coefficients import Coefficient, ConstantCoefficient
from overscreening_breakdown.layered import preset_material
from overscreening_breakdown.manufactured.transmission import layered_transmission_case
from overscreening_breakdown.notebook_examples import discontinuous_galerkin as guide


NUMERICAL = dict(quadrature_atol="1e-8", quadrature_rtol="1e-8")


def explicit_rational_problem(*, wall="2", third="-12"):
    """Manufacture u=wall/(1+x), epsilon=1+(1+x)^-1, and delta=2."""
    calls = {name: [] for name in ("epsilon", "f", "f_u", "initial_guess")}

    def epsilon(x, a):
        calls["epsilon"].append((a.use_mpmath, a.dps))
        return 1+1/(1+x)

    material = Coefficient(epsilon, lower_bound="1", upper_bound="2", wall_value="2")

    def f(x, u, a):
        calls["f"].append((a.use_mpmath, a.dps))
        t, amplitude = 1+x, a.scalar(wall)
        J, J_dx = -1/t**2+24/t**4, 2/t**3-96/t**5
        return amplitude*(-J/t**2+(1+1/t)*J_dx)+(u-amplitude/t)

    def f_u(x, u, a):
        calls["f_u"].append((a.use_mpmath, a.dps))
        return a.scalar(1)

    def guess(x, a):
        calls["initial_guess"].append((a.use_mpmath, a.dps))
        return a.scalar(wall)*a.exp(-x)

    problem = FluxProblem(delta="2", epsilon=material, f=f, f_u=f_u, initial_guess=guess,
                          boundary_conditions=((wall, None, None, third, None),
                                               (0, None, None, None, None)))
    return problem, material, calls


class DefinedGalerkinProblemTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)

    def test_supplied_callbacks_coefficient_boundaries_and_guess_are_used(self):
        problem, material, calls = explicit_rational_problem()
        result = guide.run_defined_problem(problem, N=4, newton_atol="1e-10", material=material,
                                           model="explicit rational equation", numerical=NUMERICAL)
        self.assertTrue(result.solution.converged)
        self.assertIs(result.problem, problem)
        self.assertTrue(all(calls.values()))
        physical, a = result.solution.physical, result.solution.arithmetic
        self.assertEqual(physical.delta, 2)
        self.assertEqual(physical.boundary_conditions,
                         ((2, None, None, -12, None), (0, None, None, None, None)))
        self.assertAlmostEqual(physical.epsilon(a.scalar(2)), 4/3, places=14)
        self.assertEqual(physical.physical_rhs("0.4", "0.7"),
                         problem.f(a.scalar("0.4"), a.scalar("0.7"), a))
        self.assertEqual(physical.physical_rhs_u("0.4", "0.7"), 1)
        for x in (0, 0.5, 2):
            self.assertAlmostEqual(result.solution.u(x), 2/(1+x), places=10)
            self.assertAlmostEqual(result.solution.w(x), -4/(1+x)**3, places=9)

    def test_model_labels_do_not_rebuild_the_supplied_physics(self):
        problem, material, _ = explicit_rational_problem()
        results = [guide.run_defined_problem(problem, N=4, newton_atol="1e-10", material=material,
                                             model=label, numerical=NUMERICAL)
                   for label in ("BSK", "manufactured_nonlinear")]
        for result, label in zip(results, ("BSK", "manufactured_nonlinear")):
            self.assertEqual(result.model, label)
            self.assertIs(result.problem, problem)
            self.assertIsNone(result.case)
            self.assertEqual(result.solution.physical.boundary_conditions[0][0], 2)
            self.assertAlmostEqual(result.solution.u(1), 1, places=10)
            self.assertEqual(result.settings["definition"]["description_type"].rsplit(".", 1)[-1], "FluxProblem")
            self.assertNotIn("sigma", result.settings["physical"])
        self.assertEqual(tuple(results[0].solution.coefficients), tuple(results[1].solution.coefficients))

    def test_changing_a_visible_forcing_callback_changes_the_solution(self):
        state = {"amplitude": "0"}
        material = ConstantCoefficient(1)

        def h(x, order):
            return sum(coefficient*(-1)**order*math.prod(range(power, power+order))
                       /(1+x)**(power+order)
                       for power, coefficient in enumerate((1, -4, 6, -4, 1), 1))

        def f(x, u, a):
            amplitude = a.scalar(state["amplitude"])
            exact = 1/(1+x)+amplitude*h(x, 0)
            return 2/(1+x)**3-96/(1+x)**5+amplitude*(h(x, 2)-4*h(x, 4))+(u-exact)

        problem = FluxProblem(delta="2", epsilon=material, f=f, f_u=lambda x, u, a: a.scalar(1),
                              initial_guess=lambda x, a: a.exp(-x),
                              boundary_conditions=((1, None, None, -6, None),
                                                   (0, None, None, None, None)))
        first = guide.run_defined_problem(problem, N=8, newton_atol="1e-10", material=material,
                                          numerical=NUMERICAL)
        before = first.solution.u(1)
        # A caller may change a closure between experiments; keep it fixed
        # during each solve. The second solve must use that current function.
        state["amplitude"] = "0.25"
        second = guide.run_defined_problem(problem, N=8, newton_atol="1e-10", material=material,
                                           numerical=NUMERICAL)
        self.assertTrue(first.solution.converged and second.solution.converged)
        self.assertIs(second.problem, problem)
        self.assertAlmostEqual(before, 0.5, places=9)
        self.assertAlmostEqual(second.solution.u(1), 0.5+0.25/32, places=9)
        self.assertGreater(abs(second.solution.u(1)-before), 1e-3)

    def test_one_problem_supports_independent_numpy_and_mpmath_contexts(self):
        wall = "1.000000000000000000000000000001"
        problem, material, calls = explicit_rational_problem(
            wall=wall, third="-6.000000000000000000000000000006")
        mp.mp.prec = 77
        numpy = guide.run_defined_problem(problem, N=4, newton_atol="1e-10", material=material,
                                          numerical=NUMERICAL)
        precise = guide.run_defined_problem(problem, N=4, newton_atol="1e-35", material=material,
                                            accuracy="high_precision", numerical=NUMERICAL)
        self.assertTrue(numpy.solution.converged and precise.solution.converged)
        a = precise.solution.arithmetic
        self.assertEqual(a.dps, 50)
        self.assertIsInstance(precise.solution.u(a.scalar("0.5")), type(a.scalar(0)))
        self.assertEqual(precise.solution.u(0), a.scalar(wall))
        self.assertNotEqual(precise.solution.u(0), a.scalar(1))
        self.assertEqual(numpy.solution.u(0), 1)
        self.assertEqual(problem.boundary_conditions[0][0], wall)
        self.assertTrue(any(flag for flag, _ in calls["f"]))
        self.assertTrue(any(not flag for flag, _ in calls["f"]))
        self.assertEqual(mp.mp.prec, 77)

    def test_refinement_reuses_the_original_problem_and_callbacks(self):
        problem, material, _ = explicit_rational_problem()
        coarse = guide.run_defined_problem(problem, N=4, newton_atol="1e-10", material=material,
                                            model="manufactured_nonlinear", numerical=NUMERICAL)
        comparison = guide.compare_resolution(coarse, fine_N=5, newton_atol="2e-11")
        fine = comparison.fine
        self.assertTrue(fine.solution.converged)
        self.assertIs(fine.problem, problem)
        self.assertIsNone(fine.case)
        self.assertEqual(fine.settings["physical"], coarse.settings["physical"])
        self.assertEqual(fine.settings["numerical"]["newton_atol"], "2e-11")
        for x, u in (("0.2", "0.4"), ("2", "-0.1")):
            a = fine.solution.arithmetic
            self.assertEqual(fine.solution.physical.physical_rhs(x, u),
                             problem.f(a.scalar(x), a.scalar(u), a))

    def test_exports_record_the_actual_definition_and_roundtrip_with_original_problem(self):
        problem, material, _ = explicit_rational_problem()
        result = guide.run_defined_problem(problem, N=4, newton_atol="1e-10", material=material,
                                           model="BSK", numerical=NUMERICAL)
        with tempfile.TemporaryDirectory(prefix="overscreening-defined-export-", dir="/tmp") as temporary:
            paths = guide.export_experiment(result, temporary)
            settings = json.loads(Path(paths["settings"]).read_text())
            self.assertEqual(settings["definition"]["description_type"].rsplit(".", 1)[-1], "FluxProblem")
            self.assertFalse(settings["definition"]["callbacks_serialized"])
            self.assertEqual(float(settings["physical"]["delta"]), 2)
            self.assertEqual(float(settings["physical"]["boundary_conditions"][0][0]), 2)
            self.assertEqual(float(settings["physical"]["boundary_conditions"][0][3]), -12)
            loaded = load_solution(paths["solution"], problem=problem, solver="galerkin")
            for field in ("u", "w", "flux"):
                self.assertEqual(getattr(loaded, field)(0.375), getattr(result.solution, field)(0.375))

    def test_mismatched_material_is_rejected_before_solving_or_replacing_epsilon(self):
        problem, material, _ = explicit_rational_problem()
        different_function = Coefficient(lambda x, a: 1+2/(2+x),
                                         lower_bound="1", upper_bound="2", wall_value="2")
        with patch.object(guide, "solve", side_effect=AssertionError("material mismatch must not solve")) as solve:
            for supplied in (ConstantCoefficient(1), different_function):
                with self.subTest(material=type(supplied).__name__), self.assertRaisesRegex(ValueError, "material|epsilon"):
                    guide.run_defined_problem(problem, N=4, newton_atol="1e-10", material=supplied)
            solve.assert_not_called()
        self.assertIs(problem.epsilon, material)

    def test_flux_dx_matches_independent_derivatives_and_manufactured_sources_across_interfaces(self):
        a = MpmathArithmetic(70)
        for count in (1, 3):
            for kind in ("constant", "graded"):
                case = layered_transmission_case(preset_material(count, kind=kind), delta="1.7")
                physical = case.make_problem().bind(a)
                for interface in physical.epsilon.interfaces:
                    for sign in (-1, 1):
                        with self.subTest(interfaces=count, kind=kind, side=sign):
                            x = interface+sign*a.scalar("1e-4")
                            derivative = case.flux_dx(x, a)
                            independent = a.context.diff(lambda t: case.flux(t, a), x)
                            source = physical.physical_rhs(x, case.u(x, a))
                            scale = max(1, abs(independent), abs(source))
                            self.assertIsInstance(derivative, type(a.scalar(0)))
                            self.assertLess(abs(derivative-independent), a.scalar("1e-55")*scale)
                            self.assertLess(abs(derivative-source), a.scalar("1e-55")*scale)
