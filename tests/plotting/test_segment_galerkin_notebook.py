"""Finite guided figures retain their actual geometry, counts and arithmetic."""

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from matplotlib import pyplot as plt
import numpy as np

from overscreening_breakdown import (
    BSKProblem, ConstantCoefficient, LayeredMaterial, MpmathArithmetic, NumpyArithmetic,
)
from overscreening_breakdown.galerkin.finite_solution import FiniteGalerkinSolution
from overscreening_breakdown.manufactured import finite_segment_case
from overscreening_breakdown.notebook_examples.discontinuous_galerkin import run_defined_problem
from overscreening_breakdown.notebook_examples.discontinuous_plotting import (
    plot_convergence, plot_domain_comparison, plot_errors, plot_experiment, plot_material_gallery,
)
from overscreening_breakdown.notebook_examples.homotopy_plotting import (
    plot_homotopy_comparison, plot_homotopy_path,
)


class SegmentGalerkinPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material = LayeredMaterial(((0, 1), ("0.7", 2)))
        cls.results = {}
        for R in ("2", "3"):
            problem = BSKProblem(
                epsilon=cls.material.coefficient, delta="1", sigma="1.2", mu="0.05", gamma="0.4",
                domain=(0, R), boundary_conditions=(("0.05", None, None, 0, None),
                                                   (0, 0, None, None, None)),
            )
            cls.results[R] = run_defined_problem(
                problem, N=4, newton_atol="1e-10", material=cls.material, model="BSK",
            )
            assert cls.results[R].solution.converged
        case = finite_segment_case(nonlinear=True, nonzero_boundary=True)
        cls.reference = run_defined_problem(
            case.make_problem(), N=4, newton_atol="1e-10", material=ConstantCoefficient(1),
            case=case, model="manufactured_nonlinear",
        )

    def setUp(self):
        self.addCleanup(plt.close, "all")

    def curves(self, axis):
        return [line for line in axis.lines if len(line.get_xdata()) >= 3]

    def test_finite_profiles_errors_and_history_show_actual_domain_and_count(self):
        result = self.results["2"]
        for factory, experiment in ((plot_experiment, result), (plot_errors, self.reference),
                                    (plot_convergence, result), (plot_homotopy_path, result)):
            with self.subTest(plot=factory.__name__), patch.object(
                plt, "show", side_effect=AssertionError("caller owns display")
            ):
                figure = factory(experiment)
                title = figure._suptitle.get_text()
                self.assertIn("R=2", title)
                self.assertIn("N=4", title)
                self.assertIn("8 unknowns", title)
                self.assertNotIn("7 unknowns", title)
                figure.canvas.draw()
        figure = plot_experiment(result)
        for axis in figure.axes:
            for line in self.curves(axis):
                x = np.asarray(line.get_xdata(), dtype=float)
                self.assertTrue(np.all((0 <= x) & (x <= 2)))
                self.assertFalse(x.min() < .7 < x.max(), "a field segment must not cross an interface")
        history = plot_convergence(result)
        self.assertIn("Newton absolute residual tolerance = 1e-10", history.axes[0].get_title())

    def test_explicit_finite_overrun_and_incompatible_homotopy_domains_fail_before_evaluation(self):
        result = self.results["2"]
        before = plt.get_fignums()
        with patch.object(FiniteGalerkinSolution, "u", side_effect=AssertionError("no field evaluation")):
            for factory in (plot_experiment, plot_errors):
                current = self.reference if factory is plot_errors else result
                with self.subTest(plot=factory.__name__), self.assertRaisesRegex(ValueError, "endpoint R"):
                    factory(current, x_max="2.01")
            with self.assertRaisesRegex(ValueError, "endpoint R"):
                plot_homotopy_comparison({"same interval": result}, x_max="2.01")
            with self.assertRaisesRegex(ValueError, "matching physical domains"):
                plot_homotopy_comparison(self.results)
        self.assertEqual(plt.get_fignums(), before)

    def test_homotopy_profiles_default_to_finite_endpoint(self):
        result = self.results["2"]
        figure = plot_homotopy_comparison({"Newton": result, "same result": result}, samples=8)
        self.assertIn("R=2, N=4 (8 unknowns)", figure._suptitle.get_text())
        for axis in figure.axes:
            self.assertTrue(self.curves(axis))
            self.assertTrue(all(np.max(line.get_xdata()) <= 2 for line in self.curves(axis)))
        figure.canvas.draw()

    def test_homotopy_same_domain_across_backends_retains_each_native_coordinates(self):
        high, low = MpmathArithmetic(30), MpmathArithmetic(30)
        high.context.prec += 1
        configurations = (((NumpyArithmetic(), MpmathArithmetic(80)), "2.1"),
                          ((high, low), "2.3"))
        for owners, radius in configurations:
            comparisons, seen = {}, []
            for index, a in enumerate(owners):
                end = a.scalar(radius)

                def field(x, owner=a, right=end):
                    self.assertIsInstance(x, type(owner.scalar(0)))
                    self.assertTrue(0 <= x <= right)
                    seen.append(owner)
                    return owner.exp(-x)

                solution = SimpleNamespace(
                    arithmetic=a, domain=(a.scalar(0), end), N=4, unknown_count=8, converged=True,
                    physical=SimpleNamespace(epsilon=ConstantCoefficient(1).bind(a)),
                    u=field, w=field, flux=field,
                )
                comparisons[str(index)] = SimpleNamespace(solution=solution, case=None)
            figure = plot_homotopy_comparison(comparisons, samples=8)
            self.assertEqual(set(seen), set(owners))
            figure.canvas.draw()

    def test_domain_comparison_uses_each_own_interval_at_fixed_settings(self):
        figure = plot_domain_comparison(self.results, samples=8)
        self.assertIn("Domain sensitivity", figure._suptitle.get_text())
        self.assertIn("N=4 (8 unknowns)", figure._suptitle.get_text())
        for axis in figure.axes:
            for color, endpoint in (("C0", 2), ("C1", 3)):
                lines = [line for line in self.curves(axis) if line.get_color() == color]
                self.assertTrue(lines)
                self.assertTrue(all(np.max(line.get_xdata()) <= endpoint for line in lines))
                self.assertAlmostEqual(max(np.max(line.get_xdata()) for line in lines), endpoint)
        figure.canvas.draw()
        before = plt.get_fignums()
        with patch.object(FiniteGalerkinSolution, "u", side_effect=AssertionError("no field evaluation")):
            with self.assertRaisesRegex(ValueError, "keys must match"):
                plot_domain_comparison({"4": self.results["2"]})
            changed = replace(self.results["3"], model="different source")
            with self.assertRaisesRegex(ValueError, "R-only sensitivity requires fixed"):
                plot_domain_comparison({"2": self.results["2"], "3": changed})
        self.assertEqual(plt.get_fignums(), before)

    def test_material_gallery_accepts_explicit_finite_profiles_without_solving(self):
        materials = {}
        levels = ("1", "2", "1.25", "1.75")
        for count in (1, 2, 3, 5, 7, 10):
            rows = [("0", "1")]+[(str(Decimal(2)*index/(count+1)), levels[index % 4])
                                     for index in range(1, count+1)]
            materials[count] = LayeredMaterial(rows)
        figure = plot_material_gallery(materials=materials, x_max="2", samples=8)
        self.assertEqual(len(figure.axes), 6)
        for axis in figure.axes:
            self.assertTrue(all(np.max(line.get_xdata()) <= 2 for line in self.curves(axis)))
        figure.canvas.draw()

    def test_finite_tiny_errors_are_formed_before_display_conversion(self):
        a = MpmathArithmetic(80)
        end, calls = a.scalar("2.5"), []

        def exact(x):
            self.assertIsInstance(x, type(a.scalar(0)))
            self.assertTrue(0 <= x <= end)
            calls.append(x)
            return 1/(1+x)

        delta = a.scalar("1e-60")
        numerical = lambda x: exact(x)*(1+delta)
        solution = SimpleNamespace(
            arithmetic=a, N=4, unknown_count=8, domain=(a.scalar(0), end), converged=True,
            physical=SimpleNamespace(epsilon=ConstantCoefficient(1).bind(a)),
            u=numerical, w=numerical, flux=numerical,
        )
        case = SimpleNamespace(**{field: lambda x, owner: exact(x) for field in ("u", "w", "flux")})
        result = SimpleNamespace(solution=solution, case=case, model="known answer")
        figure = plot_errors(result, samples=8)
        self.assertTrue(calls)
        for axis in figure.axes:
            values = np.asarray(self.curves(axis)[0].get_ydata(), dtype=float)
            self.assertTrue(np.all(values > 0))
            self.assertLess(values.max(), 2e-60)
            self.assertGreater(values.max(), 5e-61)
        figure.canvas.draw()
