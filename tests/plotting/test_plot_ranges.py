"""Display windows control native sampling without changing solver results."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.nonlinear_solver import NewtonIteration, NewtonResult, NewtonStage
from overscreening_breakdown.notebook_examples.discontinuous_plotting import (
    plot_convergence, plot_domain_comparison, plot_errors, plot_experiment,
    plot_material, plot_material_gallery,
)
from overscreening_breakdown.notebook_examples.homotopy_plotting import plot_homotopy_comparison
from overscreening_breakdown.solvers import QuadratureIteration


def example(*, arithmetic=None, R="4", interfaces=("1", "2", "3")):
    a = NumpyArithmetic() if arithmetic is None else arithmetic
    calls = []
    points = tuple(a.scalar(value) for value in interfaces)

    def coefficient(x, owner):
        calls.append(("epsilon", x, owner))
        return owner.scalar(1)+owner.scalar("0.2")*sum(x >= point for point in points)

    material = Coefficient(coefficient, lower_bound="1", upper_bound="2", wall_value="1",
                           interfaces=interfaces)

    def field(x):
        calls.append(("field", x, a))
        return a.exp(-x)+a.scalar("1e-70")

    def reference(x, owner):
        calls.append(("reference", x, owner))
        return owner.exp(-x)

    end = a.scalar(R)
    solution = SimpleNamespace(
        arithmetic=a, domain=(a.scalar(0), end), N=4, unknown_count=8, converged=True,
        physical=SimpleNamespace(epsilon=material.bind(a)),
        u=field, w=field, flux=field,
    )
    result = SimpleNamespace(
        solution=solution, model="known answer",
        case=SimpleNamespace(u=reference, w=reference, flux=reference),
        settings=dict(material={"description": "same layers"},
                      physical=dict(delta="1", domain=("0", R)),
                      numerical=dict(N=4, newton_atol="1", tolerance="1")),
    )
    return result, material, calls


def with_history(result):
    a = result.solution.arithmetic
    stage = NewtonStage(a.scalar(0), a.scalar(10), a.scalar(1), a.scalar(".5"), 3, True, "converged")
    history = tuple(NewtonIteration(1, index, a.scalar(0), a.scalar(value), a.scalar(1), 0)
                    for index, value in ((1, "5"), (2, "2"), (3, ".5")))
    newton = NewtonResult(a.vector([0]), True, "converged", a.scalar(0), a.scalar(".5"),
                          3, (stage,), history)
    result.solution.newton_result = newton
    result.solution.quadrature_history = tuple(QuadratureIteration(
        order, 1, True, index == 1, a.scalar(".1"), a.scalar(".01"),
        a.scalar(".2"), a.scalar(".01"), "check", newton)
        for index, order in enumerate((12, 24)))
    return result


class PlotRangeTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")

    def curves(self, axis):
        return [line for line in axis.lines if len(line.get_xdata()) >= 3]

    def test_profiles_and_errors_sample_only_the_window_and_retain_ownership(self):
        result, _, calls = example()
        for factory in (plot_experiment, plot_errors):
            calls.clear()
            figure = factory(result, x_min="1.25", x_max="2.75", samples=9)
            self.assertTrue(calls)
            self.assertTrue(all(1.25 <= x <= 2.75 for _, x, _ in calls))
            self.assertEqual(figure._overscreening_solutions, (result.solution,))
            for axis in figure.axes:
                self.assertEqual(axis.get_xlim(), (1.25, 2.75))
                guides = [line.get_xdata()[0] for line in axis.lines
                          if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]]
                self.assertEqual(guides, [2])
                for line in self.curves(axis):
                    x = np.asarray(line.get_xdata())
                    self.assertTrue(np.all((1.25 <= x) & (x <= 2.75)))
                    self.assertFalse(x.min() < 2 < x.max())
            figure.canvas.draw()

    def test_material_previews_and_gallery_use_the_actual_sampling_window(self):
        _, material, calls = example()
        for factory in (
            lambda: plot_material(material, x_min="1.25", x_max="2.75", samples=7),
            lambda: plot_material_gallery(materials={3: material}, x_min="1.25", x_max="2.75", samples=7),
        ):
            calls.clear()
            figure = factory()
            self.assertTrue(calls)
            self.assertTrue(all(1.25 <= x <= 2.75 for _, x, _ in calls))
            self.assertTrue(all(axis.get_xlim() == (1.25, 2.75) for axis in figure.axes))
            figure.canvas.draw()

    def test_invalid_physical_limits_fail_before_any_figure_or_callback(self):
        result, material, calls = example()
        for left, right in ((-1, 2), (2, 2), (3, 2), ("nan", 2), (0, "inf"), (True, 2)):
            for factory in (
                lambda: plot_experiment(result, x_min=left, x_max=right),
                lambda: plot_errors(result, x_min=left, x_max=right),
                lambda: plot_material(material, x_min=left, x_max=right),
                lambda: plot_material_gallery(materials={3: material}, x_min=left, x_max=right),
                lambda: plot_homotopy_comparison({"one": result}, x_min=left, x_max=right),
                lambda: plot_domain_comparison({"4": result}, x_min=left, x_max=right),
            ):
                with self.subTest(left=left, right=right), patch.object(
                    plt, "subplots", side_effect=AssertionError("validate before creating a figure")
                ), self.assertRaises(ValueError):
                    factory()
        for factory in (plot_experiment, plot_errors):
            with patch.object(plt, "subplots", side_effect=AssertionError("outside domain")), self.assertRaisesRegex(
                ValueError, "endpoint R"
            ):
                factory(result, x_min=1, x_max=5)
        self.assertEqual(calls, [])

    def test_native_tiny_window_is_evaluated_before_float_display_conversion(self):
        a = MpmathArithmetic(80)
        result, _, calls = example(arithmetic=a, R="4e-60", interfaces=("1.5e-60", "2.5e-60"))
        for factory in (plot_experiment, plot_errors):
            calls.clear()
            figure = factory(result, x_min="1e-60", x_max="3e-60", samples=5)
            self.assertTrue(calls)
            self.assertTrue(all(isinstance(x, type(a.scalar(0))) and owner is a for _, x, owner in calls))
            self.assertTrue(all(a.scalar("1e-60") <= x <= a.scalar("3e-60") for _, x, _ in calls))
            self.assertTrue(all(axis.get_xlim() == (1e-60, 3e-60) for axis in figure.axes))
            self.assertEqual(a.dps, 80)
            figure.canvas.draw()

    def test_unrenderable_native_bounds_fail_before_evaluation(self):
        a = MpmathArithmetic(80)
        result, _, calls = example(arithmetic=a, R="2e400", interfaces=())
        windows = ((a.scalar(0), a.scalar("1e400")),
                   (a.scalar(1)-a.scalar("1e-60"), a.scalar(1)+a.scalar("1e-60")))
        for left, right in windows:
            for factory in (
                lambda: plot_experiment(result, x_min=left, x_max=right),
                lambda: plot_errors(result, x_min=left, x_max=right),
                lambda: plot_homotopy_comparison({"native": result}, x_min=left, x_max=right),
                lambda: plot_convergence(with_history(result), iteration_xlim=(left, right)),
            ):
                with patch.object(plt, "subplots", side_effect=AssertionError("unrenderable bounds")), self.assertRaisesRegex(
                    ValueError, "float64 display"
                ):
                    factory()
        self.assertEqual(calls, [])

    def test_comparison_windows_preserve_each_native_domain_and_provenance(self):
        first, _, first_calls = example(R="2", interfaces=("1",))
        second, _, second_calls = example(R="4", interfaces=("1",))
        results = {"2": first, "4": second}
        figure = plot_domain_comparison(results, x_min=".5", samples=6)
        self.assertTrue(all(axis.get_xlim() == (.5, 4) for axis in figure.axes))
        self.assertTrue(all(.5 <= x <= 2 for _, x, _ in first_calls))
        self.assertTrue(all(.5 <= x <= 4 for _, x, _ in second_calls))
        self.assertEqual(figure._overscreening_solutions, (first.solution, second.solution))
        first_calls.clear(); second_calls.clear()
        with patch.object(plt, "subplots", side_effect=AssertionError("must fit every domain")), self.assertRaisesRegex(
            ValueError, "endpoint R"
        ):
            plot_domain_comparison(results, x_min=".5", x_max="3")
        self.assertEqual(first_calls+second_calls, [])
        homotopy = plot_homotopy_comparison({"direct": first, "continued": first},
                                           x_min=".5", x_max="1.5", samples=6)
        self.assertTrue(all(axis.get_xlim() == (.5, 1.5) for axis in homotopy.axes))
        self.assertTrue(all(.5 <= x <= 1.5 for _, x, _ in first_calls))
        self.assertEqual(homotopy._overscreening_solutions, (first.solution,))

    def test_iteration_and_quadrature_display_limits_do_not_truncate_history(self):
        result = with_history(example()[0])
        before_settings = copy.deepcopy(result.settings)
        history = result.solution.quadrature_history
        baseline = plot_convergence(result)
        ranged = plot_convergence(result, iteration_xlim=(1, 2), quadrature_xlim=(15, 30))
        self.assertEqual(ranged.axes[0].get_xlim(), (1, 2))
        self.assertEqual(ranged.axes[1].get_xlim(), (15, 30))
        for original, changed in zip(baseline.axes, ranged.axes):
            self.assertEqual(len(original.lines), len(changed.lines))
            for old_line, new_line in zip(original.lines, changed.lines):
                np.testing.assert_array_equal(old_line.get_xdata(), new_line.get_xdata())
                np.testing.assert_array_equal(old_line.get_ydata(), new_line.get_ydata())
        self.assertTrue(any(max(line.get_xdata()) >= 6 for line in self.curves(ranged.axes[0])))
        self.assertIs(result.solution.quadrature_history, history)
        self.assertEqual(result.settings, before_settings)
        self.assertEqual(ranged._overscreening_solutions, (result.solution,))
        ranged.canvas.draw()

    def test_invalid_history_axis_limits_are_rejected_before_figure_creation(self):
        result = with_history(example()[0])
        for invalid in ((2, 1), (1, 1), (-1, 2), (0, "inf"), (0,), "02", (False, 2)):
            for parameter in ("iteration_xlim", "quadrature_xlim"):
                with self.subTest(parameter=parameter, limits=invalid), patch.object(
                    plt, "subplots", side_effect=AssertionError("axis limits must be prevalidated")
                ), self.assertRaises(ValueError):
                    plot_convergence(result, **{parameter: invalid})
