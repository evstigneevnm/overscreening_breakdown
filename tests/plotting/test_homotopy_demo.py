"""Real continuation recovery and truthful plots of accepted/rejected states."""

from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from overscreening_breakdown import FluxProblem, LayeredMaterial, ScreenedLinearReference
from overscreening_breakdown.arithmetic import MpmathArithmetic
from overscreening_breakdown.continuation import ContinuationOptions
from overscreening_breakdown.manufactured import layered_transmission_case
from overscreening_breakdown.nonlinear_solver import NewtonResult, NewtonStage
from overscreening_breakdown.notebook_examples.discontinuous_galerkin import (
    homotopy_stage_rows, run_defined_problem,
)
from overscreening_breakdown.notebook_examples.homotopy_plotting import (
    plot_homotopy_comparison, plot_homotopy_path,
)


class HomotopyDemoPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = Path(__file__).resolve().parents[2]/"examples/notebooks/execution_example_galerkin_discontinuous.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        definition = next(cell for cell in cells if cell.get("id") == "hard-homotopy-definition")
        inputs = dict(FluxProblem=FluxProblem, LayeredMaterial=LayeredMaterial,
                      ScreenedLinearReference=ScreenedLinearReference,
                      ContinuationOptions=ContinuationOptions,
                      layered_transmission_case=layered_transmission_case)
        exec(compile("".join(definition["source"]), str(notebook), "exec"), inputs)
        cls.material, cls.case, cls.problem = (
            inputs["hard_material"], inputs["hard_case"], inputs["hard_problem"])
        cls.results = OrderedDict()
        for label, globalize, budget in (
            ("five-step Newton", False, inputs["hard_stage_iterations"]),
            ("reference homotopy", True, inputs["hard_stage_iterations"]),
            ("longer Newton control", False, inputs["hard_control_iterations"]),
        ):
            cls.results[label] = run_defined_problem(
                cls.problem, material=cls.material, case=cls.case, model="manufactured_nonlinear",
                N=inputs["N_hard"], newton_atol=inputs["newton_atol_hard"], numerical=dict(
                    inputs["hard_numerical"], max_iterations=budget, use_globalization=globalize),
            )

    def setUp(self):
        self.addCleanup(plt.close, "all")

    def test_actual_recovery_path_preserves_rejected_trial_and_rollback(self):
        direct, continued, control = self.results.values()
        self.assertFalse(direct.solution.converged)
        self.assertTrue(continued.solution.converged)
        self.assertTrue(control.solution.converged)
        rows = homotopy_stage_rows(continued)
        first = [row for row in rows if row["plan"] == 1]
        self.assertEqual(first[0]["kind"], "initial")
        self.assertFalse(first[0]["accepted"])
        reference = next(row for row in first if row["kind"] == "reference")
        self.assertTrue(reference["accepted"])
        rejected = next(row for row in first if row["kind"] == "continuation" and not row["accepted"])
        self.assertAlmostEqual(float(rejected["q"]), 0.5)
        self.assertEqual(rejected["accepted_q"], 1)
        with patch.object(plt, "show", side_effect=AssertionError("caller owns display")):
            figure = plot_homotopy_path(continued)
        self.assertIsInstance(figure, Figure)
        axis = figure.axes[0]
        accepted = next(line for line in axis.lines if line.get_label() == "plan 1: accepted state")
        attempts, parameters = np.asarray(accepted.get_xdata()), np.asarray(accepted.get_ydata())
        self.assertEqual(attempts[0], reference["attempt"])
        self.assertNotIn(first[0]["attempt"], attempts)
        self.assertEqual(parameters[attempts.tolist().index(rejected["attempt"])], 1)
        self.assertEqual(parameters[-1], 0)
        self.assertTrue(any(np.allclose(collection.get_offsets(),
                                      [[rejected["attempt"], float(rejected["q"])]])
                            for collection in axis.collections
                            if "rejected trial" in collection.get_label()))
        rollback = next(line for line in axis.lines if "rollback" in line.get_label())
        np.testing.assert_allclose(rollback.get_ydata(), [0.5, 1])
        for plan in {row["plan"] for row in rows if row["accepted_q"] is not None}:
            self.assertTrue(any(line.get_label() == "plan {}: accepted state".format(plan)
                                for line in axis.lines))
        figure.canvas.draw()

    def test_failed_physical_attempt_has_no_fabricated_accepted_path(self):
        figure = plot_homotopy_path(self.results["five-step Newton"])
        axis = figure.axes[0]
        self.assertFalse(any("accepted state" in line.get_label() for line in axis.lines))
        self.assertFalse(any("reference" in collection.get_label() for collection in axis.collections))
        self.assertIn("unverified", figure._suptitle.get_text())
        figure.canvas.draw()

    def test_comparison_marks_partial_results_and_never_connects_flux_across_jump(self):
        original = [tuple(result.solution.coefficients) for result in self.results.values()]
        with patch.object(plt, "show", side_effect=AssertionError("caller owns display")):
            figure = plot_homotopy_comparison(self.results, samples=12)
        self.assertEqual(len(figure.axes), 3)
        for axis in figure.axes:
            partial = [line for line in axis.lines if "unverified partial result" in line.get_label()]
            self.assertTrue(partial)
            self.assertTrue(all(line.get_linestyle() == ":" for line in partial))
            self.assertTrue(any(line.get_label() == "known answer" for line in axis.lines))
        flux_axis = figure.axes[2]
        self.assertIn("flux", flux_axis.get_ylabel())
        for line in flux_axis.lines:
            x = np.asarray(line.get_xdata())
            if len(x) >= 3:
                self.assertFalse(x.min() < 1 < x.max())
        self.assertIn("N=8 (15 unknowns)", figure._suptitle.get_text())
        figure.canvas.draw()
        self.assertEqual(original, [tuple(result.solution.coefficients) for result in self.results.values()])

    def test_native_callbacks_and_tiny_stage_targets_keep_their_context(self):
        original_bits = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_bits)
        mp.mp.prec = 57
        arithmetic = MpmathArithmetic(80)
        scalar_type = type(arithmetic.scalar(0))
        calls = []

        def native_field(x):
            self.assertIsInstance(x, scalar_type)
            calls.append(x)
            return arithmetic.scalar("1e-60")*arithmetic.exp(-x)

        def native_reference(x, owner):
            self.assertIs(owner, arithmetic)
            return native_field(x)

        target = arithmetic.scalar("1e-400")
        stage = NewtonStage(arithmetic.scalar(0), arithmetic.scalar(1), target,
                            arithmetic.scalar(0), 1, True, "converged")
        newton = NewtonResult(arithmetic.vector([0]), True, "converged", arithmetic.scalar(0),
                              arithmetic.scalar(0), 1, (stage,), ())
        solution = SimpleNamespace(
            arithmetic=arithmetic, N=8, converged=True,
            physical=SimpleNamespace(epsilon=self.material.coefficient.bind(arithmetic)),
            u=native_field, w=native_field, flux=native_field,
            newton_result=newton, quadrature_history=(),
        )
        result = SimpleNamespace(solution=solution, model="native test", case=SimpleNamespace(
            u=native_reference, w=native_reference, flux=native_reference))
        comparison = plot_homotopy_comparison({"mpmath": result}, samples=5)
        positive = np.asarray(comparison.axes[0].lines[0].get_ydata())
        self.assertTrue(np.all(positive > 0))
        self.assertLess(positive.max(), 2e-60)
        path = plot_homotopy_path(result)
        self.assertEqual(newton.stages[0].target_norm, target)
        self.assertEqual(arithmetic.dps, 80)
        self.assertEqual(mp.mp.prec, 57)
        self.assertTrue(calls)
        comparison.canvas.draw()
        path.canvas.draw()

