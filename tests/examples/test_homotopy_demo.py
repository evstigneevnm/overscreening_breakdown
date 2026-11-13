"""Actual bounded failures, continuation recovery, and optional notebook cells."""

import ast
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from matplotlib import pyplot as plt

from overscreening_breakdown import ContinuationOptions, FluxProblem, ScreenedLinearReference
from overscreening_breakdown.continuation import AffineReference, ResidualHomotopy
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.layered import LayeredMaterial
from overscreening_breakdown.manufactured import layered_transmission_case
from overscreening_breakdown.nonlinear_solver import solve_newton
from overscreening_breakdown.notebook_examples import discontinuous_galerkin as guide
from overscreening_breakdown.notebook_examples.homotopy_plotting import (
    plot_homotopy_comparison, plot_homotopy_path,
)


NOTEBOOK = Path(__file__).resolve().parents[2]/"examples/notebooks/execution_example_galerkin_discontinuous.ipynb"
HARD_CELLS = ("hard-homotopy-definition", "hard-homotopy-runs",
              "hard-homotopy-stages", "hard-homotopy-plots")


def hard_definition():
    material = LayeredMaterial((("0", "1"), ("1", "2")))
    case = layered_transmission_case(material, delta="1", nonlinear=True, beta="10")
    calls = dict(f=0, f_u=0, guess=0)

    def f(x, u, a):
        calls["f"] += 1
        exact = case.u(x, a)
        return case.flux_dx(x, a)+(u-exact)+10*(u**3-exact**3)

    def f_u(x, u, a):
        calls["f_u"] += 1
        return 1+30*u*u

    def guess(x, a):
        calls["guess"] += 1
        return 10*a.exp(-x*x)

    problem = FluxProblem(delta="1", epsilon=material.coefficient, f=f, f_u=f_u,
                          initial_guess=guess,
                          boundary_conditions=((1, None, None, -6, None),
                                               (0, None, None, None, None)))
    continuation = ContinuationOptions(
        reference=ScreenedLinearReference(strength="1"), initial_step="0.5", max_step="0.5",
        growth="1.5", min_step="1e-6", max_stages=40, max_total_iterations=150,
    )
    return material, case, problem, continuation, calls


class EnableHardDemo(ast.NodeTransformer):
    def __init__(self, enabled):
        self.enabled = enabled

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "run_homotopy_demo"
               for target in node.targets):
            node.value = ast.copy_location(ast.Constant(self.enabled), node.value)
        return self.generic_visit(node)


class HomotopyDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material, cls.case, cls.problem, cls.continuation, cls.calls = hard_definition()
        cls.results = {}
        for use_mpmath in (False, True):
            runs = {}
            for label, iterations, globalization in (("direct_limited", 5, False),
                                                    ("homotopy", 5, True),
                                                    ("direct_control", 35, False)):
                runs[label] = guide.run_defined_problem(
                    cls.problem, material=cls.material, case=cls.case, model="hard nonlinear case",
                    N=8, newton_atol="1e-9", verbosity=0,
                    numerical=dict(L="1", use_mpmath=use_mpmath, precision=30,
                                   max_iterations=iterations, use_globalization=globalization,
                                   continuation=cls.continuation, quadrature_order=20,
                                   quadrature_atol="1e-8", quadrature_rtol="1e-8",
                                   max_refinements=2, newton_fallback=None),
                )
            cls.results[use_mpmath] = runs

    def setUp(self):
        self.addCleanup(plt.close, "all")

    def test_actual_limited_newton_fails_and_reference_recovery_reaches_the_physical_problem(self):
        for use_mpmath, runs in self.results.items():
            with self.subTest(mpmath=use_mpmath):
                direct, recovered, control = (runs[name].solution for name in
                                               ("direct_limited", "homotopy", "direct_control"))
                self.assertFalse(direct.converged)
                self.assertFalse(direct.nonlinear_converged)
                self.assertEqual(direct.newton_result.reason, "max_iterations")
                self.assertLessEqual(direct.newton_result.iterations, 5)
                self.assertIsNone(direct.newton_result.reference_result)
                self.assertTrue(recovered.converged and control.converged)
                self.assertTrue(recovered.nonlinear_converged and recovered.quadrature_converged)
                self.assertEqual(recovered.newton_result.homotopy, 0)
                self.assertEqual(control.newton_result.homotopy, 0)
                self.assertTrue(recovered.newton_result.reference_result.converged)
                self.assertTrue(any(stage.converged and 0 < stage.homotopy < 1
                                    for stage in recovered.newton_result.stages))
                self.assertTrue(any(not stage.converged for stage in recovered.newton_result.stages[1:]))
                self.assertLessEqual(recovered.newton_result.iterations, 150)
                self.assertTrue(all(stage.iterations <= 5 for stage in recovered.newton_result.stages))
                self.assertLessEqual(control.newton_result.iterations, 35)
                self.assertIsNone(control.newton_result.reference_result)
                self.assertEqual(recovered.u(0), 1)
                self.assertIs(runs["homotopy"].problem, self.problem)

    def test_stage_rows_keep_native_targets_and_last_accepted_parameters_after_rejection(self):
        for use_mpmath, runs in self.results.items():
            with self.subTest(mpmath=use_mpmath):
                result = runs["homotopy"]
                a = result.solution.arithmetic
                rows = guide.homotopy_stage_rows(result)
                self.assertIsInstance(rows, tuple)
                self.assertEqual(rows[0]["kind"], "initial")
                self.assertIsNone(rows[0]["accepted_q"])
                accepted, previous_plan, next_attempt = None, None, 0
                rejected = []
                for row in rows:
                    if row["plan"] != previous_plan:
                        accepted, previous_plan, next_attempt = None, row["plan"], 0
                    self.assertEqual(row["attempt"], next_attempt)
                    next_attempt += 1
                    self.assertIsInstance(row["q"], type(a.scalar(0)))
                    self.assertIsInstance(row["target_norm"], type(a.scalar(0)))
                    self.assertEqual(row["target_norm"], a.scalar("1e-9"))
                    self.assertIsInstance(row["accepted"], bool)
                    if row["accepted"]:
                        accepted = row["q"]
                        self.assertLessEqual(row["residual_norm"], row["target_norm"])
                    elif row["kind"] == "continuation":
                        rejected.append(row)
                    self.assertEqual(row["accepted_q"], accepted)
                self.assertTrue(rejected)
                self.assertEqual(rows[-1]["accepted_q"], 0)
                with self.assertRaises(TypeError):
                    rows[0]["q"] = a.scalar(1)

    def test_the_retry_starts_from_the_verified_reference_not_the_failed_trial(self):
        for use_mpmath, runs in self.results.items():
            with self.subTest(mpmath=use_mpmath):
                solution = runs["homotopy"].solution
                a = solution.arithmetic
                stored = solution.newton_result
                failed, retry = stored.stages[1:3]
                self.assertFalse(failed.converged)
                self.assertTrue(retry.converged)
                assembly = GalerkinAssembly(self.problem, solution.basis, solution.plan)
                system = assembly.nonlinear_system()
                matrix, offset = assembly.reference_operator("1")
                blend = ResidualHomotopy(system, AffineReference(matrix, offset, system))
                reference_state = stored.reference_result.state
                independent_failure = solve_newton(
                    blend.nonlinear_system(reference_state), parameter=failed.homotopy,
                    atol="1e-9", rtol=0, max_iterations=5,
                )
                self.assertFalse(independent_failure.converged)
                retained_norm = system.residual_norm(blend.residual(reference_state, retry.homotopy))
                discarded_norm = system.residual_norm(blend.residual(independent_failure.state, retry.homotopy))
                tolerance = a.scalar("1e-20" if use_mpmath else "1e-11")*max(1, abs(retained_norm))
                self.assertLessEqual(abs(retry.initial_norm-retained_norm), tolerance)
                self.assertGreater(abs(retained_norm-discarded_norm), a.scalar("1e-3"))

    def test_reference_and_final_physical_residuals_are_independently_verified(self):
        for use_mpmath, runs in self.results.items():
            with self.subTest(mpmath=use_mpmath):
                solution = runs["homotopy"].solution
                a = solution.arithmetic
                assembly = GalerkinAssembly(self.problem, solution.basis, solution.plan)
                reference = solution.newton_result.reference_result
                matrix, offset = assembly.reference_operator(reference.strength)
                product = matrix*reference.state if use_mpmath else matrix.dot(reference.state)
                self.assertLessEqual(a.norm(product+offset), a.scalar("1e-9"))
                self.assertLessEqual(assembly.residual_norm(assembly.residual(solution.coefficients)),
                                     a.scalar("1e-9"))

    def test_recovered_and_larger_budget_control_agree_at_the_newton_scale(self):
        for use_mpmath, runs in self.results.items():
            recovered, control = runs["homotopy"].solution, runs["direct_control"].solution
            a = recovered.arithmetic
            for field in ("u", "w", "flux"):
                with self.subTest(mpmath=use_mpmath, field=field):
                    for text in ("0", "0.125", "0.5", "1", "2", "5", "20"):
                        x = a.scalar(text)
                        left, right = getattr(recovered, field)(x), getattr(control, field)(x)
                        self.assertLessEqual(abs(left-right), 10*a.scalar("1e-9")*max(1, abs(right)))

    def test_reports_use_stored_observations_without_changing_the_problem_or_results(self):
        before_calls = dict(self.calls)
        epsilon, boundaries = self.problem.epsilon, self.problem.boundary_conditions
        options = asdict(self.continuation)
        for runs in self.results.values():
            coefficients = {name: tuple(result.solution.coefficients) for name, result in runs.items()}
            comparison = guide.report_homotopy_comparison(runs)
            self.assertTrue(all(name in comparison for name in runs))
            self.assertIn("Stage cap", comparison)
            self.assertIn("Total cap", comparison)
            self.assertIn("Newton attempts", comparison)
            self.assertIn("Final residuals are physical only at q=0", comparison)
            stages = guide.report_homotopy_stages(runs["homotopy"])
            self.assertIn("rejected", stages)
            self.assertIn("Accepted q", stages)
            self.assertEqual(coefficients, {name: tuple(result.solution.coefficients)
                                             for name, result in runs.items()})
        self.assertEqual(self.calls, before_calls)
        self.assertIs(self.problem.epsilon, epsilon)
        self.assertEqual(self.problem.boundary_conditions, boundaries)
        self.assertEqual(asdict(self.continuation), options)

    def run_notebook_cells(self, enabled):
        original = NOTEBOOK.read_bytes()
        notebook = json.loads(original)
        sentinel = object()
        namespace = dict(FluxProblem=FluxProblem, LayeredMaterial=LayeredMaterial,
                         ContinuationOptions=ContinuationOptions, ScreenedLinearReference=ScreenedLinearReference,
                         layered_transmission_case=layered_transmission_case, verbosity=0, plt=plt,
                         homotopy_stage_rows=guide.homotopy_stage_rows,
                         report_homotopy_stages=guide.report_homotopy_stages,
                         report_homotopy_comparison=guide.report_homotopy_comparison,
                         report_error_table=guide.report_error_table, plot_errors=guide.plot_errors,
                         plot_convergence=guide.plot_convergence, plot_homotopy_path=plot_homotopy_path,
                         plot_homotopy_comparison=plot_homotopy_comparison,
                         plot_x_min="0", plot_x_max="8",
                         iteration_xlim=None, quadrature_xlim=None,
                         experiment=sentinel, N_selected=100, N_high_precision=100,
                         run_high_precision_example=True)
        with patch.object(guide, "run_defined_problem", wraps=guide.run_defined_problem) as run, \
                patch.object(plt, "show"), redirect_stdout(io.StringIO()):
            namespace["run_defined_problem"] = run
            found = []
            for cell in notebook["cells"]:
                if cell.get("id") not in HARD_CELLS:
                    continue
                found.append(cell["id"])
                tree = ast.fix_missing_locations(EnableHardDemo(enabled).visit(ast.parse("".join(cell["source"]))))
                exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
            calls = tuple(run.call_args_list)
        self.assertEqual(tuple(found), HARD_CELLS)
        self.assertIs(namespace["experiment"], sentinel)
        self.assertEqual(namespace["N_selected"], 100)
        self.assertEqual(namespace["N_high_precision"], 100)
        self.assertTrue(namespace["run_high_precision_example"])
        self.assertEqual(NOTEBOOK.read_bytes(), original)
        return namespace, calls

    def test_disabled_notebook_demonstration_performs_no_solves(self):
        namespace, calls = self.run_notebook_cells(False)
        self.assertFalse(calls)
        self.assertEqual(namespace["hard_results"], {})
        self.assertEqual(namespace["hard_demo_figures"], {})

    def test_isolated_enabled_notebook_demonstrates_actual_recovery_and_plots(self):
        namespace, calls = self.run_notebook_cells(True)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call.args[0] is namespace["hard_problem"] for call in calls))
        runs = namespace["hard_results"]
        self.assertEqual(set(runs), {"direct_limited", "homotopy", "direct_control"})
        self.assertFalse(runs["direct_limited"].solution.converged)
        self.assertTrue(runs["homotopy"].solution.converged)
        self.assertTrue(runs["direct_control"].solution.converged)
        self.assertTrue(namespace["hard_reference_used"] and namespace["hard_positive_stage"])
        self.assertEqual(len(namespace["hard_demo_figures"]), 4)
        for figure in namespace["hard_demo_figures"].values():
            figure.canvas.draw()
