"""Physical error ratios and bounded manufactured resolution studies."""

import ast
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mpmath as mp

from overscreening_breakdown import FluxProblem
from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import ConstantCoefficient
from overscreening_breakdown.galerkin.solution import physical_l2_error
from overscreening_breakdown.layered import LayeredMaterial, preset_material
from overscreening_breakdown.manufactured import layered_transmission_case
from overscreening_breakdown.notebook_examples import discontinuous_galerkin as guide
from overscreening_breakdown.quadrature import QuadratureFactory


NOTEBOOK = Path(__file__).resolve().parents[2]/"examples/notebooks/execution_example_galerkin_discontinuous.ipynb"
NUMERICAL = dict(quadrature_order=12, quadrature_atol="1e-8", quadrature_rtol="1e-8")
FIELDS = ("u", "w", "flux")


def field_result(arithmetic, references, computed):
    """Supply controlled native fields without a nonlinear solve."""
    plan = QuadratureFactory(arithmetic, L="1").plan(12)
    solution = SimpleNamespace(arithmetic=arithmetic, L=arithmetic.scalar(1), N=4,
                               plan=plan, converged=True,
                               physical=SimpleNamespace(epsilon=ConstantCoefficient(1).bind(arithmetic)),
                               **computed)
    solution.l2_error = lambda reference, *, field, plan: physical_l2_error(
        getattr(solution, field), reference, plan)
    case = SimpleNamespace(**{name: (lambda x, a, f=function: f(x))
                              for name, function in references.items()})
    return SimpleNamespace(solution=solution, case=case)


class TinyNotebookStudies(ast.NodeTransformer):
    def visit_Assign(self, node):
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id in ("run_linear_convergence", "run_nonlinear_convergence"):
                node.value = ast.copy_location(ast.Constant(True), node.value)
            if target.id in ("N_values_linear", "N_values_nonlinear"):
                node.value = ast.copy_location(ast.Tuple(
                    elts=[ast.Constant(4), ast.Constant(8)], ctx=ast.Load()), node.value)
        return self.generic_visit(node)


class ManufacturedConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material = LayeredMaterial((("0", "1"),))
        cls.exact_space = guide.run_experiment(
            cls.material, N=4, newton_atol="1e-10", model="manufactured_linear", delta="2",
            numerical=NUMERICAL,
        )

    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)

    def test_error_ratios_use_physical_norms_and_match_analytic_reference_norms(self):
        result = self.exact_space
        solution = result.solution
        before = tuple(solution.coefficients)
        settings, assembly_plan = result.settings, solution.plan
        metrics = guide.manufactured_error_metrics(result, order=43)
        plan = QuadratureFactory(solution.arithmetic, L=solution.L).plan(43)
        # u=(1+x)^-1, w=-2(1+x)^-3 and
        # J=-(1+x)^-2+24(1+x)^-4 have these exact L2(dx) norms.
        exact_norms = dict(u=1.0, w=math.sqrt(4/5), flux=math.sqrt(7667/105))
        self.assertEqual(set(metrics), set(FIELDS))
        for name in FIELDS:
            with self.subTest(field=name):
                numerical = getattr(solution, name)
                reference = getattr(result.case, name)
                squared_error = math.fsum(
                    float(weight)*(float(numerical(x))-float(reference(x, solution.arithmetic)))**2
                    for x, weight in zip(plan.physical_nodes, plan.physical_weights))
                absolute = math.sqrt(squared_error)
                self.assertTrue(math.isclose(metrics[name]["absolute"], absolute,
                                             rel_tol=1e-11, abs_tol=1e-28))
                self.assertAlmostEqual(metrics[name]["reference_norm"], exact_norms[name], places=11)
                self.assertTrue(math.isclose(metrics[name]["relative"], absolute/exact_norms[name],
                                             rel_tol=1e-11, abs_tol=1e-28))
        with self.assertRaises(TypeError):
            metrics["u"] = {}
        with self.assertRaises(TypeError):
            metrics["u"]["relative"] = 0
        with patch.object(guide, "run_experiment", side_effect=AssertionError("reports must not solve")):
            table = guide.report_error_table(result, order=31)
            self.assertIn("Absolute L2", table)
            self.assertIn("Relative L2 (%)", table)
        self.assertEqual(tuple(solution.coefficients), before)
        self.assertIs(result.settings, settings)
        self.assertIs(solution.plan, assembly_plan)

    def test_zero_reference_norms_and_tiny_mpmath_errors_retain_native_meaning(self):
        a = NumpyArithmetic()
        zero = lambda x: a.scalar(0)
        reference = lambda x: a.scalar(1)/(1+x)
        result = field_result(a, dict(u=zero, w=zero, flux=reference),
                              dict(u=reference, w=zero, flux=reference))
        metrics = guide.manufactured_error_metrics(result, order=17)
        self.assertAlmostEqual(metrics["u"]["absolute"], 1, places=13)
        self.assertEqual(metrics["w"]["absolute"], 0)
        self.assertIsNone(metrics["u"]["relative"])
        self.assertIsNone(metrics["w"]["relative"])
        self.assertEqual(metrics["flux"]["relative"], 0)
        table = guide.report_error_table(result, order=17)
        self.assertIn("undefined", table)
        self.assertIn("Relative L2 (%)", table)
        original = mp.mp.prec
        a = MpmathArithmetic(80)
        tiny = a.scalar("1e-60")
        reference = lambda x: a.scalar(1)/(1+x)
        computed = lambda x: (1+tiny)*reference(x)
        result = field_result(a, dict.fromkeys(FIELDS, reference), dict.fromkeys(FIELDS, computed))
        metrics = guide.manufactured_error_metrics(result, order=17)
        for values in metrics.values():
            for value in values.values():
                self.assertIsInstance(value, type(a.scalar(0)))
            self.assertGreater(values["absolute"], 0)
            self.assertLess(abs(values["absolute"]-tiny), a.scalar("1e-75"))
            self.assertLess(abs(values["relative"]-tiny), a.scalar("1e-75"))
        self.assertIn("e-58", guide.report_error_table(result, order=17))
        self.assertEqual(mp.mp.prec, original)

    def test_invalid_studies_fail_before_any_partial_work(self):
        cases = [dict(N_values=values) for values in
                 (None, (), (4, 4), (8, 4), (1, 4), (4, True), (4, 4.0), (4, 5.5), "48")]
        cases.extend((dict(model="BSK"), dict(newton_atol=0), dict(numerical={"N": 4}),
                      dict(numerical={"tolerance": "1e-10"}),
                      dict(numerical={"quadrature_order": 8}, N_values=(4, 16))))
        with patch.object(guide, "run_experiment", side_effect=AssertionError("invalid study must not start")) as run:
            for changes in cases:
                options = dict(N_values=(4, 8), newton_atol="1e-10", model="manufactured_linear")
                options.update(changes)
                with self.subTest(options=changes), self.assertRaises((TypeError, ValueError)):
                    guide.run_manufactured_convergence(self.material, **options)
            for name in ("N_values", "newton_atol", "model"):
                options = dict(N_values=(4,), newton_atol="1e-10", model="manufactured_linear")
                del options[name]
                with self.subTest(missing=name), self.assertRaises(TypeError):
                    guide.run_manufactured_convergence(self.material, **options)
            run.assert_not_called()

    def test_studies_preserve_shared_physics_precision_and_tolerances_without_warm_starts(self):
        material = preset_material(1)
        original_rows = material.rows
        for accuracy, atol in (("strict_integration", "8e-11"), ("high_precision", "3e-30")):
            with self.subTest(accuracy=accuracy):
                numerical = dict(quadrature_order=16, quadrature_atol="1e-8", quadrature_rtol="1e-8")
                before = dict(numerical)
                with patch.object(guide, "run_experiment", wraps=guide.run_experiment) as run:
                    results = guide.run_manufactured_convergence(
                        material, N_values=(4, 6), newton_atol=atol,
                        model="manufactured_nonlinear", accuracy=accuracy, delta="2", numerical=numerical,
                    )
                self.assertIsInstance(results, tuple)
                self.assertEqual([result.solution.N for result in results], [4, 6])
                self.assertTrue(all(result.solution.converged for result in results))
                for result in results:
                    self.assertEqual(result.accuracy, accuracy)
                    self.assertEqual(result.solution.physical.delta, 2)
                    self.assertEqual(result.settings["numerical"]["newton_atol"], atol)
                    self.assertFalse(result.settings["warm_start"])
                    self.assertEqual(result.solution.arithmetic.use_mpmath, accuracy == "high_precision")
                    self.assertEqual(result.solution.newton_result.stages[-1].target_norm,
                                     result.solution.arithmetic.scalar(atol))
                self.assertEqual(run.call_count, 2)
                self.assertTrue(all("initial_guess" not in (call.kwargs.get("numerical") or {})
                                    for call in run.call_args_list))
                self.assertEqual(numerical, before)
                table = guide.report_convergence_table(results, order=33)
                self.assertIn("Absolute L2", table)
                self.assertIn("Relative L2", table)
                self.assertIn("Residual norm (Euclidean)", table)
        self.assertEqual(material.rows, original_rows)

    def test_three_interface_linear_study_reduces_independent_field_errors(self):
        results = guide.run_manufactured_convergence(
            preset_material(3, kind="graded"), N_values=(8, 16), newton_atol="1e-10",
            model="manufactured_linear",
            numerical=dict(quadrature_atol="1e-8", quadrature_rtol="1e-8"),
        )
        self.assertTrue(all(result.solution.converged for result in results))
        errors = [guide.manufactured_error_metrics(result, order=49) for result in results]
        for name in FIELDS:
            with self.subTest(field=name):
                self.assertGreater(errors[0][name]["absolute"], 0)
                self.assertLess(errors[1][name]["absolute"], errors[0][name]["absolute"])
                self.assertLess(errors[1][name]["relative"], errors[0][name]["relative"])
        table = guide.report_convergence_table(results, order=49)
        self.assertIn("8 | 15 | 1e-10", table)
        self.assertIn("16 | 31 | 1e-10", table)

    def test_failed_rows_are_reported_as_provisional(self):
        results = guide.run_manufactured_convergence(
            preset_material(1), N_values=(4, 6), newton_atol="1e-10", model="manufactured_linear",
            numerical=dict(max_iterations=0, use_globalization=False),
        )
        self.assertTrue(all(not result.solution.converged for result in results))
        table = guide.report_convergence_table(results, order=25)
        self.assertGreaterEqual(table.count("PROVISIONAL"), 2)
        self.assertIn("PROVISIONAL", guide.report_error_table(results[0], order=25))

    def test_tables_validate_comparisons_and_accept_canonical_targets(self):
        bsk = guide.run_experiment(preset_material(1), N=4, newton_atol="1e-10")
        with self.assertRaisesRegex(ValueError, "exact|known|manufactured"):
            guide.manufactured_error_metrics(bsk)
        with self.assertRaisesRegex(ValueError, "known|manufactured"):
            guide.report_convergence_table((bsk,))
        variations = (dict(delta="1"), dict(newton_atol="1e-9"),
                      dict(numerical=dict(NUMERICAL, quadrature_atol="1e-7")),
                      dict(model="manufactured_nonlinear"), dict(accuracy="high_precision"))
        for changes in variations:
            options = dict(N=6, newton_atol="1e-10", model="manufactured_linear", delta="2",
                           numerical=NUMERICAL)
            options.update(changes)
            different = guide.run_experiment(self.material, **options)
            with self.subTest(changes=changes), patch.object(
                guide, "manufactured_error_metrics", side_effect=AssertionError("prevalidate comparisons")
            ), self.assertRaisesRegex(ValueError, "share|compatible"):
                guide.report_convergence_table((self.exact_space, different))

        one = guide.run_manufactured_convergence(self.material, N_values=(4,),
                                                newton_atol="1e-10", model="manufactured_linear")
        self.assertEqual(len(one), 1)
        self.assertIn("4 | 7 | 1e-10", guide.report_convergence_table(one, order=25))
        other = guide.run_experiment(self.material, N=6, newton_atol=1e-10,
                                     model="manufactured_linear", delta="2",
                                     numerical=dict(NUMERICAL, quadrature_order=18))
        table = guide.report_convergence_table((self.exact_space, other), order=31)
        self.assertIn("6 | 11 |", table)

    def test_optional_notebook_studies_execute_on_small_in_memory_sequences(self):
        original = NOTEBOOK.read_bytes()
        notebook = json.loads(original)
        namespace = dict(material=preset_material(3), delta="1", accuracy="standard", verbosity=0,
                         report_convergence_table=guide.report_convergence_table,
                         FluxProblem=FluxProblem, layered_transmission_case=layered_transmission_case)
        output = io.StringIO()
        with patch.object(guide, "run_defined_problem",
                          wraps=guide.run_defined_problem) as run, redirect_stdout(output):
            namespace["run_defined_problem"] = run
            found = []
            for cell in notebook["cells"]:
                if cell.get("id") not in ("linear-problem-definition", "nonlinear-problem-definition",
                                          "linear-resolution-study", "nonlinear-resolution-study"):
                    continue
                if cell["id"].endswith("resolution-study"):
                    found.append(cell["id"])
                tree = ast.fix_missing_locations(TinyNotebookStudies().visit(ast.parse("".join(cell["source"]))))
                exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
        self.assertEqual(len(found), 2)
        self.assertEqual(run.call_count, 4)
        self.assertEqual([call.kwargs["N"] for call in run.call_args_list], [4, 8, 4, 8])
        for name, problem_name in (("linear_convergence_results", "linear_problem"),
                                   ("nonlinear_convergence_results", "nonlinear_problem")):
            self.assertEqual([result.solution.N for result in namespace[name]], [4, 8])
            self.assertTrue(all(result.solution.converged for result in namespace[name]))
            for result in namespace[name]:
                self.assertIs(result.problem, namespace[problem_name])
                a = result.solution.arithmetic
                self.assertEqual(result.solution.physical.physical_rhs("0.4", "0.3"),
                                 result.problem.f(a.scalar("0.4"), a.scalar("0.3"), a))
        self.assertIn("Absolute L2", output.getvalue())
        self.assertIn("Relative L2", output.getvalue())
        self.assertEqual(NOTEBOOK.read_bytes(), original)
