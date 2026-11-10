"""Exact, data-only snapshots and explicit solver compatibility."""

from contextlib import redirect_stdout
from decimal import Decimal
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np

from arithmetic import MpmathArithmetic, NumpyArithmetic
from coefficients import Coefficient
from collocation_solution import CollocationSolution
from galerkin import GalerkinBasis
from physical_problem import FluxProblem
from quadrature import QuadratureFactory
from solution_io import load_solution, save_solution
from solutions import GalerkinSolution
from spectral1d import solve_nonlinear_problem


class SolutionFormatTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="overscreening-format-", dir="/tmp")
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory/"solution.json"

    def problem(self, **options):
        defaults = {
            "delta": "2", "f": lambda x, u, a: u,
            "f_u": lambda x, u, a: a.scalar(1),
            "boundary_conditions": (("1", None, None, "-6", None), (None,)*5),
        }
        defaults.update(options)
        return FluxProblem(**defaults)

    def solution(self, arithmetic=None, problem=None, L="1", plan=True, **options):
        arithmetic = NumpyArithmetic() if arithmetic is None else arithmetic
        problem = self.problem() if problem is None else problem
        basis = GalerkinBasis(4, arithmetic, L=L)
        physical = problem.bind(arithmetic)
        rule = QuadratureFactory(arithmetic, L=L, interfaces=physical.epsilon.interfaces).plan(5)
        entries = arithmetic.vector(["-0.0", "0.125", "-0.25", "-0.375", "0.5", "-0.125", "0"])
        return GalerkinSolution(basis, physical, entries, plan=rule if plan else None, **options)

    def collocation(self, arithmetic=None, problem=None, **options):
        arithmetic = NumpyArithmetic() if arithmetic is None else arithmetic
        problem = self.problem() if problem is None else problem
        entries = arithmetic.vector(["-0.0", "0.125", "-0.25", "-0.375", "0.5", "-0.125"])
        return CollocationSolution(physical=problem.bind(arithmetic), coefficients=entries, N=4, **options)

    def edit_record(self, callback):
        record = json.loads(self.path.read_text())
        callback(record)
        self.path.write_text(json.dumps(record))

    def test_numpy_roundtrip_preserves_exact_bits_and_signed_zero_for_both_solvers(self):
        problem = self.problem()
        for solution in (self.solution(problem=problem), self.collocation(problem=problem)):
            with self.subTest(solver=solution.solver_type):
                solution.save(self.path)
                loaded = load_solution(self.path, problem=problem, solver=solution.solver_type)
                self.assertEqual(loaded.solver_type, solution.solver_type)
                self.assertEqual(loaded.unknown_count, solution.unknown_count)
                np.testing.assert_array_equal(loaded.coefficients.view(np.uint64),
                                              solution.coefficients.view(np.uint64))
                self.assertTrue(np.signbit(loaded.coefficients[0]))
                for x in (0, "0.125", "2", "1e30"):
                    self.assertEqual(loaded.u(x), solution.u(x))
                    self.assertEqual(loaded.w(x), solution.w(x))
                record = json.loads(self.path.read_text())
                self.assertEqual(record["format_version"], 2)
                self.assertEqual(record["coefficients"]["u"][0]["value"], "-0x0.0p+0")

    def test_mpmath_roundtrip_preserves_every_bit_and_noncanonical_working_precision(self):
        global_bits = mp.mp.prec
        problem = self.problem()
        for digits in (15, 30, 50, 100, 120):
            arithmetic = MpmathArithmetic(digits)
            arithmetic.context.prec += 1
            original = self.solution(arithmetic, problem)
            entries = list(original.coefficients)
            entries[0] = arithmetic.sqrt(arithmetic.scalar(2))
            entries[1] = arithmetic.scalar(arithmetic.pi)
            original = GalerkinSolution(original.basis, original.physical, entries, plan=original.plan)
            with self.subTest(dps=digits):
                save_solution(original, self.path)
                loaded = load_solution(self.path, problem=problem)
                self.assertEqual(loaded.arithmetic.context.prec, arithmetic.context.prec)
                self.assertEqual([value._mpf_ for value in loaded.coefficients],
                                 [value._mpf_ for value in original.coefficients])
                self.assertIsNot(loaded.arithmetic, arithmetic)
                self.assertEqual(mp.mp.prec, global_bits)
        # A global precision change does not alter either solution or its save.
        try:
            mp.mp.dps = 15
            save_solution(original, self.path)
            loaded = load_solution(self.path, problem=problem)
            self.assertEqual(loaded.arithmetic.context.prec, arithmetic.context.prec)
            self.assertEqual([value._mpf_ for value in loaded.coefficients],
                             [value._mpf_ for value in original.coefficients])
        finally:
            mp.mp.prec = global_bits

    def test_collocation_mpmath_coefficients_restore_without_global_precision_changes(self):
        arithmetic = MpmathArithmetic(120)
        problem = self.problem()
        original = self.collocation(arithmetic, problem)
        bits = mp.mp.prec
        save_solution(original, self.path)
        loaded = load_solution(self.path, problem=problem)
        self.assertEqual([value._mpf_ for value in loaded.coefficients],
                         [value._mpf_ for value in original.coefficients])
        self.assertEqual(mp.mp.prec, bits)

    def test_explicit_backend_and_precision_conversions_round_and_clear_flags(self):
        problem = self.problem()
        arithmetic = MpmathArithmetic(120)
        original = self.solution(arithmetic, problem, converged=True,
                                 nonlinear_converged=True, quadrature_converged=True)
        entries = list(original.coefficients)
        entries[0] = arithmetic.sqrt(arithmetic.scalar(2))
        original = GalerkinSolution(original.basis, original.physical, entries, plan=original.plan,
                                    converged=True, nonlinear_converged=True, quadrature_converged=True)
        save_solution(original, self.path)
        lower = load_solution(self.path, problem=problem, precision=20)
        self.assertEqual(lower.arithmetic.dps, 20)
        self.assertFalse(lower.converged)
        self.assertFalse(lower.nonlinear_converged)
        self.assertFalse(lower.quadrature_converged)
        for value, source in zip(lower.coefficients, original.coefficients):
            self.assertEqual(value, lower.arithmetic.scalar(source))
        numpy = load_solution(self.path, problem=problem, use_mpmath=False)
        self.assertEqual(numpy.coefficients.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(numpy.coefficients,
                                      np.array([float(value) for value in original.coefficients]))
        self.assertFalse(numpy.converged)
        exact = load_solution(self.path, problem=problem)
        self.assertTrue(exact.converged)
        self.assertEqual([value._mpf_ for value in exact.coefficients],
                         [value._mpf_ for value in original.coefficients])
        save_solution(self.solution(problem=problem), self.path)
        promoted = load_solution(self.path, problem=problem, use_mpmath=True, precision=60)
        self.assertEqual(promoted.arithmetic.dps, 60)
        self.assertTrue(all(hasattr(value, "_mpf_") for value in promoted.coefficients))

    def test_explicit_same_arithmetic_retains_saved_provenance_flags(self):
        problem = self.problem()
        original = self.solution(MpmathArithmetic(50), problem, converged=True,
                                 nonlinear_converged=True, quadrature_converged=True)
        save_solution(original, self.path)
        loaded = load_solution(self.path, problem=problem, use_mpmath=True, precision=50)
        self.assertTrue(loaded.converged)
        self.assertTrue(loaded.nonlinear_converged)
        self.assertTrue(loaded.quadrature_converged)
        self.assertIsNone(loaded.newton_result)
        self.assertEqual(loaded.quadrature_history, ())

    def test_raw_decimal_geometry_and_physical_parameters_survive_precision_upgrade(self):
        length = "1."+"0"*79+"1"
        problem = self.problem(delta=length)
        original = self.solution(MpmathArithmetic(30), problem, L=Decimal(length))
        self.assertEqual(original.L, 1)
        save_solution(original, self.path)
        exact = load_solution(self.path, problem=problem)
        self.assertEqual(exact.L, 1)
        upgraded = load_solution(self.path, problem=problem, precision=120)
        self.assertGreater(upgraded.L, 1)
        self.assertGreater(upgraded.physical.delta, 1)
        self.assertLess(abs((upgraded.L-1)-upgraded.arithmetic.scalar("1e-80")),
                        upgraded.arithmetic.scalar("1e-118"))

    def test_quadrature_geometry_preserves_extra_panels_subdivisions_and_raw_inputs(self):
        problem = self.problem()
        arithmetic = MpmathArithmetic(50)
        original = self.solution(arithmetic, problem)
        plan = QuadratureFactory(arithmetic, L=1, interfaces=("1", Decimal("3"))).plan(6, 2)
        original = GalerkinSolution(original.basis, original.physical, original.coefficients, plan=plan)
        save_solution(original, self.path)
        loaded = load_solution(self.path, problem=problem)
        self.assertEqual((loaded.plan.order, loaded.plan.subdivisions), (6, 2))
        self.assertEqual(loaded.plan.panels, plan.panels)
        self.assertEqual(loaded.plan.physical_nodes, plan.physical_nodes)
        self.assertEqual(loaded.plan.raw_interfaces, ("1", "3"))
        self.assertIs(loaded.plan.arithmetic, loaded.arithmetic)
        self.assertEqual(plan.raw_interfaces, ("1", Decimal("3")))

    def test_planless_mixed_fields_can_restore_without_assembling_or_solving(self):
        problem = self.problem()
        original = self.solution(MpmathArithmetic(30), problem, plan=False)
        save_solution(original, self.path)
        with patch("nonlinear_solver.solve_newton", side_effect=AssertionError("must not solve")):
            loaded = load_solution(self.path, problem=problem)
        self.assertIsNone(loaded.plan)
        self.assertEqual(loaded.u(2), original.u(2))
        self.assertEqual(loaded.w(2), original.w(2))

    def test_finite_interval_collocation_restores_its_domain_and_fields(self):
        problem = self.problem(delta=0, domain=(0, "3"),
                               boundary_conditions=((1, None, None, None, None),
                                                    (0, None, None, None, None)))
        original = self.collocation(MpmathArithmetic(50), problem)
        save_solution(original, self.path)
        loaded = load_solution(self.path, problem=problem)
        self.assertEqual(loaded.domain, original.domain)
        self.assertEqual(loaded.u("0.75"), original.u("0.75"))
        self.assertEqual(loaded.w("0.75"), original.w("0.75"))

    def test_wrong_solver_requests_are_rejected_in_both_directions(self):
        problem = self.problem()
        for solution, wrong in ((self.solution(problem=problem), "collocation"),
                                (self.collocation(problem=problem), "galerkin")):
            with self.subTest(solver=solution.solver_type):
                save_solution(solution, self.path)
                with self.assertRaisesRegex(ValueError, "solver type.*explicit projection"):
                    load_solution(self.path, problem=problem, solver=wrong)

    def test_problem_metadata_must_match_before_a_requested_conversion(self):
        original = self.solution(MpmathArithmetic(50))
        save_solution(original, self.path)
        alternatives = (
            self.problem(delta=3), self.problem(domain=(0, 5)),
            self.problem(boundary_conditions=((2, None, None, -6, None), (None,)*5)),
            self.problem(epsilon=2),
        )
        for problem in alternatives:
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                load_solution(self.path, problem=problem, use_mpmath=False)

    def test_declared_epsilon_callback_is_checked_at_all_saved_nodes(self):
        def coefficient(scale):
            return Coefficient(lambda x, a: 1+a.scalar(scale)*x/(1+x),
                               lower_bound=1, upper_bound=2, wall_value=1, interfaces=(1,))
        problem = self.problem(epsilon=coefficient("0.125"))
        original = self.solution(MpmathArithmetic(50), problem)
        save_solution(original, self.path)
        loaded = load_solution(self.path, problem=problem)
        self.assertEqual(loaded.physical.epsilon(2), original.physical.epsilon(2))
        with self.assertRaisesRegex(ValueError, "callback.*samples"):
            load_solution(self.path, problem=self.problem(epsilon=coefficient("0.25")))
        with self.assertRaisesRegex(ValueError, "interfaces"):
            load_solution(self.path, problem=self.problem(epsilon=Coefficient(
                lambda x, a: 1+a.scalar("0.125")*x/(1+x), lower_bound=1, upper_bound=2,
                wall_value=1, interfaces=(2,),
            )))

    def test_source_callbacks_are_supplied_again_and_not_identity_certified(self):
        problem = self.problem()
        save_solution(self.solution(problem=problem), self.path)
        new_problem = self.problem(f=lambda x, u, a: 7*u,
                                   f_u=lambda x, u, a: a.scalar(7))
        loaded = load_solution(self.path, problem=new_problem)
        self.assertEqual(loaded.physical.physical_rhs(1, 2), 14)
        text = self.path.read_text()
        self.assertNotIn("lambda", text)
        self.assertNotIn("__reduce__", text)
        self.assertEqual(json.loads(text)["callback_verification"], "coefficient_samples_only")

    def test_projection_provenance_restores_portable_high_precision_metadata(self):
        arithmetic = MpmathArithmetic(120)
        problem = self.problem()
        original = self.solution(arithmetic, problem, method="projection", projection_info=(
            ("source_solver", "collocation"), ("source_N", 8), ("norm", "physical_h1"),
            ("wall_defect", arithmetic.scalar("1e-80")),
            ("corrections", (arithmetic.scalar("1e-90"), arithmetic.scalar("-1e-90"))),
        ))
        save_solution(original, self.path)
        loaded = load_solution(self.path, problem=problem)
        self.assertEqual(loaded.method, "projection")
        self.assertEqual(loaded.projection_info, original.projection_info)
        converted = load_solution(self.path, problem=problem, precision=30)
        self.assertEqual(dict(converted.projection_info)["source_solver"], "collocation")

    def test_malformed_solver_basis_geometry_and_status_metadata_are_rejected(self):
        problem = self.problem()
        original = self.solution(problem=problem)
        mutations = (
            lambda r: r.update(format_version=99),
            lambda r: r.update(format_version=2.0),
            lambda r: r.update(solver_type="unknown"),
            lambda r: r.update(solver_type=[]),
            lambda r: r["basis"].update(id="cosine_tau"),
            lambda r: r["basis"].update(version=True),
            lambda r: r["basis"].update(N=True),
            lambda r: r["basis"].update(unknown_count=8),
            lambda r: r["geometry"].update(L={"type": "float64", "value": "0x0p+0"}),
            lambda r: r["coefficients"]["u"].pop(),
            lambda r: r["status"].update(converged=1),
            lambda r: r["status"].update(quadrature_converged=None),
            lambda r: r["status"].update(method="root"),
            lambda r: r.update(unrecognized="field"),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                save_solution(original, self.path)
                self.edit_record(mutation)
                with self.assertRaises(ValueError):
                    load_solution(self.path, problem=problem)

    def test_malformed_binary_scalars_and_arithmetic_metadata_are_rejected(self):
        problem = self.problem()
        original = self.solution(MpmathArithmetic(50), problem)
        mutations = (
            lambda r: r["arithmetic"].update(precision_bits=0),
            lambda r: r["arithmetic"].update(precision_dps=49),
            lambda r: r["coefficients"]["u"].__setitem__(0, {"type": "mpf", "value": [0, "2", 0, 2]}),
            lambda r: r["coefficients"]["u"].__setitem__(0, {"type": "mpf", "value": [1, "0", 0, 0]}),
            lambda r: r["coefficients"]["u"].__setitem__(0, {"type": "mpf", "value": [0, "1", 0, 2]}),
            lambda r: r["coefficients"]["u"].__setitem__(0, {"type": "mpf", "value": [0, str(2**200+1), -200, 201]}),
            lambda r: r["coefficients"]["u"].__setitem__(0, {"type": "infinity", "value": 1}),
            lambda r: r["coefficients"]["u"].__setitem__(0, {"type": "float64", "value": "nan"}),
            lambda r: r["coefficients"]["u"].__setitem__(0, {"type": "float64", "value": "0x1p+100000"}),
            lambda r: r["coefficients"]["u"].__setitem__(0, {"type": "float64", "value": "0x1p+0"}),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                save_solution(original, self.path)
                self.edit_record(mutation)
                with self.assertRaises(ValueError):
                    load_solution(self.path, problem=problem)

    def test_malformed_quadrature_and_sample_metadata_are_rejected(self):
        problem = self.problem()
        original = self.solution(problem=problem)
        mutations = (
            lambda r: r["quadrature"].update(order=4),
            lambda r: r["quadrature"].update(subdivisions=0),
            lambda r: r["quadrature"].update(panels=[]),
            lambda r: r["epsilon_samples"].pop(),
            lambda r: r["epsilon_samples"][0].update(x={"type": "float64", "value": "0x0p+0"}),
            lambda r: r["epsilon_samples"][0].update(value={"type": "float64", "value": "0x1p+1"}),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                save_solution(original, self.path)
                self.edit_record(mutation)
                with self.assertRaises(ValueError):
                    load_solution(self.path, problem=problem)

    def test_duplicate_nonstandard_and_legacy_data_are_rejected_by_json_loader(self):
        problem = self.problem()
        for text in ('{"format":1,"format":2}', '{"value":NaN}', '[]', '{broken'):
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaises(ValueError):
                    load_solution(self.path, problem=problem)
        self.path.write_bytes(pickle.dumps(np.array([1, 2])))
        with self.assertRaisesRegex(ValueError, "legacy coefficient"):
            load_solution(self.path, problem=problem)

    def test_problem_is_required_unbound_and_conversion_options_are_validated(self):
        problem = self.problem()
        original = self.solution(problem=problem)
        save_solution(original, self.path)
        with self.assertRaises(TypeError):
            load_solution(self.path)
        with self.assertRaises(TypeError):
            load_solution(self.path, problem=original.physical)
        for options in ({"use_mpmath": 1}, {"precision": True}, {"precision": 0},
                        {"precision": 30}, {"use_mpmath": False, "precision": 30}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                load_solution(self.path, problem=problem, **options)

    def test_stale_precision_snapshots_cannot_be_saved(self):
        arithmetic = MpmathArithmetic(30)
        original = self.solution(arithmetic)
        arithmetic.context.prec += 1
        with self.assertRaises(RuntimeError):
            save_solution(original, self.path)

    def test_legacy_writer_tags_collocation_and_untagged_files_remain_compatible(self):
        with redirect_stdout(io.StringIO()):
            solver = solve_nonlinear_problem(4, domain=[0, np.inf])
            path = self.directory/"legacy.pickle"
            solver.save_solution(path, [1, 2, 3])
            with path.open("rb") as handle:
                record = pickle.load(handle)
            self.assertEqual(record["solver_type"], "collocation")
            del record["solver_type"]
            path.write_bytes(pickle.dumps(record))
            np.testing.assert_array_equal(solver.load_data(path), [1, 2, 3])
            path.write_bytes(pickle.dumps(np.array([1, 2, 3])))
            np.testing.assert_array_equal(solver.load_data(path), [1, 2, 3])

    def test_legacy_reader_rejects_wrong_solver_before_flattening(self):
        with redirect_stdout(io.StringIO()):
            solver = solve_nonlinear_problem(4, domain=[0, np.inf])
            self.path.write_bytes(pickle.dumps({"format_version": 999,
                                                "solver_type": "galerkin", "coefficients": [1]}))
            with self.assertRaisesRegex(ValueError, "different solver type"):
                solver.load_data(self.path)
            save_solution(self.solution(), self.path)
            for action in (solver.load_data, solver.set_base_solution):
                with self.assertRaisesRegex(ValueError, "different solver type"):
                    action(self.path)
            save_solution(self.collocation(), self.path)
            with self.assertRaisesRegex(ValueError, "solution_io.load_solution"):
                solver.load_data(self.path)


if __name__ == "__main__":
    unittest.main()
