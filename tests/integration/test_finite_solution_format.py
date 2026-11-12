"""Finite affine snapshots remain distinct and retain their native geometry."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import mpmath as mp

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.galerkin.finite_basis import FiniteGalerkinBasis
from overscreening_breakdown.galerkin.finite_solution import FiniteGalerkinSolution
from overscreening_breakdown.problems.flux import FluxProblem
from overscreening_breakdown.quadrature import QuadratureFactory
from overscreening_breakdown.solution_io import FINITE_BASIS_ID, load_solution


class FiniteSolutionFormatTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="overscreening-finite-format-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)
        self.path = Path(temporary.name)/"solution.json"
        self.coordinates = []

    def problem(self, R="2"):
        def epsilon(x, a):
            if not 0 <= x <= a.scalar(R):
                raise AssertionError("finite coefficient callback evaluated outside its domain")
            self.coordinates.append(x)
            return a.scalar(1 if x < a.scalar("0.75") else 2)

        return FluxProblem(delta="2", domain=(0, R),
                           epsilon=Coefficient(epsilon, lower_bound=1, upper_bound=2,
                                               wall_value=1, interfaces=("0.75",)),
                           f=lambda x, u, a: u, f_u=lambda x, u, a: a.scalar(1),
                           boundary_conditions=((1, None, None, 0, None), (0, 0, None, None, None)))

    def solution(self, arithmetic=None, *, R="2", plan=True, **status):
        a = NumpyArithmetic() if arithmetic is None else arithmetic
        problem = self.problem(R)
        basis = FiniteGalerkinBasis(4, a, domain=(0, R))
        rule = QuadratureFactory(a, domain=(0, R), interfaces=("0.75",)).plan(6)
        entries = ["-0.0", "0.123456789012345678901234567890123456789012345678901234567890123456789",
                   "-0.25", "-0.375", "0.5", "-0.125", "0", "0.0625"]
        return problem, FiniteGalerkinSolution(basis, problem.bind(a), entries,
                                               plan=rule if plan else None, **status)

    def test_roundtrip_preserves_both_fields_geometry_and_exact_mpmath_bits(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(30), MpmathArithmetic(100)):
            with self.subTest(mpmath=a.use_mpmath, digits=a.dps):
                if a.use_mpmath:
                    a.context.prec += 1
                problem, original = self.solution(a)
                original.save(self.path)
                record = json.loads(self.path.read_text())
                self.assertEqual(record["basis"]["id"], FINITE_BASIS_ID)
                self.assertEqual(record["basis"]["unknown_count"], 8)
                self.assertEqual(record["geometry"]["mapping_type"], "affine_interval")
                self.assertNotIn("L", record["geometry"])
                self.assertEqual((len(record["coefficients"]["u"]), len(record["coefficients"]["w"])), (3, 5))
                loaded = load_solution(self.path, problem=problem, solver="galerkin")
                self.assertIsInstance(loaded, FiniteGalerkinSolution)
                self.assertEqual(loaded.domain, original.domain)
                self.assertEqual(loaded.basis.mapping.raw_domain, original.basis.mapping.raw_domain)
                self.assertEqual(tuple(loaded.coefficients), tuple(original.coefficients))
                if a.use_mpmath:
                    self.assertEqual(loaded.arithmetic.context.prec, a.context.prec)
                for field in ("u", "w", "u_dx", "w_dx", "flux"):
                    for x in ("0", "0.125", "1", "2"):
                        self.assertEqual(getattr(loaded, field)(x), getattr(original, field)(x))

    def test_no_plan_roundtrip_samples_only_inside_the_finite_callback_domain(self):
        problem, original = self.solution(plan=False)
        original.save(self.path)
        loaded = load_solution(self.path, problem=problem)
        self.assertIsNone(loaded.plan)
        self.assertTrue(self.coordinates)
        self.assertTrue(all(0 <= x <= 2 for x in self.coordinates))
        self.assertEqual(tuple(loaded.coefficients), tuple(original.coefficients))

    def test_raw_right_endpoint_survives_arithmetic_promotion_and_flags_are_cleared(self):
        R = "2."+"0"*79+"1"
        problem, original = self.solution(R=R, converged=True, nonlinear_converged=True,
                                           quadrature_converged=True)
        original.save(self.path)
        exact = load_solution(self.path, problem=problem)
        self.assertTrue(exact.converged)
        self.assertEqual(exact.domain[1], 2)
        promoted = load_solution(self.path, problem=problem, use_mpmath=True, precision=120)
        a = promoted.arithmetic
        self.assertGreater(promoted.domain[1], 2)
        self.assertLess(abs(promoted.domain[1]-2-a.scalar("1e-80")), a.scalar("1e-115"))
        self.assertEqual(promoted.basis.mapping.raw_domain[1], R)
        self.assertFalse(promoted.converged or promoted.nonlinear_converged or promoted.quadrature_converged)
        self.assertEqual(promoted.plan.mapping.domain, promoted.domain)

    def test_basis_mapping_domain_solver_and_layout_mismatches_are_rejected(self):
        problem, original = self.solution()
        original.save(self.path)
        valid = json.loads(self.path.read_text())
        bad_records = []
        for field, value in (("id", "rational_chebyshev_mixed"), ("unknown_count", 7)):
            record = copy.deepcopy(valid)
            record["basis"][field] = value
            bad_records.append(record)
        record = copy.deepcopy(valid)
        record["geometry"]["mapping_type"] = "rational_half_line"
        bad_records.append(record)
        record = copy.deepcopy(valid)
        record["geometry"]["raw_domain"][1] = {"type": "integer", "value": "3"}
        bad_records.append(record)
        record = copy.deepcopy(valid)
        record["coefficients"]["w"].pop()
        bad_records.append(record)
        for index, record in enumerate(bad_records):
            with self.subTest(record=index), self.assertRaises(ValueError):
                self.path.write_text(json.dumps(record))
                load_solution(self.path, problem=problem)
        self.path.write_text(json.dumps(valid))
        with self.assertRaisesRegex(ValueError, "solver"):
            load_solution(self.path, problem=problem, solver="collocation")
        with self.assertRaisesRegex(ValueError, "domain"):
            load_solution(self.path, problem=self.problem("3"))
