"""Explicit physical projection, trace correction and one-way restart transfer."""

from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.collocation.solution import CollocationSolution
from overscreening_breakdown.problems.flux import FluxProblem
from overscreening_breakdown.projection import project_collocation
from overscreening_breakdown.quadrature import QuadratureFactory


def zero_source(x, u, arithmetic):
    return arithmetic.scalar(0)


class GalerkinProjectionTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for precision in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(precision)

    def tolerance(self, arithmetic):
        return arithmetic.scalar("2e-11" if not arithmetic.use_mpmath
                                 else "2e-{}".format(max(10, arithmetic.dps-7)))

    def problem(self, wall=1, q_b=-6, delta=2, domain=(0, "inf"), boundaries=None):
        return FluxProblem(
            delta=delta, f=zero_source, f_u=zero_source, domain=domain,
            boundary_conditions=(boundaries if boundaries is not None else
                                 ((wall, None, None, q_b, None), (None,)*5)))

    def source(self, arithmetic, entries=None, *, problem=None, L=1):
        problem = self.problem() if problem is None else problem
        physical = problem.bind(arithmetic)
        boundary_count = sum(value is not None for row in physical.boundary_conditions for value in row)
        if entries is None:
            wall = physical.boundary_conditions[0][0]
            entries = [wall/2, -wall/2]+[0]*boundary_count
        return CollocationSolution(
            physical=physical, coefficients=arithmetic.vector(entries),
            N=len(entries)-boundary_count, L=L)

    def test_exact_rational_pair_projects_without_spatial_error_at_every_precision(self):
        for arithmetic in self.arithmetics():
            with self.subTest(backend=arithmetic.backend, dps=arithmetic.dps):
                source = self.source(arithmetic)
                result = project_collocation(source, N=4)
                self.assertEqual(result.solver_type, "galerkin")
                self.assertEqual(result.method, "projection")
                self.assertFalse(result.converged)
                self.assertFalse(result.nonlinear_converged)
                self.assertFalse(result.quadrature_converged)
                self.assertIs(result.physical, source.physical)
                self.assertIs(result.arithmetic, arithmetic)
                expected = arithmetic.vector([0, 0, 0, "-0.375", "0.5", "-0.125", 0])
                self.assertLess(arithmetic.norm(result.coefficients-expected), self.tolerance(arithmetic))
                for raw_x in ("0", "0.125", "1", "4", "13"):
                    x = arithmetic.scalar(raw_x)
                    self.assertLess(abs(result.u(x)-1/(1+x)), self.tolerance(arithmetic))
                    self.assertLess(abs(result.w(x)+2/(1+x)**3), self.tolerance(arithmetic))
                self.assertEqual(result.u(arithmetic.inf), 0)
                self.assertEqual(result.w(arithmetic.inf), 0)
                self.assertIsInstance(result.u(0), type(arithmetic.scalar(0)))
                self.assertIsInstance(result.w(0), type(arithmetic.scalar(0)))

    def test_default_degree_contains_independent_second_field(self):
        arithmetic = NumpyArithmetic()
        entries = ["0.375", "-0.4375", "0.125", "-0.0625"]
        source = self.source(arithmetic, entries, problem=self.problem(q_b=-78))
        result = project_collocation(source)
        self.assertEqual(result.N, len(entries)+1)
        self.assertGreaterEqual(result.plan.order, result.N+1)
        for x in (0, 0.25, 1, 5):
            s = 1/(1+x)
            self.assertLess(abs(result.u(x)-(s-2*s*s+2*s**3)), 1e-10)
            self.assertLess(abs(result.w(x)-(-2*s**3+12*s**4-24*s**5)), 1e-9)

    def test_tiny_endpoint_corrections_are_explicit_factored_and_recorded(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            eps = (arithmetic.scalar(arithmetic.context.eps) if arithmetic.use_mpmath else
                   arithmetic.scalar(np.finfo(np.float64).eps))
            entries = [arithmetic.scalar("0.375")+eps, arithmetic.scalar("-0.4375"),
                       arithmetic.scalar("0.125"), arithmetic.scalar("-0.0625")]
            source = self.source(arithmetic, entries, problem=self.problem(q_b=-78))
            original = tuple(source.coefficients)
            self.assertNotEqual(source.u(arithmetic.inf), 0)
            result = project_collocation(source)
            info = dict(result.projection_info)
            self.assertEqual(info["source_solver"], "collocation")
            self.assertEqual(info["norm"], "physical_h1")
            self.assertEqual(info["far_adjustment"], -source.u(arithmetic.inf))
            self.assertLessEqual(abs(info["wall_adjustment"]), info["trace_tolerance"])
            self.assertEqual(result.u(0), 1)
            self.assertEqual(result.u(arithmetic.inf), 0)
            # The corrected result has a genuine decaying tail, even after y
            # rounds to 1. Subtracting the defect at evaluation time loses it.
            x = arithmetic.scalar("1e100")
            self.assertGreater(result.u(x), 0)
            self.assertLess(abs(result.u(x)*x-1), 100*self.tolerance(arithmetic))
            self.assertEqual(tuple(source.coefficients), original)

    def test_genuine_endpoint_defects_and_invalid_trace_tolerances_are_rejected(self):
        arithmetic = NumpyArithmetic()
        for entries in ([0.501, -0.5, 0, 0], [0.501, -0.501, 0, 0]):
            with self.assertRaisesRegex(ValueError, "endpoint defects"):
                project_collocation(self.source(arithmetic, entries), N=4)
        source = self.source(arithmetic, [0.501, -0.5, 0, 0])
        corrected = project_collocation(source, N=4, trace_tolerance="0.01")
        self.assertLess(abs(corrected.u(1)-0.5), 1e-11)
        self.assertGreater(abs(dict(corrected.projection_info)["far_adjustment"]), 0)
        for invalid in (-1, "nan", "inf"):
            with self.assertRaises(ValueError):
                project_collocation(self.source(arithmetic), trace_tolerance=invalid)

    def test_coarse_projection_satisfies_independent_physical_H1_orthogonality(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(50)):
            source = self.source(arithmetic, ["0.375", "-0.4375", "0.125", "-0.0625"],
                                 problem=self.problem(q_b=-78))
            result = project_collocation(source, N=2)
            validation = QuadratureFactory(arithmetic).plan(9)
            for values, derivatives, source_value, source_dx, target_value, target_dx in (
                (result.basis.v, result.basis.v_dx, source.u, source.u_dx, result.u, result.u_dx),
                (result.basis.q, result.basis.q_dx, source.w, source.w_dx, result.w, result.w_dx),
            ):
                for index in range(len(values(0))):
                    moment = sum((weight*((source_value(x)-target_value(x))*values(y)[index]
                                          +(source_dx(x)-target_dx(x))*derivatives(y)[index])
                                  for y, x, weight in zip(validation.nodes, validation.physical_nodes,
                                                         validation.physical_weights)), arithmetic.scalar(0))
                    self.assertLess(abs(moment), 100*self.tolerance(arithmetic))

    def test_different_mapping_uses_physical_coordinates_and_explicit_quadrature(self):
        arithmetic = NumpyArithmetic()
        source = self.source(arithmetic)
        result = project_collocation(source, N=8, L=2, quadrature_order=48, subdivisions=2)
        self.assertEqual(result.L, 2)
        self.assertFalse(dict(result.projection_info)["same_mapping"])
        self.assertEqual(result.plan.order, 48)
        self.assertEqual(result.plan.subdivisions, 2)
        for x in (0, 0.1, 0.5, 1, 4):
            self.assertLess(abs(result.u(x)-1/(1+x)), 1e-4)
            self.assertLess(abs(result.w(x)+2/(1+x)**3), 1e-3)

    def test_120_digit_parameters_and_coefficients_do_not_leak_float64(self):
        arithmetic = MpmathArithmetic(120)
        correction = arithmetic.scalar("1e-80")
        wall = 1+correction
        length = "1."+"0"*79+"1"
        length_value = arithmetic.scalar(length)
        problem = self.problem(wall=wall, q_b=-6*wall/length_value**3)
        source = self.source(arithmetic, [wall/2, -wall/2, 0, 0], problem=problem, L=length)
        precision = mp.mp.prec
        result = project_collocation(source, N=4)
        self.assertEqual(mp.mp.prec, precision)
        self.assertGreater(result.L, 1)
        self.assertLess(abs((result.u(0)-1)-correction), arithmetic.scalar("1e-118"))
        x = arithmetic.scalar(1)
        expected_u = wall*length_value/(x+length_value)
        expected_w = -2*wall*length_value/(x+length_value)**3
        self.assertLess(abs(result.u(x)-expected_u), arithmetic.scalar("1e-112"))
        self.assertLess(abs(result.w(x)-expected_w), arithmetic.scalar("1e-112"))

    def test_exponential_collocation_fields_transfer_as_a_mixed_restart(self):
        from overscreening_breakdown.collocation.spectral1d import solve_nonlinear_problem
        from overscreening_breakdown.solvers import solve

        original_precision = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_precision)
        problem = FluxProblem(
            delta=2, f=lambda x, u, arithmetic: u-4*arithmetic.exp(-x),
            f_u=lambda x, u, arithmetic: arithmetic.scalar(1),
            boundary_conditions=((1, None, None, -1, None), (0, None, None, None, None)),
        )
        solver = solve_nonlinear_problem(12, problem=problem, L=2,
                                         tolerance=1e-11, use_globalization=False)
        with redirect_stdout(io.StringIO()):
            coefficients = solver.solve_problem()
        self.assertTrue(solver.is_converged())
        arithmetic = NumpyArithmetic()
        source = CollocationSolution(physical=problem.bind(arithmetic),
                                     coefficients=coefficients, N=12, L=2)
        result = project_collocation(source)
        resumed = solve(problem, result.N, solver="galerkin", L=result.L,
                        initial_guess=result, tolerance="1e-10", use_globalization=False)
        self.assertTrue(resumed.converged)
        self.assertEqual(resumed.method, "newton")
        for x in (0, 0.1, 0.5, 1, 3, 6):
            self.assertLess(abs(result.u(x)-source.u(x)), 1e-8)
            self.assertLess(abs(result.w(x)-source.w(x)), 1e-7)
            self.assertLess(abs(result.u(x)-np.exp(-x)), 0.005)
            self.assertLess(abs(resumed.u(x)-np.exp(-x)), 0.005)

    def test_incompatible_domains_orders_and_wall_data_are_rejected(self):
        arithmetic = NumpyArithmetic()
        problems = (
            self.problem(delta=0, boundaries=((1, None, None, None, None), (None,)*5)),
            self.problem(domain=(0, 2), boundaries=((1, None, None, -6, None),
                                                  (0, None, 0, None, None))),
            self.problem(boundaries=((1, -1, None, None, None), (None,)*5)),
        )
        for problem in problems:
            with self.assertRaises(ValueError):
                project_collocation(self.source(arithmetic, problem=problem))

    def test_reverse_projection_foreign_contexts_stale_sources_and_bad_geometry_fail(self):
        arithmetic = MpmathArithmetic(30)
        source = self.source(arithmetic)
        result = project_collocation(source, arithmetic=arithmetic)
        with self.assertRaises(TypeError):
            project_collocation(result)
        with self.assertRaises(TypeError):
            project_collocation(source.coefficients)
        with self.assertRaisesRegex(ValueError, "arithmetic owner"):
            project_collocation(source, arithmetic=MpmathArithmetic(30))
        for arguments in (dict(N=1), dict(L=0), dict(quadrature_order=1), dict(subdivisions=0)):
            with self.assertRaises(ValueError):
                project_collocation(source, **arguments)
        arithmetic.context.prec += 1
        with self.assertRaises(RuntimeError):
            project_collocation(source)

    def test_explicit_typed_file_projection_roundtrip_preserves_provenance(self):
        from overscreening_breakdown.solution_io import load_solution, save_solution

        arithmetic = MpmathArithmetic(50)
        description = self.problem()
        source = self.source(arithmetic, problem=description)
        with tempfile.TemporaryDirectory(prefix="overscreening-project-") as directory:
            collocation_path = Path(directory)/"collocation.json"
            galerkin_path = Path(directory)/"galerkin.json"
            save_solution(source, collocation_path)
            restored_source = load_solution(collocation_path, problem=description)
            projected = project_collocation(restored_source, N=4)
            save_solution(projected, galerkin_path)
            restored = load_solution(galerkin_path, problem=description)
            self.assertEqual(restored.solver_type, "galerkin")
            self.assertEqual(restored.method, "projection")
            self.assertFalse(restored.converged)
            self.assertEqual(restored.projection_info, projected.projection_info)
            self.assertEqual(tuple(restored.coefficients), tuple(projected.coefficients))
            self.assertLess(abs(restored.u(1)-restored.arithmetic.scalar("0.5")),
                            restored.arithmetic.scalar("1e-43"))


if __name__ == "__main__":
    unittest.main()
