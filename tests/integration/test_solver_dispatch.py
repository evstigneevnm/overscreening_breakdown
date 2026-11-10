"""Public dispatch, owned collocation snapshots, and checked mixed restarts."""

from contextlib import redirect_stdout
import io
import pickle
from pathlib import Path
import tempfile
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.collocation.solution import CollocationSolution
from overscreening_breakdown.problems.flux import FluxProblem
from overscreening_breakdown.solvers import load_legacy_collocation, solve
from overscreening_breakdown.manufactured import rational_case


def make_problem():
    return rational_case(nonlinear=True, beta="0.1", initial="square").make_problem()


def variable_problem():
    return rational_case("smooth", nonlinear=True).make_problem()


def collocation_problem():
    return rational_case(nonlinear=True, beta="0.1", initial="exp").make_problem()


class SolverDispatchTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)

    def test_common_entry_covers_collocation_methods_and_all_galerkin_precisions(self):
        problem = collocation_problem()
        configurations = [(False, 15, method) for method in
                          ("newton", "root", "root_no_jac", "nonlin_solve")]
        configurations += [(True, digits, "newton") for digits in (15, 30, 50, 100, 120)]
        for use_mp, digits, method in configurations:
            with self.subTest(mpmath=use_mp, precision=digits, method=method):
                tolerance = "1e-{}".format(max(10, digits-10)) if use_mp else "1e-10"
                collocation = solve(problem, 6, method=method, use_mpmath=use_mp,
                                    precision=digits, tolerance=tolerance, use_globalization=False)
                galerkin = solve(problem, 4, solver="galerkin", use_mpmath=use_mp,
                                 precision=digits, tolerance=tolerance, use_globalization=False)
                self.assertTrue(collocation.converged)
                self.assertTrue(galerkin.converged)
                self.assertEqual(collocation.solver_type, "collocation")
                self.assertEqual(galerkin.solver_type, "galerkin")
                self.assertEqual(galerkin.unknown_count, 7)
                a = galerkin.arithmetic
                bound = a.scalar(tolerance)*10000
                for text in ("0", "0.125", "1", "10"):
                    x = a.scalar(text)
                    self.assertLess(abs(galerkin.u(x)-1/(1+x)), bound)
                    self.assertLess(abs(galerkin.w(x)+2/(1+x)**3), bound)
                    self.assertLess(abs(a.scalar(collocation.u(x))-galerkin.u(x)), bound)

    def test_finite_second_order_collocation_remains_available(self):
        problem = FluxProblem(
            delta=0, domain=(0, 1),
            f=lambda x, u, a: -2+u-x*(1-x), f_u=lambda x, u, a: a.scalar(1),
            boundary_conditions=((0, None, None, None, None), (0, None, None, None, None)),
        )
        for use_mp in (False, True):
            result = solve(problem, 4, use_mpmath=use_mp, precision=50, tolerance="1e-30" if use_mp else "1e-10")
            self.assertTrue(result.converged)
            for text in ("0", "0.125", "0.5", "1"):
                x = result.arithmetic.scalar(text)
                bound = result.arithmetic.scalar("1e-28" if use_mp else "1e-9")
                self.assertLess(abs(result.u(x)-x*(1-x)), bound)
                self.assertLess(abs(result.w(x)-2), bound)
                self.assertLess(abs(result.flux(x)-(1-2*x)), bound)
        with self.assertRaises(ValueError):
            solve(problem, 4, solver="galerkin")

    def test_incompatible_coefficient_method_and_bound_views_fail_explicitly(self):
        for options in ({"solver": "unknown"}, {"method": "missing"},
                        {"solver": "galerkin", "method": "root"},
                        {"use_mpmath": True, "method": "root_no_jac"}):
            with self.assertRaises(ValueError):
                solve(make_problem(), 4, **options)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            solve(variable_problem(), 8)
        with self.assertRaises(TypeError):
            solve(make_problem().bind(NumpyArithmetic()), 4)

    def test_quadrature_failure_and_refinement_are_distinct_from_newton(self):
        problem = rational_case("exponential", nonlinear=True).make_problem()
        coarse = solve(problem, 4, solver="galerkin", quadrature_order=5,
                       quadrature_atol="1e-10", quadrature_rtol="1e-10", max_refinements=0)
        self.assertTrue(coarse.nonlinear_converged)
        self.assertFalse(coarse.quadrature_converged)
        self.assertFalse(coarse.converged)
        self.assertEqual(len(coarse.quadrature_history), 1)
        refined = solve(problem, 4, solver="galerkin", quadrature_order=5,
                        quadrature_atol="1e-9", quadrature_rtol="1e-9", max_refinements=3)
        self.assertTrue(refined.converged)
        self.assertGreater(len(refined.quadrature_history), 1)
        self.assertEqual(refined.plan.order, refined.quadrature_history[-1].order)
        failed = solve(variable_problem(), 4, solver="galerkin", max_iterations=0,
                       use_globalization=False)
        self.assertFalse(failed.nonlinear_converged)
        self.assertFalse(failed.converged)
        self.assertEqual(failed.quadrature_history[-1].reason, "max_iterations")

    def test_solution_restarts_validate_solver_basis_and_geometry(self):
        problem = make_problem()
        for solver, N in (("collocation", 8), ("galerkin", 4)):
            first = solve(problem, N, solver=solver)
            resumed = solve(problem, N, solver=solver, initial_guess=first)
            self.assertTrue(resumed.converged)
            self.assertLess(abs(resumed.u(1)-first.u(1)), 1e-8)
            with tempfile.TemporaryDirectory(dir="/tmp") as directory:
                path = Path(directory)/"restart.json"
                first.save(path)
                file_restart = solve(problem, N, solver=solver, initial_guess=path)
                self.assertTrue(file_restart.converged)
            for options in ({"N": N+1}, {"L": 2}):
                with self.assertRaises(ValueError):
                    solve(problem, options.get("N", N), solver=solver,
                          L=options.get("L", 1), initial_guess=first)
        collocation = solve(problem, 8)
        galerkin = solve(problem, 4, solver="galerkin")
        with self.assertRaisesRegex(ValueError, "projection"):
            solve(problem, 4, solver="galerkin", initial_guess=collocation)
        with self.assertRaises(ValueError):
            solve(problem, 8, initial_guess=galerkin)

    def test_caller_global_precision_is_restored_and_snapshots_remain_owned(self):
        mp.mp.prec = 71
        original = mp.mp.prec
        first = solve(make_problem(), 8, use_mpmath=True, precision=100, tolerance="1e-80")
        self.assertEqual(mp.mp.prec, original)
        value = first.u("0.125")
        solve(make_problem(), 4, solver="galerkin", use_mpmath=True, precision=30)
        self.assertEqual(mp.mp.prec, original)
        self.assertEqual(first.u("0.125"), value)
        first.arithmetic.context.prec += 1
        with self.assertRaises(RuntimeError):
            first.u(1)

    def test_initial_field_callbacks_are_projected_and_verbose_is_explicit(self):
        calls = []
        def initial(x, a):
            calls.append(x)
            return a.scalar(1)/(1+x)**2
        for solver, N in (("collocation", 8), ("galerkin", 4)):
            output = io.StringIO()
            with redirect_stdout(output):
                result = solve(make_problem(), N, solver=solver, initial_guess=initial)
            self.assertTrue(result.converged)
            self.assertEqual(output.getvalue(), "")
        self.assertTrue(calls)

    def test_owned_collocation_evaluation_matches_independent_polynomials(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(120)):
            physical = make_problem().bind(a)
            result = CollocationSolution(physical=physical, coefficients=["0.5", "-0.5", 0], N=1)
            for text in ("0", "0.125", "1", "10", "1e100"):
                x = a.scalar(text)
                relative = abs(result.u(x)*(1+x)-1)
                self.assertLess(relative, a.scalar("1e-13" if not a.use_mpmath else "1e-115"))
                self.assertGreater(abs(result.w(x)), 0)
            self.assertEqual(result.u(a.inf), 0)
            values = np.array([[a.scalar(0), a.scalar(1)]], dtype=object)
            self.assertEqual(result.u(values).shape, values.shape)
            self.assertEqual(result.u(np.empty((0, 2))).shape, (0, 2))
            altered = result.coefficients
            altered[0] += 1
            self.assertEqual(result.u(1), a.scalar("0.5"))
        tiny = CollocationSolution(physical=make_problem().bind(NumpyArithmetic()),
                                  coefficients=["0.5", "-0.5", 0], N=1, L="1e-300")
        self.assertEqual(tiny.u(np.float64("1e-300")), 0.5)

    def test_legacy_import_requires_explicit_geometry_and_is_unverified(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory)/"legacy.pickle"
            with path.open("wb") as handle:
                pickle.dump(np.array([0.5, -0.5, 0.0]), handle)
            result = load_legacy_collocation(path, problem=make_problem(), N=1, L=1)
            self.assertFalse(result.converged)
            self.assertEqual(result.u(1), 0.5)
            with self.assertRaises(ValueError):
                load_legacy_collocation(path, problem=make_problem(), N=2, L=1)

    def test_runnable_common_example_saves_projects_and_restores_two_fields(self):
        from examples.common_solver_example import run
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            with redirect_stdout(io.StringIO()):
                source, restored = run(directory)
            self.assertTrue(restored.converged)
            self.assertEqual(source.solver_type, "collocation")
            self.assertEqual(restored.solver_type, "galerkin")
            self.assertTrue((Path(directory)/"galerkin.json").is_file())
            self.assertLess(abs(restored.u(1)-restored.arithmetic.scalar("0.5")),
                            restored.arithmetic.scalar("1e-35"))


if __name__ == "__main__":
    unittest.main()
