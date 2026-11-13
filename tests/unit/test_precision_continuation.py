"""Owned discrete systems and references reject exact-bit precision changes."""

from dataclasses import replace
import unittest

import mpmath as mp

from overscreening_breakdown import FluxProblem, MpmathArithmetic, NumpyArithmetic, solve
from overscreening_breakdown.arithmetic import arithmetic_signature, require_precision
from overscreening_breakdown.collocation.adapter import CollocationProblemAdapter
from overscreening_breakdown.collocation.spectral1d import solve_nonlinear_problem
from overscreening_breakdown.continuation import AffineReference, ResidualHomotopy, solve_continuation
from overscreening_breakdown.diagnostics import ProgressReporter
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.galerkin.finite_assembly import FiniteGalerkinAssembly
from overscreening_breakdown.galerkin.finite_basis import FiniteGalerkinBasis
from overscreening_breakdown.nonlinear_solver import NonlinearSystem, solve_newton
from overscreening_breakdown.quadrature import QuadratureFactory


class ContinuationPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, mp.mp, "prec", mp.mp.prec)

    def problem(self, finite=False):
        return FluxProblem(
            delta="2", f=lambda x, u, a: u, f_u=lambda x, u, a: a.scalar(1),
            domain=(0, 2 if finite else "inf"),
            boundary_conditions=((1, None, None, 0, None),
                                 (0, 0, None, None, None) if finite else (None,)*5),
        )

    def galerkin(self, finite=False, arithmetic=None):
        a = MpmathArithmetic(50) if arithmetic is None else arithmetic
        problem = self.problem(finite)
        basis = (FiniteGalerkinBasis(4, a, problem.domain) if finite else GalerkinBasis(4, a))
        plan = QuadratureFactory(a, domain=problem.domain).plan(8)
        assembly = (FiniteGalerkinAssembly if finite else GalerkinAssembly)(problem, basis, plan)
        return a, assembly

    def linear_system(self, a, guard=True):
        signature = arithmetic_signature(a)

        def check():
            require_precision(a, signature)

        return NonlinearSystem(
            residual=lambda z, q=0: a.vector([z[0]-1]),
            jacobian=lambda z, q=0: a.matrix([[1]]),
            solve_linear=a.solve, residual_norm=a.norm,
            initial_guess=lambda: a.vector([0]), scalar=a.scalar, isfinite=a.isfinite,
            check_precision=check if guard else None,
        )

    def test_galerkin_reference_and_all_homotopy_endpoints_check_exact_bits(self):
        for finite in (False, True):
            for shift in (-1, 1):
                with self.subTest(finite=finite, shift=shift):
                    a, assembly = self.galerkin(finite)
                    system = assembly.nonlinear_system()
                    z = system.initial_guess()
                    matrix, offset = assembly.reference_operator()
                    reference = AffineReference(matrix, offset, system)
                    homotopy = ResidualHomotopy(system, reference)
                    blended = homotopy.nonlinear_system(z)
                    old_bits, old_digits = a.context.prec, a.dps
                    a.context.prec += shift
                    if shift == 1:
                        self.assertEqual(a.dps, old_digits)
                    callbacks = [
                        system.check_precision, system.initial_guess, blended.initial_guess,
                        reference.check_precision, lambda: reference.matrix, lambda: reference.offset,
                        lambda: reference.residual(z), reference.solve,
                        lambda: AffineReference(matrix, offset, system),
                        lambda: homotopy.nonlinear_system(z),
                        lambda: solve_newton(system),
                        lambda: solve_continuation(system, reference_factory=assembly.reference_operator),
                    ]
                    for q in (0, "0.5", 1):
                        callbacks.extend((lambda q=q: homotopy.residual(z, q),
                                          lambda q=q: homotopy.jacobian(z, q)))
                    for callback in callbacks:
                        with self.subTest(callback=repr(callback)):
                            with self.assertRaisesRegex(RuntimeError, "precision changed"):
                                callback()
                    a.context.prec = old_bits
                    self.assertEqual(tuple(system.initial_guess()), tuple(z))
                    self.assertEqual(tuple(reference.residual(z)), tuple(homotopy.residual(z, 1)))

    def test_collocation_checks_global_and_bound_physical_owners(self):
        for finite in (False, True):
            for mutate_physical in (False, True):
                with self.subTest(finite=finite, mutate_physical=mutate_physical):
                    a = MpmathArithmetic(50)
                    adapter = CollocationProblemAdapter(self.problem(finite).bind(a))
                    kernel = solve_nonlinear_problem(4, problem=adapter, use_mpmath=True,
                                                     prec=50, use_globalization=False)
                    system = kernel.nonlinear_system()
                    z = system.initial_guess()
                    matrix, offset = kernel.reference_operator()
                    reference = AffineReference(matrix, offset, system)
                    owner = a.context if mutate_physical else mp.mp
                    bits, digits = owner.prec, owner.dps
                    owner.prec += 1
                    self.assertEqual(owner.dps, digits)
                    for callback in (kernel.check_precision, kernel.reference_operator,
                                     lambda: kernel.nonlinear_system(z), kernel.solve_problem,
                                     system.initial_guess, lambda: system.residual(z),
                                     lambda: system.jacobian(z), lambda: reference.residual(z),
                                     lambda: reference.matrix):
                        with self.subTest(callback=repr(callback)):
                            with self.assertRaisesRegex(RuntimeError, "precision changed"):
                                callback()
                    owner.prec = bits
                    system.check_precision()

    def test_newton_observer_mutation_stops_before_returning_convergence(self):
        a = MpmathArithmetic(50)
        system = self.linear_system(a)
        observed = []

        def callback(state, iteration):
            observed.append(iteration)
            a.context.prec += 1

        with self.assertRaisesRegex(RuntimeError, "precision changed"):
            solve_newton(system, atol="1e-40", callback=callback)
        self.assertEqual(len(observed), 1)

    def test_newton_progress_phase_mutation_cannot_bypass_guard(self):
        a = MpmathArithmetic(50)
        system = self.linear_system(a)

        def progress(event):
            if event.kind == "phase_end":
                a.context.prec += 1

        reporter = ProgressReporter(arithmetic=a, progress_callback=progress)
        with self.assertRaisesRegex(RuntimeError, "precision changed"):
            solve_newton(system, atol="1e-40", progress=reporter)

    def test_control_exceptions_still_interrupt_after_callback_precision_changes(self):
        for exception in (KeyboardInterrupt, SystemExit):
            for progress_callback in (False, True):
                a = MpmathArithmetic(50)
                system = self.linear_system(a)

                def interrupt(*args):
                    a.context.prec += 1
                    raise exception

                with self.subTest(exception=exception.__name__, progress=progress_callback):
                    with self.assertRaises(exception):
                        if progress_callback:
                            reporter = ProgressReporter(arithmetic=a, progress_callback=interrupt)
                            solve_newton(system, atol="1e-40", progress=reporter)
                        else:
                            solve_newton(system, atol="1e-40", callback=interrupt)

    def test_reference_factory_mutation_is_fatal_instead_of_recoverable(self):
        a = MpmathArithmetic(50)
        system = self.linear_system(a)
        # An admissible finite physical residual misses a zero-iteration budget.
        def factory(strength):
            matrix, offset = a.matrix([[strength]]), a.vector([-strength])
            a.context.prec += 1
            return matrix, offset

        with self.assertRaisesRegex(RuntimeError, "precision changed"):
            solve_continuation(system, max_iterations=0, reference_factory=factory)

    def test_callback_mutation_raising_overflow_is_not_numerical_recovery(self):
        a = MpmathArithmetic(50)
        system = self.linear_system(a)

        def broken(state, parameter):
            a.context.prec += 1
            raise OverflowError("trial failure after changing precision")

        with self.assertRaisesRegex(RuntimeError, "precision changed"):
            solve_newton(replace(system, residual=broken))

    def test_generic_systems_without_precision_hook_remain_supported(self):
        for a in (NumpyArithmetic(), MpmathArithmetic(50)):
            with self.subTest(backend=a.backend):
                system = self.linear_system(a, guard=False)
                self.assertIsNone(system.check_precision)
                reference = AffineReference(a.matrix([[1]]), a.vector([-1]), system)
                self.assertEqual(reference.solve()[0], 1)
                self.assertTrue(solve_newton(system, atol="1e-12").converged)

    def test_owned_public_collocation_result_ignores_later_global_precision_changes(self):
        result = solve(self.problem(), 6, solver="collocation", use_mpmath=True,
                       precision=30, tolerance="1e-20")
        self.assertTrue(result.converged)
        value = result.u(1)
        bits = result.arithmetic.context.prec
        mp.mp.prec += 1
        self.assertEqual(result.arithmetic.context.prec, bits)
        self.assertEqual(result.u(1), value)


if __name__ == "__main__":
    unittest.main()
