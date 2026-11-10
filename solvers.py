"""Common physical-problem entry point for collocation and mixed Galerkin."""

from dataclasses import dataclass, replace
from contextlib import nullcontext, redirect_stdout
import io
from numbers import Integral
from pathlib import Path

import mpmath as mp

from arithmetic import MpmathArithmetic, NumpyArithmetic
from collocation_problem import CollocationProblemAdapter
from collocation_solution import CollocationSolution
from galerkin import GalerkinBasis
from galerkin_assembly import GalerkinAssembly
from nonlinear_solver import solve_newton
from physical_problem import BoundFluxProblem
from quadrature import QuadratureFactory
from solutions import GalerkinSolution


@dataclass(frozen=True)
class QuadratureIteration:
    order: int
    subdivisions: int
    nonlinear_converged: bool
    quadrature_converged: bool
    residual_difference: object = None
    residual_target: object = None
    jacobian_difference: object = None
    jacobian_target: object = None
    reason: str = ""


def _bind(problem, arithmetic):
    if isinstance(problem, BoundFluxProblem):
        raise TypeError("supply the original unbound physical problem description")
    bind = getattr(problem, "bind", None)
    if not callable(bind):
        raise TypeError("problem must provide bind(arithmetic); use FluxProblem for legacy fixtures")
    physical = bind(arithmetic)
    if not isinstance(physical, BoundFluxProblem) or physical.arithmetic is not arithmetic:
        raise TypeError("problem.bind must return a snapshot in the supplied arithmetic")
    physical.check_precision()
    return physical


def _compatible_restart(source, physical, N, L, solver_type):
    source.check_precision()
    if source.solver_type != solver_type:
        raise ValueError("restart solver type differs; collocation-to-Galerkin requires explicit projection")
    if source.N != N:
        raise ValueError("restart N must match the target basis")
    old, new = source.arithmetic, physical.arithmetic
    old_bits = old.context.prec if old.use_mpmath else 53
    new_bits = new.context.prec if new.use_mpmath else 53
    scalar = (old if old_bits <= new_bits else new).scalar
    if scalar(source.L) != scalar(L):
        raise ValueError("restart mapping length must match the target basis")
    for values, targets in ((source.domain, physical.domain),
                            ((source.physical.delta,), (physical.delta,)),
                            (source.physical.epsilon.interfaces, physical.epsilon.interfaces)):
        if len(values) != len(targets) or any(scalar(a) != scalar(b) for a, b in zip(values, targets)):
            raise ValueError("restart physical geometry or parameters are incompatible")
    for old_row, new_row in zip(source.physical.boundary_conditions, physical.boundary_conditions):
        if any((a is None) != (b is None) or
               (a is not None and scalar(a) != scalar(b)) for a, b in zip(old_row, new_row)):
            raise ValueError("restart boundary data are incompatible")
    for name in ("lower_bound", "upper_bound", "wall_value"):
        if scalar(getattr(source.physical.epsilon, name)) != scalar(getattr(physical.epsilon, name)):
            raise ValueError("restart epsilon metadata are incompatible")
    for point in (new.scalar(L), 2*new.scalar(L)):
        if point not in physical.epsilon.interfaces:
            if scalar(source.physical.epsilon(old.scalar(point))) != scalar(physical.epsilon(point)):
                raise ValueError("restart epsilon samples are incompatible")


def _initial_coefficients(initial, physical, N, L, solver_type):
    if isinstance(initial, (GalerkinSolution, CollocationSolution)):
        _compatible_restart(initial, physical, N, L, solver_type)
        return physical.arithmetic.vector(initial.coefficients)
    return initial


def _project_initial_u(assembly, callback):
    arithmetic = assembly.arithmetic
    table, weights = assembly.table, assembly.plan.physical_weights
    m = assembly.basis.n_u
    values = []
    for x, y in zip(assembly.plan.physical_nodes, assembly.plan.nodes):
        value = arithmetic.scalar(callback(x, arithmetic))
        if not arithmetic.isfinite(value):
            raise ValueError("initial field callback must return finite values")
        values.append(value-assembly.lift.value(y))
    matrix = arithmetic.matrix([
        [assembly._sum(weight*v[i]*v[j] for weight, v in zip(weights, table.v))
         for j in range(m)] for i in range(m)
    ])
    rhs = arithmetic.vector([
        assembly._sum(weight*value*v[i] for weight, value, v in zip(weights, values, table.v))
        for i in range(m)
    ])
    first = arithmetic.solve(matrix, rhs)
    return arithmetic.vector(tuple(first)+(arithmetic.scalar(0),)*assembly.basis.n_w)


def _owned_newton_result(result, arithmetic, coefficients):
    if result is None:
        return None
    scalar = arithmetic.scalar
    return replace(
        result, state=arithmetic.vector(coefficients),
        homotopy=scalar(result.homotopy), residual_norm=scalar(result.residual_norm),
        stages=tuple(replace(stage, homotopy=scalar(stage.homotopy),
                             initial_norm=scalar(stage.initial_norm),
                             target_norm=scalar(stage.target_norm), final_norm=scalar(stage.final_norm))
                     for stage in result.stages),
        history=tuple(replace(record, homotopy=scalar(record.homotopy),
                              residual_norm=scalar(record.residual_norm), step=scalar(record.step))
                      for record in result.history),
    )


def solve(problem, N, *, solver="collocation", L=1, use_mpmath=False, precision=50,
          method="newton", tolerance="1e-10", initial_guess=None,
          max_iterations=50, use_globalization=True, globalization_init="1",
          quadrature_order=None, subdivisions=1, quadrature_atol=None,
          quadrature_rtol=None, max_refinements=3, verbose=False):
    """Solve an unbound physical problem and return an evaluable solution.

    Collocation supports epsilon=1 and its existing method/boundary families.
    Galerkin supports the fourth-order half-line split formulation and Newton.
    Its ``converged`` flag requires both final-q=0 Newton convergence and an
    independent fixed-state residual/Jacobian quadrature comparison.

    A solution object can restart the same solver/basis. Convert collocation
    explicitly with ``projection.project_collocation`` before a mixed restart.
    A raw vector is interpreted in the explicitly selected solver's basis.
    A callable initial guess has signature (x, arithmetic); Galerkin projects
    its u field and starts the independent w field at zero.
    """
    if solver not in ("collocation", "galerkin"):
        raise ValueError("solver must be collocation or galerkin")
    methods = ("newton", "root", "root_no_jac", "nonlin_solve")
    if method not in methods or (solver == "galerkin" and method != "newton"):
        raise ValueError("Galerkin supports newton; collocation supports newton/root/root_no_jac/nonlin_solve")
    if use_mpmath and method != "newton":
        raise ValueError("mpmath supports the newton method only")
    if isinstance(N, bool) or not isinstance(N, Integral) or N < (2 if solver == "galerkin" else 1):
        raise ValueError("N is too small or is not an integer")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 0):
        raise ValueError("max_iterations must be a nonnegative integer")
    arithmetic = MpmathArithmetic(precision) if use_mpmath else NumpyArithmetic()
    physical = _bind(problem, arithmetic)
    tolerance = arithmetic.scalar(tolerance)
    if not arithmetic.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and strictly positive")
    if isinstance(initial_guess, (str, Path)):
        from solution_io import load_solution
        initial_guess = load_solution(initial_guess, problem=problem, solver=solver,
                                      use_mpmath=use_mpmath,
                                      precision=precision if use_mpmath else None)
    initial_guess = _initial_coefficients(initial_guess, physical, N, L, solver)
    if solver == "collocation":
        from spectral1d import solve_nonlinear_problem
        adapter = CollocationProblemAdapter(physical)
        if callable(initial_guess):
            physical = replace(physical, _initial_guess=initial_guess)
            adapter = CollocationProblemAdapter(physical)
            initial_guess = None
        # The legacy implementation uses global mpmath internally. Snapshot
        # its result in owned arithmetic and restore the caller's exact bits.
        previous_bits = mp.mp.prec
        try:
            legacy = solve_nonlinear_problem(
                int(N), problem=adapter, L=arithmetic.scalar(L), use_mpmath=use_mpmath,
                prec=precision, use_method=method, tolerance=tolerance,
                total_iterations=max_iterations, use_globalization=use_globalization,
                globalization_init=arithmetic.scalar(globalization_init),
            )
            native_initial = None
            if initial_guess is not None:
                native_initial = (mp.matrix(tuple(initial_guess)) if use_mpmath else
                                  arithmetic.vector(initial_guess))
                if len(native_initial) != legacy._M:
                    raise ValueError("initial coefficient count does not match the collocation basis")
            with nullcontext() if verbose else redirect_stdout(io.StringIO()):
                coefficients = tuple(arithmetic.scalar(value)
                                     for value in legacy.solve_problem(native_initial))
            result = _owned_newton_result(legacy.get_newton_result(), arithmetic, coefficients)
            return CollocationSolution(
                physical=physical, coefficients=coefficients, N=int(N), L=L, method=method,
                converged=legacy.is_converged(), newton_result=result,
            )
        finally:
            mp.mp.prec = previous_bits
    if (isinstance(max_refinements, bool) or not isinstance(max_refinements, Integral)
            or max_refinements < 0):
        raise ValueError("max_refinements must be a nonnegative integer")
    basis = GalerkinBasis(int(N), arithmetic, L)
    order = 2*int(N) if quadrature_order is None else quadrature_order
    raw_interfaces = getattr(getattr(problem, "epsilon", None), "interfaces", physical.epsilon.interfaces)
    plan = QuadratureFactory(arithmetic, L=L, interfaces=raw_interfaces).plan(order, subdivisions)
    # Bind once: all refinement systems must retain this exact snapshot.
    assembly = GalerkinAssembly._from_snapshot(physical, basis, plan)
    if callable(initial_guess):
        initial_guess = _project_initial_u(assembly, initial_guess)
    elif initial_guess is None and physical._initial_guess is not None:
        initial_guess = _project_initial_u(assembly, physical._initial_guess)
    atol = tolerance if quadrature_atol is None else arithmetic.scalar(quadrature_atol)
    rtol = tolerance if quadrature_rtol is None else arithmetic.scalar(quadrature_rtol)
    if (not arithmetic.isfinite(atol) or not arithmetic.isfinite(rtol) or atol < 0 or rtol < 0):
        raise ValueError("quadrature tolerances must be finite and nonnegative")
    history = []
    checked = False
    for refinement in range(int(max_refinements)+1):
        result = solve_newton(
            assembly.nonlinear_system(initial_guess), atol=tolerance, rtol=0,
            max_iterations=max_iterations, use_globalization=use_globalization,
            globalization_init=globalization_init,
        )
        if not result.converged:
            history.append(QuadratureIteration(assembly.plan.order, assembly.plan.subdivisions,
                                               False, False, reason=result.reason))
            break
        check = assembly.check_refinement(result.state, atol=atol, rtol=rtol)
        checked = check.converged
        history.append(QuadratureIteration(
            assembly.plan.order, assembly.plan.subdivisions, True, checked,
            check.residual_difference, check.residual_target,
            check.jacobian_difference, check.jacobian_target,
            "converged" if checked else "quadrature_not_converged",
        ))
        if checked or refinement == max_refinements:
            break
        initial_guess = result.state.copy()
        assembly = check.refined_assembly
    return GalerkinSolution(
        basis=basis, physical=physical, coefficients=result.state, plan=assembly.plan,
        converged=bool(result.converged and checked), nonlinear_converged=result.converged,
        quadrature_converged=checked, newton_result=result, quadrature_history=tuple(history),
        method="newton",
    )


def load_legacy_collocation(path, *, problem, N, L, use_mpmath=False, precision=50):
    """Import a trusted legacy pickle with explicitly supplied missing geometry.

    This does not solve or certify the imported coefficients. The legacy
    reader rejects tagged mixed data; a mixed restart still needs projection.
    """
    from spectral1d import solve_nonlinear_problem
    arithmetic = MpmathArithmetic(precision) if use_mpmath else NumpyArithmetic()
    physical = _bind(problem, arithmetic)
    previous = mp.mp.prec
    try:
        legacy = solve_nonlinear_problem(N, problem=CollocationProblemAdapter(physical), L=L,
                                        use_mpmath=use_mpmath, prec=precision)
        coefficients = tuple(legacy.load_data(path))
    finally:
        mp.mp.prec = previous
    return CollocationSolution(physical=physical, coefficients=coefficients, N=N, L=L)


def load_solution(path, *, problem, **options):
    from solution_io import load_solution as load
    return load(path, problem=problem, **options)


def project_collocation(source, **options):
    from projection import project_collocation as project
    return project(source, **options)
