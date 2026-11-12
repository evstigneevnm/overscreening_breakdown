"""Common physical-problem entry point for collocation and mixed Galerkin."""

from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path

import mpmath as mp

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.collocation.adapter import CollocationProblemAdapter
from overscreening_breakdown.collocation.solution import CollocationSolution
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.continuation import ContinuationOptions, solve_continuation
from overscreening_breakdown.problems.flux import BoundFluxProblem
from overscreening_breakdown.quadrature import QuadratureFactory
from overscreening_breakdown.galerkin.solution import GalerkinSolution
from overscreening_breakdown.galerkin.finite_basis import FiniteGalerkinBasis
from overscreening_breakdown.galerkin.finite_assembly import FiniteGalerkinAssembly
from overscreening_breakdown.galerkin.finite_solution import FiniteGalerkinSolution
from overscreening_breakdown.initialization import validate_initializer
from overscreening_breakdown.diagnostics import ProgressReporter


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
    newton_result: object = None


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
    for values, targets in ((source.domain, physical.domain),
                            ((source.physical.delta,), (physical.delta,)),
                            (source.physical.epsilon.interfaces, physical.epsilon.interfaces)):
        if len(values) != len(targets) or any(scalar(a) != scalar(b) for a, b in zip(values, targets)):
            raise ValueError("restart physical geometry or parameters are incompatible")
    if physical.domain[1] == new.inf and scalar(source.L) != scalar(L):
        raise ValueError("restart mapping length must match the target basis")
    for old_row, new_row in zip(source.physical.boundary_conditions, physical.boundary_conditions):
        if any((a is None) != (b is None) or
               (a is not None and scalar(a) != scalar(b)) for a, b in zip(old_row, new_row)):
            raise ValueError("restart boundary data are incompatible")
    for name in ("lower_bound", "upper_bound", "wall_value"):
        if scalar(getattr(source.physical.epsilon, name)) != scalar(getattr(physical.epsilon, name)):
            raise ValueError("restart epsilon metadata are incompatible")
    if physical.domain[1] == new.inf:
        points = (new.scalar(L), 2*new.scalar(L))
    else:
        boundaries = (physical.domain[0],)+physical.epsilon.interfaces+(physical.domain[1],)
        points = tuple((left+right)/2 for left, right in zip(boundaries[:-1], boundaries[1:]))
    for point in points:
        if point not in physical.epsilon.interfaces:
            if scalar(source.physical.epsilon(old.scalar(point))) != scalar(physical.epsilon(point)):
                raise ValueError("restart epsilon samples are incompatible")


def _initial_coefficients(initial, physical, N, L, solver_type):
    if isinstance(initial, (GalerkinSolution, FiniteGalerkinSolution, CollocationSolution)):
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
                              residual_norm=scalar(record.residual_norm), step=scalar(record.step),
                              merit=None if record.merit is None else scalar(record.merit),
                              gradient_norm=(None if record.gradient_norm is None
                                             else scalar(record.gradient_norm)))
                      for record in result.history),
        reference_result=(None if result.reference_result is None else replace(
            result.reference_result,
            state=arithmetic.vector(result.reference_result.state),
            residual_norm=scalar(result.reference_result.residual_norm),
            target_norm=scalar(result.reference_result.target_norm),
            strength=scalar(result.reference_result.strength),
        )),
    )


def solve(problem, N, *, solver="collocation", L=1, use_mpmath=False, precision=50,
          method="newton", tolerance="1e-10", initial_guess=None,
          max_iterations=50, use_globalization=True, continuation=None,
          quadrature_order=None, subdivisions=1, quadrature_atol=None,
          quadrature_rtol=None, max_refinements=3, verbosity=0,
          initializer=None, initializer_tolerance="0.1", initializer_max_evaluations=200,
          newton_fallback=None, adjoint_max_steps=20,
          progress_callback=None, progress_interval=1.0):
    """Solve an unbound physical problem and return an evaluable solution.

    Collocation supports epsilon=1 and its existing method/boundary families.
    Galerkin supports fourth-order split formulations on [0,infinity) and
    [0,R], using Newton. Finite domains prescribe left u,u''' and right u,u'.
    Their geometry comes from problem.domain; L is a half-line parameter.
    Its ``converged`` flag requires both final-q=0 Newton convergence and an
    independent fixed-state residual/Jacobian quadrature comparison.
    SciPy collocation methods expose ``scipy_result`` stop diagnostics and
    require a freshly verified full q=0 residual, including boundary rows;
    SciPy's success flag alone does not certify convergence.

    A solution object can restart the same solver/basis. Convert collocation
    explicitly with ``projection.project_collocation`` before a mixed restart.
    A raw vector is interpreted in the explicitly selected solver's basis.
    A callable initial guess has signature (x, arithmetic); Galerkin projects
    its u field and starts the independent w field at zero.

    ``initializer='fsolve'`` is an optional bounded NumPy collocation prepass;
    its evaluation budget includes residual and Jacobian calls. Accepted
    candidates still require the main solver's convergence check.
    ``newton_fallback='adjoint'`` enables native merit descent after a failed
    Newton step, with at most ``adjoint_max_steps`` accepted recoveries per stage.
    ``use_globalization`` controls homotopy; Newton backtracking remains active.
    Optional ``ContinuationOptions`` configure the verified screened-linear
    reference and adaptive stepping. ``max_iterations`` applies per Newton
    stage; continuation options bound total Newton work and rejected stages,
    shared across Galerkin quadrature refinements. Physical callbacks remain
    independent of continuation and discretization normalization.

    ``verbosity`` selects quiet (0), phase summaries (1), iteration/evaluation
    progress (2), or detailed trials (3). ``progress_callback(event)`` receives
    immutable native-precision observations independently of text verbosity.
    Frequent evaluation/trial text uses ``progress_interval`` seconds; phase
    transitions and final outcomes are unthrottled. Reporting does not change
    numerical tolerances or introduce additional numerical evaluations.
    """
    progress = ProgressReporter(
        verbosity=verbosity, progress_callback=progress_callback,
        solver=solver, method=method, progress_interval=progress_interval,
    )
    try:
        result = _solve(
            problem, N, solver=solver, L=L, use_mpmath=use_mpmath, precision=precision,
            method=method, tolerance=tolerance, initial_guess=initial_guess,
            max_iterations=max_iterations, use_globalization=use_globalization,
            continuation=continuation, quadrature_order=quadrature_order,
            subdivisions=subdivisions, quadrature_atol=quadrature_atol,
            quadrature_rtol=quadrature_rtol, max_refinements=max_refinements,
            initializer=initializer, initializer_tolerance=initializer_tolerance,
            initializer_max_evaluations=initializer_max_evaluations,
            newton_fallback=newton_fallback, adjoint_max_steps=adjoint_max_steps,
            progress=progress,
        )
        diagnostic = result.newton_result or result.scipy_result
        info = {}
        if diagnostic is not None:
            info.update(reason=diagnostic.reason, residual_norm=diagnostic.residual_norm,
                        homotopy=diagnostic.homotopy)
            if result.scipy_result is not None:
                info.update(target_norm=diagnostic.target_norm,
                            scipy_success=diagnostic.scipy_success, status=diagnostic.status)
            elif diagnostic.stages:
                info["target_norm"] = diagnostic.stages[-1].target_norm
        if result.nonlinear_converged and result.quadrature_converged is False:
            info["reason"] = "quadrature_not_converged"
        progress.emit(
            "solve_end", converged=result.converged,
            nonlinear_converged=result.nonlinear_converged,
            quadrature_converged=result.quadrature_converged,
            norm="rms" if solver == "collocation" else "euclidean", **info,
        )
        return result
    except BaseException as error:
        progress.emit(
            "solve_interrupted" if isinstance(error, KeyboardInterrupt) else "solve_error",
            error_type=type(error).__name__, reason=str(error),
        )
        raise


def _solve(problem, N, *, solver, L, use_mpmath, precision, method, tolerance,
           initial_guess, max_iterations, use_globalization, continuation,
           quadrature_order, subdivisions, quadrature_atol, quadrature_rtol,
           max_refinements, initializer, initializer_tolerance,
           initializer_max_evaluations, newton_fallback, adjoint_max_steps, progress):
    if solver not in ("collocation", "galerkin"):
        raise ValueError("solver must be collocation or galerkin")
    methods = ("newton", "root", "root_no_jac", "nonlin_solve")
    if method not in methods or (solver == "galerkin" and method != "newton"):
        raise ValueError("Galerkin supports newton; collocation supports newton/root/root_no_jac/nonlin_solve")
    if use_mpmath and method != "newton":
        raise ValueError("mpmath supports the newton method only")
    validate_initializer(initializer, use_mpmath=use_mpmath, solver=solver,
                         tolerance=initializer_tolerance,
                         max_evaluations=initializer_max_evaluations)
    if newton_fallback not in (None, "adjoint"):
        raise ValueError("newton_fallback must be None or 'adjoint'")
    if newton_fallback is not None and method != "newton":
        raise ValueError("adjoint recovery requires the newton method")
    if (isinstance(adjoint_max_steps, bool) or not isinstance(adjoint_max_steps, Integral)
            or adjoint_max_steps < 0):
        raise ValueError("adjoint_max_steps must be a nonnegative integer")
    if isinstance(N, bool) or not isinstance(N, Integral) or N < (2 if solver == "galerkin" else 1):
        raise ValueError("N is too small or is not an integer")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 0):
        raise ValueError("max_iterations must be a nonnegative integer")
    arithmetic = MpmathArithmetic(precision) if use_mpmath else NumpyArithmetic()
    if continuation is None:
        continuation = ContinuationOptions()
    if not isinstance(continuation, ContinuationOptions):
        raise TypeError("continuation must be ContinuationOptions or None")
    continuation = continuation.bind(arithmetic.scalar, arithmetic.isfinite)
    # Set the formatter's owner once, before the first numerical event.
    progress.arithmetic = arithmetic
    tolerance = arithmetic.scalar(tolerance)
    if not arithmetic.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and strictly positive")
    progress.emit("solve_start", N=int(N), precision=arithmetic.dps,
                  backend=arithmetic.backend, arithmetic="mpmath" if use_mpmath else "numpy",
                  tolerance=tolerance, use_globalization=use_globalization,
                  initializer=initializer, newton_fallback=newton_fallback)
    with progress.phase("problem_binding"):
        physical = _bind(problem, arithmetic)
    finite = physical.domain[1] != arithmetic.inf
    raw_domain = getattr(problem, "domain", physical.domain)
    if solver == "galerkin":
        with progress.phase("basis"):
            basis = (FiniteGalerkinBasis(int(N), arithmetic, domain=raw_domain) if finite else
                     GalerkinBasis(int(N), arithmetic, L))
        unknowns = basis.unknown_count
    else:
        unknowns = int(N)+sum(value is not None for row in physical.boundary_conditions for value in row)
    progress.emit("problem_ready", unknown_count=unknowns,
                  norm="rms" if solver == "collocation" else "euclidean")
    if isinstance(initial_guess, (str, Path)):
        from overscreening_breakdown.solution_io import load_solution
        with progress.phase("restart_loading"):
            initial_guess = load_solution(initial_guess, problem=problem, solver=solver,
                                          use_mpmath=use_mpmath,
                                          precision=precision if use_mpmath else None)
    initial_guess = _initial_coefficients(initial_guess, physical, N, L, solver)
    if solver == "collocation":
        from overscreening_breakdown.collocation.spectral1d import solve_nonlinear_problem
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
                continuation=continuation,
                initializer=initializer, initializer_tolerance=initializer_tolerance,
                initializer_max_evaluations=int(initializer_max_evaluations),
                newton_fallback=newton_fallback, adjoint_max_steps=int(adjoint_max_steps),
                _reporter=progress.bind(norm="rms"),
            )
            native_initial = None
            if initial_guess is not None:
                native_initial = (mp.matrix(tuple(initial_guess)) if use_mpmath else
                                  arithmetic.vector(initial_guess))
                if len(native_initial) != legacy._M:
                    raise ValueError("initial coefficient count does not match the collocation basis")
            coefficients = tuple(arithmetic.scalar(value)
                                 for value in legacy.solve_problem(native_initial))
            result = _owned_newton_result(legacy.get_newton_result(), arithmetic, coefficients)
            initialization = legacy.get_initialization_result()
            if initialization is not None:
                initialization = replace(
                    initialization, state=arithmetic.vector(initialization.state),
                    initial_norm=arithmetic.scalar(initialization.initial_norm),
                    final_norm=arithmetic.scalar(initialization.final_norm),
                )
            return CollocationSolution(
                physical=physical, coefficients=coefficients, N=int(N), L=L, method=method,
                converged=legacy.is_converged(), newton_result=result,
                initialization_result=initialization,
                scipy_result=legacy.get_scipy_result(),
            )
        finally:
            mp.mp.prec = previous_bits
    if (isinstance(max_refinements, bool) or not isinstance(max_refinements, Integral)
            or max_refinements < 0):
        raise ValueError("max_refinements must be a nonnegative integer")
    progress = progress.bind(norm="euclidean")
    order = 2*int(N) if quadrature_order is None else quadrature_order
    raw_interfaces = getattr(getattr(problem, "epsilon", None), "interfaces", physical.epsilon.interfaces)
    with progress.phase("quadrature", order=order, subdivisions=subdivisions):
        plan = QuadratureFactory(arithmetic, L=L, interfaces=raw_interfaces,
                                 domain=raw_domain if finite else None).plan(order, subdivisions)
    # Bind once: all refinement systems must retain this exact snapshot.
    assembly_type = FiniteGalerkinAssembly if finite else GalerkinAssembly
    assembly = assembly_type._from_snapshot(physical, basis, plan, progress=progress)
    if callable(initial_guess):
        with progress.phase("initial_projection"):
            initial_guess = _project_initial_u(assembly, initial_guess)
    elif initial_guess is None and physical._initial_guess is not None:
        with progress.phase("initial_projection"):
            initial_guess = _project_initial_u(assembly, physical._initial_guess)
    atol = tolerance if quadrature_atol is None else arithmetic.scalar(quadrature_atol)
    rtol = tolerance if quadrature_rtol is None else arithmetic.scalar(quadrature_rtol)
    if (not arithmetic.isfinite(atol) or not arithmetic.isfinite(rtol) or atol < 0 or rtol < 0):
        raise ValueError("quadrature tolerances must be finite and nonnegative")
    history = []
    checked = False
    used_iterations = used_stages = 0
    for refinement in range(int(max_refinements)+1):
        remaining = replace(
            continuation,
            max_total_iterations=continuation.max_total_iterations-used_iterations,
            max_stages=continuation.max_stages-used_stages,
        )
        result = solve_continuation(
            assembly.nonlinear_system(initial_guess),
            reference_factory=assembly.reference_operator,
            options=remaining, atol=tolerance, rtol=0,
            max_iterations=max_iterations, use_globalization=use_globalization,
            newton_fallback=newton_fallback, adjoint_max_steps=int(adjoint_max_steps),
            progress=progress.bind(refinement=refinement, order=assembly.plan.order,
                                   subdivisions=assembly.plan.subdivisions),
        )
        used_iterations += result.iterations
        used_stages += result.continuation_attempts
        if not result.converged:
            history.append(QuadratureIteration(assembly.plan.order, assembly.plan.subdivisions,
                                               False, False, reason=result.reason,
                                               newton_result=result))
            break
        with progress.phase("quadrature_verification", refinement=refinement) as check_progress:
            check = assembly.check_refinement(result.state, atol=atol, rtol=rtol,
                                             progress=check_progress)
        checked = check.converged
        progress.emit(
            "quadrature_check", refinement=refinement, order=assembly.plan.order,
            subdivisions=assembly.plan.subdivisions, converged=checked,
            residual_difference=check.residual_difference, residual_target=check.residual_target,
            jacobian_difference=check.jacobian_difference, jacobian_target=check.jacobian_target,
            jacobian_norm="frobenius",
        )
        history.append(QuadratureIteration(
            assembly.plan.order, assembly.plan.subdivisions, True, checked,
            check.residual_difference, check.residual_target,
            check.jacobian_difference, check.jacobian_target,
            "converged" if checked else "quadrature_not_converged",
            result,
        ))
        if checked or refinement == max_refinements:
            break
        initial_guess = result.state.copy()
        assembly = check.refined_assembly
    solution_type = FiniteGalerkinSolution if finite else GalerkinSolution
    return solution_type(
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
    from overscreening_breakdown.collocation.spectral1d import solve_nonlinear_problem
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
    from overscreening_breakdown.solution_io import load_solution as load
    return load(path, problem=problem, **options)


def project_collocation(source, **options):
    from overscreening_breakdown.projection import project_collocation as project
    return project(source, **options)
