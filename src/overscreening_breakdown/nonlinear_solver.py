"""Damped Newton iteration at a fixed parameter, independent of discretization.

The system supplies its arithmetic, residual norm, linear solver and admissible
states. States are real coefficient vectors supporting iteration, ``copy()``,
addition and scalar multiplication (for example NumPy arrays or mpmath columns).
The Jacobian is the derivative of the residual: ``J * step = -residual``.
"""

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from typing import Callable, Optional


class LinearSolveError(ArithmeticError):
    """A numerical linear-solve failure from which recovery may proceed."""


@dataclass(frozen=True)
class NonlinearSystem:
    """Fixed-system callbacks and an optional owner-specific precision guard.

    Discretizations with cached numerical data supply ``check_precision``.
    Arithmetic-independent user systems may leave it as None.
    """

    residual: Callable
    jacobian: Callable
    solve_linear: Callable
    residual_norm: Callable
    initial_guess: Callable
    scalar: Callable
    isfinite: Callable
    restart: Optional[Callable] = None
    merit_gradient: Optional[Callable] = None
    dot: Optional[Callable] = None
    adjoint_direction: Optional[Callable] = None
    check_precision: Optional[Callable] = None


def check_system_precision(system):
    """Check an owned discretization; generic callable systems may omit it."""
    guard = getattr(system, "check_precision", None)
    if guard is not None:
        guard()


def checked_system_call(system, callback, *args, **kwargs):
    """Keep callback-induced precision changes outside numerical recovery."""
    check_system_precision(system)
    try:
        value = callback(*args, **kwargs)
    except Exception:
        # A changed owner must not turn an OverflowError into recoverable
        # numerical failure. Control exceptions still interrupt immediately.
        check_system_precision(system)
        raise
    check_system_precision(system)
    return value


@dataclass(frozen=True)
class NewtonIteration:
    stage: int
    iteration: int
    homotopy: object
    residual_norm: object
    step: object
    backtracks: int
    kind: str = "newton"
    merit: object = None
    gradient_norm: object = None
    recovery_reason: str = ""
    evaluations: int = 0


@dataclass(frozen=True)
class NewtonStage:
    homotopy: object
    initial_norm: object
    target_norm: object
    final_norm: object
    iterations: int
    converged: bool
    reason: str
    adjoint_steps: int = 0
    adjoint_evaluations: int = 0
    recovery_reasons: tuple = ()


@dataclass(frozen=True)
class NewtonResult:
    state: object
    converged: bool
    reason: str
    homotopy: object
    residual_norm: object
    iterations: int
    stages: tuple
    history: tuple
    adjoint_steps: int = 0
    adjoint_evaluations: int = 0
    recovery_reasons: tuple = ()
    reference_result: object = None
    continuation_attempts: int = 0


def solve_newton(
    system,
    *,
    parameter=0,
    stage=1,
    iteration_offset=0,
    atol="1e-8",
    rtol="0",
    max_iterations=50,
    max_backtracks=34,
    min_step="1e-10",
    callback=None,
    newton_fallback=None,
    adjoint_max_steps=20,
    progress=None,
):
    """Solve one nonlinear stage while holding ``parameter`` fixed.

    The measured initial residual sets the target
    ``atol + rtol * initial_norm``. ``max_iterations`` and the result's
    ``iterations`` count attempted Newton steps in this stage. Accepted-state
    records and progress use the supplied ``stage`` and the global iteration
    label ``iteration_offset + local_iteration``.

    Convergence certifies this stage, including a nonzero parameter. A
    continuation controller owns parameter changes and physical-endpoint
    acceptance. This function never restarts or invokes ``system.restart``.
    Only accepted steps reach the observer callback, which receives a copy
    of the current state.

    Linear solvers should translate known numerical failures into
    :class:`LinearSolveError`. Other exceptions propagate to the caller.

    ``newton_fallback='adjoint'`` tries one native merit-descent step after a
    failed linear solve, nonfinite Newton step, or failed Newton line search.
    An accepted recovery retries Newton at the same q. ``adjoint_max_steps``
    bounds accepted recovery steps per stage; ``iterations`` still counts
    Newton attempts only. Stationarity never certifies a nonzero residual.
    A non-Euclidean residual norm requires its matching merit_gradient callback.
    ``progress`` is an internal scalar-event reporter. It is independent of
    the accepted-state callback and never requests additional evaluations.
    """
    check_system_precision(system)
    for name, value, minimum in (
        ("max_iterations", max_iterations, 0),
        ("stage", stage, 1),
        ("iteration_offset", iteration_offset, 0),
        ("max_backtracks", max_backtracks, 1),
        ("adjoint_max_steps", adjoint_max_steps, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if newton_fallback not in (None, "adjoint"):
        raise ValueError("newton_fallback must be None or adjoint")

    scalar = system.scalar
    finite = system.isfinite
    zero, one = scalar(0), scalar(1)
    half = scalar("0.5")
    infinity = scalar("inf")
    atol, rtol = scalar(atol), scalar(rtol)
    min_step = scalar(min_step)
    q = scalar(parameter)
    if not finite(atol) or not finite(rtol) or atol < zero or rtol < zero:
        raise ValueError("atol and rtol must be finite and nonnegative")
    if atol == zero and rtol == zero:
        raise ValueError("at least one residual tolerance must be positive")
    if not finite(min_step) or not zero < min_step <= one:
        raise ValueError("min_step must be finite and in (0, 1]")
    if not finite(q):
        raise ValueError("parameter must be finite")

    def finite_vector(values):
        return all(finite(value) for value in values)

    def matrix_entries(matrix):
        # NumPy matrices iterate over rows; mpmath matrices over scalar entries.
        for row in matrix:
            try:
                entries = iter(row)
            except TypeError:
                yield row
            else:
                yield from entries

    def evaluate(state, parameter):
        if not finite_vector(state):
            return None, infinity
        try:
            with phase("residual", stage=stage,
                       iteration=iteration_offset+total_iterations, homotopy=parameter):
                residual = checked_system_call(system, system.residual, state, parameter)
                norm = checked_system_call(system, system.residual_norm, residual)
        except OverflowError:
            return None, infinity
        if not finite_vector(residual) or not finite(norm):
            return None, infinity
        if norm < zero:
            raise ValueError("residual_norm must be nonnegative")
        return residual, norm

    state = checked_system_call(system, system.initial_guess).copy()
    stages, history = [], []
    total_iterations = 0
    total_adjoint_steps = total_adjoint_evaluations = 0
    recovery_reasons = []

    def emit(kind, *, level=1, **data):
        if progress is not None:
            checked_system_call(system, progress.emit, kind, level=level, **data)

    @contextmanager
    def phase(name, **data):
        check_system_precision(system)
        try:
            with (progress.phase(name, level=3, **data)
                  if progress is not None else nullcontext()):
                check_system_precision(system)
                yield
        except Exception:
            check_system_precision(system)
            raise
        else:
            check_system_precision(system)

    def result(converged, reason, norm):
        emit("newton_end", stage=stage, converged=converged, reason=reason,
             homotopy=q, residual_norm=norm, iterations=total_iterations,
             adjoint_steps=total_adjoint_steps,
             adjoint_evaluations=total_adjoint_evaluations)
        return NewtonResult(
            state.copy(), converged, reason, q, norm, total_iterations,
            tuple(stages), tuple(history),
            total_adjoint_steps, total_adjoint_evaluations, tuple(recovery_reasons),
        )

    emit("stage_start", stage=stage, homotopy=q)
    residual, norm = evaluate(state, q)
    initial_norm = norm
    target = atol + rtol * norm if finite(norm) else atol
    if not finite(target):
        raise ValueError("residual tolerance overflowed in the selected arithmetic")
    emit("stage_residual", level=2, stage=stage,
         homotopy=q, residual_norm=norm, target_norm=target)
    stage_iterations = 0
    stage_adjoint_steps = stage_adjoint_evaluations = 0
    stage_recovery_reasons = []
    converged = residual is not None and norm <= target
    reason = "converged" if converged else "max_iterations"

    if residual is None:
        reason = "nonfinite_residual"
    else:
        while not converged and stage_iterations < max_iterations:
            stage_iterations += 1
            total_iterations += 1
            failure = None
            jacobian = None
            emit("iteration_start", level=2, stage=stage,
                 iteration=iteration_offset+total_iterations, homotopy=q,
                 residual_norm=norm, target_norm=target, step_kind="newton")
            try:
                try:
                    with phase("jacobian", stage=stage,
                               iteration=iteration_offset+total_iterations, homotopy=q):
                        jacobian = checked_system_call(system, system.jacobian, state, q)
                except OverflowError:
                    reason = "nonfinite_jacobian"
                    break
                if not finite_vector(matrix_entries(jacobian)):
                    reason = "nonfinite_jacobian"
                    break
                # Recovery needs the original derivative even when a
                # supplied linear solver factors its input in place.
                linear_matrix = jacobian.copy() if newton_fallback == "adjoint" else jacobian
                with phase("linear_solve", stage=stage,
                           iteration=iteration_offset+total_iterations, homotopy=q):
                    step = checked_system_call(system, system.solve_linear, linear_matrix, -residual)
            except LinearSolveError:
                failure = "linear_solve_failed"
            if failure is None and not finite_vector(step):
                failure = "nonfinite_step"

            weight = one
            accepted = False
            for backtracks in range(max_backtracks) if failure is None else ():
                if weight < min_step:
                    break
                candidate = state + weight * step
                candidate_residual, candidate_norm = evaluate(candidate, q)
                if candidate_residual is not None and (
                    candidate_norm < norm or candidate_norm <= target
                ):
                    state, residual, norm = candidate, candidate_residual, candidate_norm
                    accepted = True
                    record = NewtonIteration(
                        stage, iteration_offset+total_iterations, q, norm, weight, backtracks
                    )
                    history.append(record)
                    emit("iteration_end", level=2, stage=stage,
                         iteration=iteration_offset+total_iterations, homotopy=q,
                         residual_norm=norm, target_norm=target,
                         weight=weight, backtracks=backtracks, step_kind="newton")
                    if callback is not None:
                        checked_system_call(system, callback, state.copy(), record)
                    break
                emit("trial_rejected", level=3, throttle=True, stage=stage,
                     iteration=iteration_offset+total_iterations, homotopy=q,
                     residual_norm=candidate_norm, target_norm=target,
                     weight=weight, backtracks=backtracks, step_kind="newton",
                     reason="nonfinite_residual" if candidate_residual is None
                     else "insufficient_decrease")
                weight *= half

            if not accepted:
                failure = failure or "line_search_failed"
                reason = failure
                if newton_fallback != "adjoint" or jacobian is None:
                    break
                if stage_adjoint_steps >= adjoint_max_steps:
                    reason = "adjoint_budget_exhausted"
                    break
                from overscreening_breakdown.nonlinear_globalization import adjoint_step
                stage_recovery_reasons.append(failure)
                recovery_reasons.append(failure)
                emit("recovery_start", stage=stage,
                     iteration=iteration_offset+total_iterations, homotopy=q,
                     residual_norm=norm, target_norm=target,
                     reason=failure, step_kind="adjoint")

                def evaluate_recovery(candidate, parameter):
                    values, candidate_norm = evaluate(candidate, parameter)
                    emit("trial_evaluation", level=3, throttle=True, stage=stage,
                         iteration=iteration_offset+total_iterations, homotopy=parameter,
                         residual_norm=candidate_norm, target_norm=target,
                         finite=values is not None, step_kind="adjoint")
                    return values, candidate_norm

                try:
                    recovery = checked_system_call(
                        system, adjoint_step, system, state, residual, jacobian, q, residual_norm=norm,
                        evaluate=evaluate_recovery, max_backtracks=max_backtracks,
                        min_step=min_step,
                    )
                except LinearSolveError:
                    reason = "adjoint_direction_failed"
                    emit("recovery_end", stage=stage,
                         iteration=iteration_offset+total_iterations, homotopy=q,
                         accepted=False, reason=reason, step_kind="adjoint")
                    break
                stage_adjoint_evaluations += recovery.evaluations
                total_adjoint_evaluations += recovery.evaluations
                emit("recovery_end", stage=stage,
                     iteration=iteration_offset+total_iterations, homotopy=q,
                     accepted=recovery.accepted, reason=recovery.reason,
                     residual_norm=recovery.residual_norm, target_norm=target,
                     weight=recovery.step, backtracks=recovery.backtracks,
                     evaluations=recovery.evaluations, step_kind="adjoint")
                if not recovery.accepted:
                    reason = recovery.reason
                    break
                state, residual, norm = recovery.state, recovery.residual, recovery.residual_norm
                stage_adjoint_steps += 1
                total_adjoint_steps += 1
                record = NewtonIteration(
                    stage, iteration_offset+total_iterations, q, norm, recovery.step,
                    recovery.backtracks, kind="adjoint", merit=recovery.merit,
                    gradient_norm=recovery.gradient_norm, recovery_reason=failure,
                    evaluations=recovery.evaluations,
                )
                history.append(record)
                emit("iteration_end", level=2, stage=stage,
                     iteration=iteration_offset+total_iterations, homotopy=q,
                     residual_norm=norm, target_norm=target,
                     weight=recovery.step, backtracks=recovery.backtracks,
                     step_kind="adjoint", reason=failure)
                if callback is not None:
                    checked_system_call(system, callback, state.copy(), record)
            converged = norm <= target
            reason = "converged" if converged else "max_iterations"

    stages.append(NewtonStage(
        q, initial_norm, target, norm, stage_iterations, converged, reason,
        stage_adjoint_steps, stage_adjoint_evaluations, tuple(stage_recovery_reasons),
    ))
    emit("stage_end", stage=stage, homotopy=q,
         converged=converged, reason=reason, residual_norm=norm,
         target_norm=target, iterations=stage_iterations,
         adjoint_steps=stage_adjoint_steps,
         adjoint_evaluations=stage_adjoint_evaluations)
    return result(converged, reason, norm)
