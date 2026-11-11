"""Damped Newton iteration with continuation, independent of discretization.

The system supplies its arithmetic, residual norm, linear solver and admissible
states. States are real coefficient vectors supporting iteration, ``copy()``,
addition and scalar multiplication (for example NumPy arrays or mpmath columns).
The Jacobian is the derivative of the residual: ``J * step = -residual``.
"""

from dataclasses import dataclass
from typing import Callable, Optional


class LinearSolveError(ArithmeticError):
    """A numerical linear-solve failure from which continuation may recover."""


@dataclass(frozen=True)
class NonlinearSystem:
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


def solve_newton(
    system,
    *,
    atol="1e-8",
    rtol="0",
    max_iterations=50,
    use_globalization=True,
    globalization_init="1",
    max_stages=100,
    max_backtracks=34,
    min_step="1e-10",
    homotopy_cutoff="1e-11",
    callback=None,
    newton_fallback=None,
    adjoint_max_steps=20,
):
    """Solve the physical system at homotopy parameter zero.

    Each stage uses the measured initial residual to set the target
    ``atol + rtol * initial_norm``. ``max_iterations`` applies per stage;
    ``iterations`` in the result counts all attempted Newton steps.

    Try the physical problem first. On failure, restart at a positive homotopy
    parameter, increasing it by 1.25 after failed stages and halving it after
    successful ones. A small positive parameter is replaced by an explicit
    final zero-parameter stage, never by an unverified convergence claim.

    ``restart(state, new_parameter)`` must return an admissible state. Without
    that callback, ``initial_guess()`` supplies the restart. Only accepted steps
    reach the observer callback, which receives a copy of the current state.

    Linear solvers should translate known numerical failures into
    :class:`LinearSolveError`. Other exceptions propagate to the caller.

    ``newton_fallback='adjoint'`` tries one native merit-descent step after a
    failed linear solve, nonfinite Newton step, or failed Newton line search.
    An accepted recovery retries Newton at the same q. ``adjoint_max_steps``
    bounds accepted recovery steps per stage; ``iterations`` still counts
    Newton attempts only. Stationarity never certifies a nonzero residual.
    A non-Euclidean residual norm requires its matching merit_gradient callback.
    """
    for name, value, minimum in (
        ("max_iterations", max_iterations, 0),
        ("max_stages", max_stages, 1),
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
    half, growth = scalar("0.5"), scalar("1.25")
    infinity = scalar("inf")
    atol, rtol = scalar(atol), scalar(rtol)
    min_step, cutoff = scalar(min_step), scalar(homotopy_cutoff)
    initial_q = scalar(globalization_init)
    if not finite(atol) or not finite(rtol) or atol < zero or rtol < zero:
        raise ValueError("atol and rtol must be finite and nonnegative")
    if atol == zero and rtol == zero:
        raise ValueError("at least one residual tolerance must be positive")
    if not finite(min_step) or not zero < min_step <= one:
        raise ValueError("min_step must be finite and in (0, 1]")
    if not finite(cutoff) or cutoff <= zero:
        raise ValueError("homotopy_cutoff must be finite and positive")
    if use_globalization and (not finite(initial_q) or initial_q <= zero):
        raise ValueError("globalization_init must be finite and positive")

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
            residual = system.residual(state, parameter)
            norm = system.residual_norm(residual)
        except OverflowError:
            return None, infinity
        if not finite_vector(residual) or not finite(norm):
            return None, infinity
        if norm < zero:
            raise ValueError("residual_norm must be nonnegative")
        return residual, norm

    state = system.initial_guess().copy()
    q = zero
    stages, history = [], []
    total_iterations = 0
    total_adjoint_steps = total_adjoint_evaluations = 0
    recovery_reasons = []

    def result(converged, reason, norm):
        return NewtonResult(
            state.copy(), converged, reason, q, norm, total_iterations,
            tuple(stages), tuple(history),
            total_adjoint_steps, total_adjoint_evaluations, tuple(recovery_reasons),
        )

    for stage_index in range(max_stages):
        residual, norm = evaluate(state, q)
        initial_norm = norm
        target = atol + rtol * norm if finite(norm) else atol
        if not finite(target):
            raise ValueError("residual tolerance overflowed in the selected arithmetic")
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
                try:
                    try:
                        jacobian = system.jacobian(state, q)
                    except OverflowError:
                        reason = "nonfinite_jacobian"
                        break
                    if not finite_vector(matrix_entries(jacobian)):
                        reason = "nonfinite_jacobian"
                        break
                    # Recovery needs the original derivative even when a
                    # supplied linear solver factors its input in place.
                    linear_matrix = jacobian.copy() if newton_fallback == "adjoint" else jacobian
                    step = system.solve_linear(linear_matrix, -residual)
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
                            stage_index + 1, total_iterations, q, norm, weight, backtracks
                        )
                        history.append(record)
                        if callback is not None:
                            callback(state.copy(), record)
                        break
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
                    try:
                        recovery = adjoint_step(
                            system, state, residual, jacobian, q, residual_norm=norm,
                            evaluate=evaluate, max_backtracks=max_backtracks, min_step=min_step,
                        )
                    except LinearSolveError:
                        reason = "adjoint_direction_failed"
                        break
                    stage_adjoint_evaluations += recovery.evaluations
                    total_adjoint_evaluations += recovery.evaluations
                    if not recovery.accepted:
                        reason = recovery.reason
                        break
                    state, residual, norm = recovery.state, recovery.residual, recovery.residual_norm
                    stage_adjoint_steps += 1
                    total_adjoint_steps += 1
                    record = NewtonIteration(
                        stage_index+1, total_iterations, q, norm, recovery.step,
                        recovery.backtracks, kind="adjoint", merit=recovery.merit,
                        gradient_norm=recovery.gradient_norm, recovery_reason=failure,
                        evaluations=recovery.evaluations,
                    )
                    history.append(record)
                    if callback is not None:
                        callback(state.copy(), record)
                converged = norm <= target
                reason = "converged" if converged else "max_iterations"

        stages.append(NewtonStage(
            q, initial_norm, target, norm, stage_iterations, converged, reason,
            stage_adjoint_steps, stage_adjoint_evaluations, tuple(stage_recovery_reasons),
        ))
        if converged and q == zero:
            return result(True, "converged", norm)
        if not use_globalization:
            return result(False, reason, norm)
        if stage_index + 1 == max_stages:
            return result(False, "max_stages", norm)

        if converged:
            next_q = zero if q < cutoff else q * half
        else:
            next_q = (initial_q if q == zero else q) * growth
        if not finite(next_q):
            return result(False, "nonfinite_homotopy", norm)
        if not converged:
            if system.restart is None:
                state = system.initial_guess().copy()
            else:
                state = system.restart(state.copy(), next_q).copy()
        q = next_q

    raise AssertionError("unreachable continuation state")
