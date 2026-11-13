"""Bounded, optional initial-guess refinement before a nonlinear solve."""

from dataclasses import dataclass, field
from contextlib import nullcontext
from numbers import Integral

import numpy as np

from overscreening_breakdown.nonlinear_solver import check_system_precision, checked_system_call


@dataclass(frozen=True)
class InitializationResult:
    """An initializer outcome, never a nonlinear convergence certificate.

    ``evaluations`` counts residual and Jacobian calls, including validation.
    ``state`` is the best verified finite state visited within that budget.
    SciPy termination and acceptance are deliberately reported separately.
    """

    state: object = field(repr=False)
    accepted: bool
    reason: str
    initial_norm: object
    final_norm: object
    evaluations: int
    function_evaluations: int
    jacobian_evaluations: int
    scipy_status: object = None
    scipy_success: bool = False
    message: str = ""
    method: str = "fsolve"


def validate_initializer(initializer, *, use_mpmath=False, solver="collocation",
                         tolerance="0.1", max_evaluations=200):
    """Validate before discretization or conversion to a SciPy array."""
    if initializer not in (None, "fsolve"):
        raise ValueError("initializer must be None or 'fsolve'")
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral)
            or max_evaluations < 1):
        raise ValueError("initializer_max_evaluations must be a positive integer")
    if initializer is not None:
        if use_mpmath or solver != "collocation":
            raise ValueError("the fsolve initializer supports NumPy collocation only")
        if isinstance(tolerance, bool):
            raise ValueError("initializer_tolerance must be finite and positive")
        value = np.float64(tolerance)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("initializer_tolerance must be finite and positive")


class _EvaluationLimit(Exception):
    pass


def initialize_fsolve(system, initial, *, tolerance="0.1", max_evaluations=200,
                      progress=None):
    """Refine a NumPy guess at q=0, retaining only verified improvements.

    A hard combined residual/Jacobian budget bounds the prepass. If that
    budget or a known numerical failure stops SciPy, an already verified
    improving candidate may still be used. Callback/programming errors
    propagate. The main nonlinear solver must check the returned state.
    """
    check_system_precision(system)
    validate_initializer("fsolve", tolerance=tolerance, max_evaluations=max_evaluations)
    if (not isinstance(initial, np.ndarray) or initial.ndim != 1
            or initial.dtype != np.dtype("float64")):
        raise TypeError("fsolve initialization requires a NumPy float64 coefficient vector")
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial coefficients must be finite")

    from scipy import optimize

    zero = system.scalar(0)
    original = initial.copy()
    best = original.copy()
    best_norm = system.scalar("inf")
    initial_norm = best_norm
    counts = [0, 0]

    def emit(kind, *, level=1, throttle=False, **data):
        if progress is not None:
            checked_system_call(system, progress.emit, kind, level=level, throttle=throttle, **data)

    emit("initializer_start", initializer="fsolve", budget=int(max_evaluations),
         step_tolerance=np.float64(tolerance), homotopy=zero)

    def consume(index):
        if sum(counts) >= max_evaluations:
            raise _EvaluationLimit
        counts[index] += 1

    def state(values):
        if (not isinstance(values, np.ndarray) or values.shape != original.shape
                or values.dtype != original.dtype):
            raise ValueError("initializer coefficient shape or dtype changed")
        if not np.all(np.isfinite(values)):
            raise OverflowError("nonfinite initializer trial coefficients")
        return values

    def evaluate(values):
        nonlocal best, best_norm
        consume(0)
        values = state(values)
        residual = np.asarray(checked_system_call(system, system.residual, values.copy(), zero))
        if residual.shape != original.shape or residual.dtype != original.dtype:
            raise ValueError("initializer residual must match the NumPy coefficient vector")
        norm = checked_system_call(system, system.residual_norm, residual)
        if not np.all(np.isfinite(residual)) or not system.isfinite(norm):
            raise OverflowError("nonfinite initializer residual")
        if norm < zero:
            raise ValueError("residual_norm must be nonnegative")
        if norm < best_norm:
            best, best_norm = values.copy(), system.scalar(norm)
        emit("initializer_evaluation", level=2, throttle=True,
             initializer="fsolve", evaluations=sum(counts), budget=int(max_evaluations),
             function_evaluations=counts[0], jacobian_evaluations=counts[1],
             residual_norm=norm, best_residual_norm=best_norm, homotopy=zero)
        return residual

    def jacobian(values):
        consume(1)
        values = state(values)
        with (progress.phase("initializer_jacobian", level=3,
                             evaluations=sum(counts), homotopy=zero)
              if progress is not None else nullcontext()):
            matrix = np.asarray(checked_system_call(system, system.jacobian, values.copy(), zero))
        check_system_precision(system)
        if matrix.shape != (len(original), len(original)) or matrix.dtype != original.dtype:
            raise ValueError("initializer Jacobian must be a square NumPy float64 matrix")
        if not np.all(np.isfinite(matrix)):
            raise OverflowError("nonfinite initializer Jacobian")
        emit("initializer_jacobian", level=3, initializer="fsolve",
             evaluations=sum(counts), budget=int(max_evaluations),
             function_evaluations=counts[0], jacobian_evaluations=counts[1], homotopy=zero)
        return matrix

    status, success, message = None, False, ""
    reason = "no_improvement"
    try:
        evaluate(original)
        initial_norm = best_norm
        if initial_norm == zero:
            reason = "already_solved"
        elif sum(counts) >= max_evaluations:
            reason = "evaluation_limit"
        else:
            with (progress.phase("fsolve", level=1, budget=int(max_evaluations))
                  if progress is not None else nullcontext()):
                candidate, info, status, message = optimize.fsolve(
                    evaluate, original.copy(), fprime=jacobian,
                    xtol=float(tolerance), maxfev=int(max_evaluations), full_output=True,
                )
            status, message = int(status), str(message)
            success = status == 1
            # fsolve may return without a final callback at its reported state.
            # Re-evaluate if budget remains; otherwise use the verified best.
            if sum(counts) < max_evaluations:
                evaluate(candidate)
            reason = "improved" if best_norm < initial_norm else "no_improvement"
    except _EvaluationLimit:
        reason = "evaluation_limit"
        message = "initializer residual/Jacobian evaluation budget exhausted"
    except (OverflowError, np.linalg.LinAlgError) as error:
        reason = "numerical_failure"
        message = str(error)

    # A nonfinite starting residual cannot establish an improvement. Return
    # the original guess for the main driver's normal numerical-failure path.
    accepted = bool(system.isfinite(initial_norm) and best_norm < initial_norm)
    selected = best if accepted else original
    final_norm = best_norm if accepted else initial_norm
    result = InitializationResult(
        state=selected.copy(), accepted=accepted, reason=reason,
        initial_norm=initial_norm, final_norm=final_norm,
        evaluations=sum(counts), function_evaluations=counts[0], jacobian_evaluations=counts[1],
        scipy_status=status, scipy_success=success, message=message,
    )
    emit("initializer_end", initializer="fsolve", accepted=accepted, reason=reason,
         initial_norm=initial_norm, residual_norm=final_norm,
         evaluations=sum(counts), budget=int(max_evaluations),
         function_evaluations=counts[0], jacobian_evaluations=counts[1],
         scipy_status=status, scipy_success=success, message=message, homotopy=zero)
    return result
