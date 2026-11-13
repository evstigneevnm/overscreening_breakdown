"""Reference-based continuation of complete discrete residuals.

Physical callbacks remain unscaled and independent of continuation. Each
discretization builds an affine reference in its own basis and arithmetic;
this module supplies the common endpoint blend and adaptive stage controller.
"""

from dataclasses import dataclass, replace
from numbers import Integral

from overscreening_breakdown.nonlinear_solver import (
    LinearSolveError, NewtonResult, check_system_precision, checked_system_call, solve_newton,
)


@dataclass(frozen=True)
class ScreenedLinearReference:
    """Constant-wall-epsilon reference with normalized reaction ``strength*u``.

    Collocation uses epsilon=1. Galerkin uses epsilon_ref=epsilon(0+) and
    physical f_ref=epsilon(0+)*strength*u. Values are converted only after
    binding to the solver's numerical context.
    """

    strength: object = "1"


@dataclass(frozen=True)
class ContinuationOptions:
    """Bounded adaptive stepping from the verified reference q=1 to q=0.

    ``max_stages`` counts continuation attempts, including rejected stages;
    the initial physical attempt is additional. ``max_total_iterations``
    limits attempted Newton steps across the entire solve, including that
    physical attempt. Reference construction uses a separate linear solve.
    """

    reference: ScreenedLinearReference = ScreenedLinearReference()
    initial_step: object = "0.25"
    min_step: object = "1e-6"
    max_step: object = "0.5"
    growth: object = "1.5"
    max_stages: int = 100
    max_total_iterations: int = 1000
    max_reference_corrections: int = 3

    def bind(self, scalar, isfinite):
        if not isinstance(self.reference, ScreenedLinearReference):
            raise TypeError("reference must be a ScreenedLinearReference")
        for name in ("max_stages", "max_total_iterations", "max_reference_corrections"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        values = {}
        for name in ("initial_step", "min_step", "max_step", "growth"):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ValueError(f"{name} must be a finite positive scalar")
            values[name] = scalar(raw)
            if not isfinite(values[name]):
                raise ValueError(f"{name} must be finite")
        strength = scalar(self.reference.strength)
        if isinstance(self.reference.strength, bool) or not isfinite(strength) or strength <= 0:
            raise ValueError("reference strength must be finite and positive")
        if not 0 < values["min_step"] <= values["initial_step"] <= values["max_step"] <= 1:
            raise ValueError("require 0 < min_step <= initial_step <= max_step <= 1")
        if values["growth"] <= 1:
            raise ValueError("growth must be greater than one")
        return replace(self, reference=ScreenedLinearReference(strength),
                       max_stages=int(self.max_stages),
                       max_total_iterations=int(self.max_total_iterations),
                       max_reference_corrections=int(self.max_reference_corrections), **values)


def _entries(matrix):
    for row in matrix:
        try:
            yield from iter(row)
        except TypeError:
            yield row


def _matvec(matrix, vector):
    # mpmath uses * for matrix products; NumPy arrays use dot.
    return matrix*vector if hasattr(matrix, "rows") else matrix.dot(vector)


class AffineReference:
    """Native copies of a square reference operator and its affine load."""

    def __init__(self, matrix, offset, system):
        check_system_precision(system)
        shape = ((matrix.rows, matrix.cols) if hasattr(matrix, "rows")
                 else getattr(matrix, "shape", ()))
        if shape != (len(offset), len(offset)) or len(offset) == 0:
            raise ValueError("reference matrix and offset must define a nonempty square system")
        if len(checked_system_call(system, system.initial_guess)) != len(offset):
            raise ValueError("physical and reference coefficient counts differ")
        if not all(system.isfinite(value) for value in _entries(matrix)):
            raise OverflowError("reference matrix must be finite")
        if not all(system.isfinite(value) for value in offset):
            raise OverflowError("reference affine load must be finite")
        self._matrix = matrix.copy()
        self._offset = offset.copy()
        self.system = system

    def check_precision(self):
        check_system_precision(self.system)

    @property
    def matrix(self):
        self.check_precision()
        return self._matrix.copy()

    @property
    def offset(self):
        self.check_precision()
        return self._offset.copy()

    def residual(self, state):
        self.check_precision()
        if len(state) != len(self._offset):
            raise ValueError("reference state has an incompatible coefficient count")
        return _matvec(self._matrix, state) + self._offset

    def solve(self):
        return checked_system_call(self.system, self.system.solve_linear, self.matrix, -self.offset)


class ResidualHomotopy:
    """Blend endpoint systems, preserving all affine boundary equations.

    The physical system's norm and merit geometry apply to the full blended
    residual and Jacobian. In particular, adjoint recovery must not blend
    separately calculated endpoint gradients.
    """

    def __init__(self, system, reference):
        if not isinstance(reference, AffineReference) or reference.system is not system:
            raise ValueError("reference must belong to this physical system")
        self.physical = system
        self.reference = reference
        self.check_precision()

    def check_precision(self):
        check_system_precision(self.physical)
        self.reference.check_precision()

    def _parameter(self, q):
        self.check_precision()
        q = self.physical.scalar(q)
        if not self.physical.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("homotopy parameter must be finite and lie in [0, 1]")
        return q

    def residual(self, state, q):
        q = self._parameter(q)
        if q == 0:
            return checked_system_call(self.physical, self.physical.residual,
                                       state, self.physical.scalar(0))
        if q == 1:
            return self.reference.residual(state)
        return ((1-q)*checked_system_call(self.physical, self.physical.residual,
                                         state, self.physical.scalar(0))
                + q*self.reference.residual(state))

    def jacobian(self, state, q):
        q = self._parameter(q)
        if q == 0:
            return checked_system_call(self.physical, self.physical.jacobian,
                                       state, self.physical.scalar(0))
        if q == 1:
            return self.reference.matrix
        return ((1-q)*checked_system_call(self.physical, self.physical.jacobian,
                                         state, self.physical.scalar(0))
                + q*self.reference.matrix)

    def nonlinear_system(self, initial_guess=None):
        self.check_precision()
        initial = (checked_system_call(self.physical, self.physical.initial_guess)
                   if initial_guess is None else initial_guess).copy()

        def copy_initial():
            self.check_precision()
            return initial.copy()

        return replace(self.physical, residual=self.residual, jacobian=self.jacobian,
                       initial_guess=copy_initial, restart=None, check_precision=self.check_precision)


@dataclass(frozen=True)
class ReferenceResult:
    state: object
    converged: bool
    residual_norm: object
    target_norm: object
    strength: object
    reason: str
    linear_solves: int = 1


def solve_continuation(
    system, *, reference_factory=None, use_globalization=True, options=None,
    atol="1e-8", rtol="0", max_iterations=50, callback=None,
    newton_fallback=None, adjoint_max_steps=20, progress=None,
):
    """Try the physical system, then recover along a verified reference path.

    ``reference_factory(strength)`` lazily returns the native affine matrix
    and load for the same basis, boundary data, normalization and fixed plan.
    A failed trial is never the starting state for another continuation
    attempt. Only the last successfully solved parameter/state is retained.
    """
    check_system_precision(system)
    if not isinstance(use_globalization, bool):
        raise TypeError("use_globalization must be a boolean")
    if options is None:
        options = ContinuationOptions()
    if not isinstance(options, ContinuationOptions):
        raise TypeError("continuation must be ContinuationOptions or None")
    options = options.bind(system.scalar, system.isfinite)
    scalar, finite = system.scalar, system.isfinite
    zero, one = scalar(0), scalar(1)
    atol, rtol = scalar(atol), scalar(rtol)
    results = []
    reference_result = None
    total_iterations = attempts = 0

    def emit(kind, *, level=1, **data):
        if progress is not None:
            checked_system_call(system, progress.emit, kind, level=level, **data)

    def collect(result):
        nonlocal total_iterations
        results.append(result)
        total_iterations += result.iterations
        return result

    def finish(state, q, norm, converged, reason):
        if converged:
            # Success always certifies a fresh evaluation of the actual
            # physical callback, including when the initial q=0 solve succeeds.
            emit("physical_verification_start", homotopy=zero)
            target = results[-1].stages[-1].target_norm
            try:
                values = checked_system_call(system, system.residual, state, zero)
                norm = checked_system_call(system, system.residual_norm, values)
                converged = bool(all(finite(value) for value in values)
                                 and finite(norm) and norm <= target)
            except OverflowError:
                norm, converged = scalar("inf"), False
            if not converged:
                reason = "physical_verification_failed"
            emit("physical_verification_end", homotopy=zero, converged=converged,
                 residual_norm=norm, target_norm=target)
        emit("continuation_end", converged=converged, reason=reason, homotopy=q,
             residual_norm=norm, iterations=total_iterations,
             continuation_attempts=attempts)
        return NewtonResult(
            state.copy(), converged, reason, q, norm, total_iterations,
            tuple(stage for result in results for stage in result.stages),
            tuple(record for result in results for record in result.history),
            sum(result.adjoint_steps for result in results),
            sum(result.adjoint_evaluations for result in results),
            tuple(reason for result in results for reason in result.recovery_reasons),
            reference_result=reference_result, continuation_attempts=attempts,
        )

    physical = collect(solve_newton(
        system, atol=atol, rtol=rtol,
        max_iterations=min(max_iterations, options.max_total_iterations),
        callback=callback, newton_fallback=newton_fallback,
        adjoint_max_steps=adjoint_max_steps, progress=progress,
    ))
    if physical.converged or not use_globalization:
        return finish(physical.state, zero, physical.residual_norm,
                      physical.converged, physical.reason)
    if total_iterations >= options.max_total_iterations:
        return finish(physical.state, zero, physical.residual_norm, False,
                      "max_total_iterations")
    if options.max_stages == 0:
        return finish(physical.state, zero, physical.residual_norm, False, "max_stages")
    if not callable(reference_factory):
        raise TypeError("reference_factory is required when continuation is needed")

    emit("reference_start", homotopy=one, strength=options.reference.strength)
    try:
        if progress is not None:
            with progress.phase("reference_assembly"):
                matrix, offset = checked_system_call(system, reference_factory,
                                                     options.reference.strength)
        else:
            matrix, offset = checked_system_call(system, reference_factory,
                                                 options.reference.strength)
        reference = AffineReference(matrix, offset, system)
        if progress is not None:
            with progress.phase("reference_linear_solve"):
                state = reference.solve()
        else:
            state = reference.solve()
        if not all(finite(value) for value in state):
            raise OverflowError("reference solution is nonfinite")
        norm = checked_system_call(system, system.residual_norm, reference.residual(state))
        initial_norm = checked_system_call(system, system.residual_norm, offset)
        target = atol+rtol*initial_norm if finite(initial_norm) else atol
        if not finite(target):
            raise ValueError("reference tolerance overflowed in the selected arithmetic")
        if norm < zero:
            raise ValueError("residual_norm must be nonnegative")
        linear_solves = 1
        # Native iterative refinement can recover a valid reference when the
        # first factorization leaves cancellation error above the target.
        # Never relax the target or replace a better state by a worse one.
        for _ in range(options.max_reference_corrections):
            if finite(norm) and norm <= target:
                break
            correction = checked_system_call(system, system.solve_linear, reference.matrix,
                                              -reference.residual(state))
            linear_solves += 1
            candidate = state+correction
            if not all(finite(value) for value in candidate):
                break
            candidate_norm = checked_system_call(system, system.residual_norm,
                                                 reference.residual(candidate))
            emit("reference_correction", homotopy=one, linear_solves=linear_solves,
                 residual_norm=candidate_norm, target_norm=target)
            if candidate_norm < zero:
                raise ValueError("residual_norm must be nonnegative")
            if not finite(candidate_norm) or candidate_norm >= norm:
                break
            state, norm = candidate, candidate_norm
        verified = bool(finite(norm) and norm <= target)
        reference_result = ReferenceResult(
            state.copy(), verified, norm, target, options.reference.strength,
            "converged" if verified else "reference_not_converged",
            linear_solves=linear_solves,
        )
    except (LinearSolveError, OverflowError) as error:
        emit("reference_end", homotopy=one, converged=False,
             reason="reference_construction_failed", message=str(error))
        return finish(physical.state, zero, physical.residual_norm, False,
                      "reference_construction_failed")
    emit("reference_end", homotopy=one, converged=verified,
         residual_norm=norm, target_norm=target, reason=reference_result.reason,
         linear_solves=linear_solves)
    if not verified:
        return finish(physical.state, zero, physical.residual_norm, False,
                      "reference_not_converged")

    homotopy = ResidualHomotopy(system, reference)
    accepted_q, accepted_state, accepted_norm = one, state.copy(), norm
    step = options.initial_step
    emit("continuation_start", homotopy=one, step=step)
    for _ in range(int(options.max_stages)):
        remaining = int(options.max_total_iterations)-total_iterations
        if remaining <= 0:
            return finish(accepted_state, accepted_q, accepted_norm, False,
                          "max_total_iterations")
        trial_q = max(zero, accepted_q-step)
        if trial_q == accepted_q:
            return finish(accepted_state, accepted_q, accepted_norm, False,
                          "parameter_step_underflow")
        attempts += 1
        emit("homotopy_transition", stage=len(results)+1, previous_homotopy=accepted_q,
             homotopy=trial_q, step=step)
        trial = collect(solve_newton(
            homotopy.nonlinear_system(accepted_state), parameter=trial_q,
            stage=len(results)+1, iteration_offset=total_iterations,
            atol=atol, rtol=rtol, max_iterations=min(max_iterations, remaining),
            callback=callback, newton_fallback=newton_fallback,
            adjoint_max_steps=adjoint_max_steps, progress=progress,
        ))
        if trial.converged:
            accepted_q, accepted_state = trial_q, trial.state.copy()
            accepted_norm = trial.residual_norm
            emit("continuation_accepted", homotopy=accepted_q, residual_norm=accepted_norm,
                 step=step, continuation_attempts=attempts)
            if accepted_q == zero:
                return finish(accepted_state, zero, accepted_norm, True, "converged")
            step = (options.max_step if step >= options.max_step/options.growth
                    else step*options.growth)
        else:
            emit("continuation_rejected", homotopy=trial_q,
                 accepted_homotopy=accepted_q, reason=trial.reason,
                 residual_norm=trial.residual_norm, step=step)
            step = (accepted_q-trial_q)/scalar(2)
            if step < options.min_step:
                return finish(accepted_state, accepted_q, accepted_norm, False,
                              "min_parameter_step")
    return finish(accepted_state, accepted_q, accepted_norm, False, "max_stages")
