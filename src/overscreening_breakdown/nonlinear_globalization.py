"""Native discrete merit descent at one fixed system and homotopy parameter.

No ODE integrator or backend conversion is involved. The merit is
``0.5*residual_norm(R)**2``. A supplied merit_gradient must be its derivative
in coefficient coordinates; by default the residual norm is assumed Euclidean
and J.T*R is used. The optional direction callback may apply an SPD inverse
metric or an admissible tangent-space projection. Descent is verified before
line search rather than inferred from a preconditioner name.
"""

from dataclasses import dataclass

from overscreening_breakdown.nonlinear_solver import check_system_precision, checked_system_call


@dataclass(frozen=True)
class AdjointStepResult:
    state: object
    residual: object
    residual_norm: object
    accepted: bool
    reason: str
    step: object
    backtracks: int
    initial_merit: object
    merit: object
    gradient_norm: object
    directional_derivative: object
    evaluations: int


def adjoint_step(system, state, residual, jacobian, parameter, *, residual_norm,
                 evaluate, max_backtracks=34, min_step="1e-10", armijo="1e-4"):
    """Try one bounded Armijo step without changing q or accepting a no-op.

    ``evaluate(candidate, parameter)`` returns (residual, norm), using None for
    an inadmissible/nonfinite trial. Evaluations count finite trial states
    submitted to this callback. Numerical overflow is a rejected recovery;
    programming exceptions propagate. No user observer is called here.
    """
    check_system_precision(system)
    scalar, finite = system.scalar, system.isfinite
    zero, one, half = scalar(0), scalar(1), scalar("0.5")
    minimum, sufficient = scalar(min_step), scalar(armijo)
    if (isinstance(max_backtracks, bool) or not isinstance(max_backtracks, int)
            or max_backtracks < 1):
        raise ValueError("max_backtracks must be a positive integer")
    if not finite(minimum) or not zero < minimum <= one:
        raise ValueError("min_step must be finite and in (0,1]")
    if not finite(sufficient) or not zero < sufficient < one:
        raise ValueError("armijo must be finite and in (0,1)")
    initial_merit = half*residual_norm*residual_norm
    gradient_norm = slope = None
    evaluations = 0

    def finish(reason, *, candidate=None, candidate_residual=None, norm=None,
               weight=None, backtracks=0, merit=None):
        check_system_precision(system)
        accepted = candidate is not None
        return AdjointStepResult(
            candidate if accepted else state.copy(),
            candidate_residual if accepted else residual,
            norm if accepted else residual_norm, accepted, reason,
            weight if accepted else zero, backtracks, initial_merit,
            initial_merit if merit is None else merit, gradient_norm, slope, evaluations,
        )

    def dot(left, right):
        if len(left) != len(right):
            raise ValueError("coefficient dot products require equal vector lengths")
        callback = system.dot
        return scalar(checked_system_call(system, callback, left, right) if callback is not None
                      else sum((a*b for a, b in zip(left, right)), zero))

    if not finite(initial_merit):
        return finish("adjoint_nonfinite_merit")
    try:
        if system.merit_gradient is None:
            gradient = state.copy()
            for column in range(len(state)):
                gradient[column] = sum((jacobian[row, column]*value
                                        for row, value in enumerate(residual)), zero)
        else:
            gradient = checked_system_call(system, system.merit_gradient, residual, jacobian)
        if len(gradient) != len(state):
            raise ValueError("merit gradient must match the state dimension")
        if not all(finite(value) for value in gradient):
            return finish("adjoint_nonfinite_gradient")
        scale = max((abs(value) for value in gradient), default=zero)
        if scale == zero:
            gradient_norm = zero
            return finish("adjoint_stationary")
        scaled = gradient.copy()
        for index, value in enumerate(gradient):
            scaled[index] = value/scale
        squared = dot(scaled, scaled)
        if not finite(squared):
            return finish("adjoint_nonfinite_gradient")
        if squared < zero:
            raise ValueError("coefficient dot product must define a nonnegative squared norm")
        gradient_norm = scale*squared**half
        if not finite(gradient_norm):
            return finish("adjoint_nonfinite_gradient")
        direction = (-gradient if system.adjoint_direction is None
                     else checked_system_call(system, system.adjoint_direction, gradient))
        if len(direction) != len(state):
            raise ValueError("adjoint direction must match the state dimension")
        if not all(finite(value) for value in direction):
            return finish("adjoint_nonfinite_direction")
        size = max((abs(value) for value in direction), default=zero)
        if size == zero:
            return finish("adjoint_stationary")
        if size > one:
            direction = direction/size
        slope = dot(gradient, direction)
        if not finite(slope):
            return finish("adjoint_nonfinite_direction")
        if slope >= zero:
            return finish("adjoint_stationary" if slope == zero else "adjoint_not_descent")
    except OverflowError:
        return finish("adjoint_overflow")

    weight = one
    for backtracks in range(max_backtracks):
        if weight < minimum:
            return finish("adjoint_line_search_failed", backtracks=backtracks)
        candidate = state+weight*direction
        if all(finite(value) for value in candidate):
            evaluations += 1
            candidate_residual, candidate_norm = checked_system_call(
                system, evaluate, candidate, parameter)
            if candidate_residual is not None:
                merit = half*candidate_norm*candidate_norm
                if (finite(merit) and candidate_norm < residual_norm
                        and merit <= initial_merit+sufficient*weight*slope):
                    return finish("adjoint_accepted", candidate=candidate,
                                  candidate_residual=candidate_residual, norm=candidate_norm,
                                  weight=weight, backtracks=backtracks, merit=merit)
        weight *= half
    return finish("adjoint_line_search_failed", backtracks=max_backtracks)
