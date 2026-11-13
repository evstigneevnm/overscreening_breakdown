"""Timing observations cover numerical work without requesting extra evaluations."""

import numpy as np

from overscreening_breakdown import solve
from overscreening_breakdown.diagnostics import ProgressReporter
from overscreening_breakdown.manufactured import rational_case
from overscreening_breakdown.nonlinear_solver import NonlinearSystem, solve_newton


def completed(events, phase):
    return [event for event in events if event.kind == "phase_end"
            and event.data.get("phase") == phase]


def test_newton_times_initial_and_rejected_trial_residuals():
    events, calls = [], []

    def residual(state, parameter):
        calls.append(state.copy())
        # An initially excessive Newton step is a recoverable rejected trial.
        if state[0] > 10:
            raise OverflowError("trial outside source range")
        return np.array([state[0]**3-1])

    system = NonlinearSystem(
        residual=residual, jacobian=lambda state, q: np.array([[3*state[0]**2]]),
        solve_linear=np.linalg.solve, residual_norm=np.linalg.norm,
        initial_guess=lambda: np.array([0.1]), scalar=np.float64, isfinite=np.isfinite,
    )
    result = solve_newton(system, atol="1e-12",
                          progress=ProgressReporter(progress_callback=events.append))
    assert result.converged
    timed = completed(events, "residual")
    assert len(timed) == len(calls)
    assert len(timed) > result.iterations+1
    assert timed[0].data["iteration"] == 0
    assert any(not event.data["success"] for event in timed)
    assert any(event.kind == "trial_rejected" for event in events)


def test_public_setup_and_projection_phases_enclose_the_work():
    case = rational_case(nonlinear=True, beta="0.1", initial="exp")
    for solver, resolution in (("collocation", 6), ("galerkin", 4)):
        events = []
        result = solve(case.make_problem(), resolution, solver=solver,
                       use_globalization=False, progress_callback=events.append)
        assert result.converged
        assert completed(events, "residual")
        assert completed(events, "physical_verification")
        if solver == "collocation":
            assert len(completed(events, "collocation_setup")) == 1
        else:
            stack, projection_solves = [], []
            for event in events:
                if event.kind == "phase_start":
                    stack.append(event.data["phase"])
                    if event.data.get("purpose") == "initial_projection":
                        projection_solves.append(tuple(stack))
                elif event.kind == "phase_end":
                    assert stack.pop() == event.data["phase"]
            assert projection_solves == [("initial_projection", "linear_solve")]
            assert not stack


def test_scipy_callbacks_and_initializer_are_timed_individually():
    events = []
    result = solve(rational_case(nonlinear=True, initial="exp").make_problem(), 6,
                   solver="collocation", method="root", initializer="fsolve",
                   initializer_max_evaluations=40, use_globalization=False,
                   progress_callback=events.append)
    assert result.converged
    assert len(completed(events, "scipy_solve")) == 1
    scipy_evaluations = [event for event in events if event.kind == "scipy_evaluation"]
    assert len(completed(events, "scipy_residual")) == len(scipy_evaluations)
    initializer = result.initialization_result
    assert len(completed(events, "initializer_residual")) == initializer.function_evaluations
    assert len(completed(events, "initializer_jacobian")) == initializer.jacobian_evaluations
    assert not completed(events, "linear_solve")  # SciPy's internal factors are unobservable.
