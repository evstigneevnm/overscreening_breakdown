"""Opt-in historical SciPy/VODE coefficient flow, outside the solver API.

This retains the old preconditioned flow ``-2 S**-1 J.T R`` for research. It is
not the native residual-merit adjoint fallback used by ``solve``: the historical
matrix S need not define a positive metric. No convergence guarantee is made.
"""

import numpy as np
from scipy.integrate import ode


def adjoint_descent(legacy, coefficients, exec_time):
    """Run the original NumPy-only VODE experiment on a configured instance.

    ``legacy`` is the canonical ``solve_nonlinear_problem`` instance after
    ``set_problem``. This intentionally uses its private legacy operators;
    changes to that research interface may require updating this experiment.
    The inverse is computed only here, rather than cached by every solver.
    """
    if legacy.is_using_mpmath():
        raise ValueError("The historical SciPy/VODE flow requires NumPy arithmetic.")
    duration = float(exec_time)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("exec_time must be positive and finite.")
    matrix = legacy._solve_nonlinear_problem__S
    if matrix is None:
        raise ValueError("Call set_problem before running the historical flow.")
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    initial = np.asarray(coefficients, dtype=np.float64).copy()
    residual = legacy._solve_nonlinear_problem__residual_operator
    jacobian = legacy._solve_nonlinear_problem__linearization_operator

    def flow(t, state):
        print(t)
        return np.dot(inverse, -2*np.dot(jacobian(state).transpose(), residual(state)))

    integrator = ode(flow).set_integrator("vode", rtol=1.0e-3, atol=1.0e-5)
    integrator.set_initial_value(initial, 0)
    residual_before = legacy.residual_L2_norm(initial)
    dt = duration/1000
    while integrator.successful() and integrator.t < duration:
        integrator.integrate(integrator.t+dt)
    print(integrator)
    result = integrator.y
    print("residual_norm_b4 =", residual_before,
          "residual_norm_aft =", legacy.residual_L2_norm(result))
    return result
