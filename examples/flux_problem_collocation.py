"""Solve one physical FluxProblem with NumPy and mpmath collocation.

From the repository root, run::

    python -B -m examples.flux_problem_collocation

The manufactured solution is u(x) = 1 / (1 + x), with delta = 2,
epsilon = 1, u(0) = 1, and u'''(0) = -6. Both runs reuse the same
arithmetic-independent problem description. No plotting is required.
"""

import numpy as np

from arithmetic import MpmathArithmetic, NumpyArithmetic
from physical_problem import FluxProblem
from spectral1d import solve_nonlinear_problem


def exact_solution(x, arithmetic):
    return arithmetic.scalar(1) / (arithmetic.scalar(1) + x)


def physical_source(x, u, arithmetic):
    """Unscaled f for u'' - 4 u'''' = f(x, u)."""
    one = arithmetic.scalar(1)
    denominator = one + x
    exact = exact_solution(x, arithmetic)
    return (
        arithmetic.scalar(2) / denominator**3
        - arithmetic.scalar(96) / denominator**5
        + (u - exact)
        + (u**3 - exact**3) / arithmetic.scalar(10)
    )


def physical_source_u(x, u, arithmetic):
    return arithmetic.scalar(1) + arithmetic.scalar(3) * u**2 / arithmetic.scalar(10)


def initial_guess(x, arithmetic):
    # A different field from the exact solution exercises nonlinear iteration.
    return exact_solution(x, arithmetic)**2


def make_problem():
    return FluxProblem(
        delta="2",
        epsilon=1,
        f=physical_source,
        f_u=physical_source_u,
        boundary_conditions=(
            ("1", None, None, "-6", None),
            (None, None, None, None, None),
        ),
        initial_guess=initial_guess,
    )


def solve_example(problem, *, use_mpmath=False, precision=50):
    """Return the solver, coefficients, and sampled error in native arithmetic."""
    arithmetic = MpmathArithmetic(precision) if use_mpmath else NumpyArithmetic()
    solver = solve_nonlinear_problem(
        N=8,
        L=1,
        problem=problem,
        use_method="newton" if use_mpmath else "root",
        use_mpmath=use_mpmath,
        prec=precision,
        # Leave guard digits for cancellation in the fourth-order operator.
        tolerance="1e-{}".format(max(10, precision - 15)) if use_mpmath else 1e-11,
        use_globalization=True,
    )
    coefficients = solver.solve_problem()
    if not solver.is_converged():
        raise RuntimeError("The manufactured collocation problem did not converge")

    # The public evaluation API accepts one-dimensional NumPy arrays in either
    # backend; retain mpmath scalar entries through an object array.
    samples = np.array(
        [arithmetic.scalar(value) for value in
         ("0", "0.1", "0.5", "1", "2", "5", "10")],
        dtype=object if use_mpmath else float,
    )
    numerical = solver.solution_in_domain(coefficients, samples)
    error = max(abs(value - exact_solution(point, arithmetic))
                for point, value in zip(samples, numerical))
    return solver, coefficients, error


def main():
    problem = make_problem()
    for use_mpmath in (False, True):
        solver, _, error = solve_example(problem, use_mpmath=use_mpmath)
        backend = "mpmath, 50 digits" if use_mpmath else "NumPy, float64"
        print("{}: converged={}, max sampled error={}".format(
            backend, solver.is_converged(), error))


if __name__ == "__main__":
    main()
