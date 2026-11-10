"""Compatibility entry point for the catalog-based collocation example.

Use ``python -m examples.collocation`` for the current command-line example.
``solve_example`` retains its three return values; the first is now the common
solution with ``converged`` and ``newton_result`` diagnostics.
"""

if __package__:
    from ._common import sampled_error
    from .collocation import main, make_case, run
else:
    from _common import sampled_error
    from collocation import main, make_case, run


def make_problem():
    return make_case().make_problem()


def solve_example(problem, *, use_mpmath=False, precision=50):
    solution = run(problem=problem, use_mpmath=use_mpmath, precision=precision)
    return solution, solution.coefficients, sampled_error(solution, make_case())


if __name__ == "__main__":
    main()
