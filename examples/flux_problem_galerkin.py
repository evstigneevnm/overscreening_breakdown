"""Compatibility entry point for the catalog-based variable-epsilon example.

Use ``python -m examples.variable_epsilon`` for the current example.
``solve_example`` retains four return values: the evaluable solution, Newton
result, final quadrature-history record, and sampled two-field error.
"""

if __package__:
    from ._common import sampled_error
    from .variable_epsilon import main, make_case, run
else:
    from _common import sampled_error
    from variable_epsilon import main, make_case, run


def make_problem():
    return make_case().make_problem()


def solve_example(problem, *, use_mpmath=False, precision=50, max_refinements=3):
    solution = run(problem=problem, use_mpmath=use_mpmath, precision=precision,
                   max_refinements=max_refinements)
    return (solution, solution.newton_result, solution.quadrature_history[-1],
            sampled_error(solution, make_case()))


if __name__ == "__main__":
    main()
