"""Solve the same constant-epsilon FluxProblem with the two-field method."""

from overscreening_breakdown import solve
from overscreening_breakdown.manufactured import rational_case

if __package__:
    from ._common import backend_choices, parser, report, sampled_error, tolerance, write_outputs
else:
    from _common import backend_choices, parser, report, sampled_error, tolerance, write_outputs


def make_case():
    return rational_case(nonlinear=True)


def run(*, use_mpmath=False, precision=50, problem=None, max_refinements=3):
    """Return both independent fields with separate Newton/quadrature status."""
    case = make_case()
    solution = solve(case.make_problem() if problem is None else problem, 4,
                     solver="galerkin", use_mpmath=use_mpmath, precision=precision,
                     tolerance=tolerance(use_mpmath, precision), max_refinements=max_refinements)
    if not solution.converged:
        raise RuntimeError("The Galerkin example failed its nonlinear or quadrature check")
    return solution


def main(argv=None):
    options = parser(__doc__).parse_args(argv)
    case = make_case()
    problem = case.make_problem()
    for use_mpmath in backend_choices(options.backend):
        solution = run(problem=problem, use_mpmath=use_mpmath, precision=options.precision)
        report(solution, sampled_error(solution, case))
        if options.output_dir is not None:
            write_outputs(solution, options.output_dir,
                          "galerkin-"+("mpmath" if use_mpmath else "numpy"))


if __name__ == "__main__":
    main()
