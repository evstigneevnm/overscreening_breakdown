"""Solve a catalog FluxProblem with collocation, in NumPy or owned mpmath."""

from overscreening_breakdown import solve
from overscreening_breakdown.manufactured import rational_case

if __package__:
    from ._common import backend_choices, parser, report, sampled_error, tolerance, write_outputs
else:
    from _common import backend_choices, parser, report, sampled_error, tolerance, write_outputs


def make_case():
    return rational_case(nonlinear=True, beta="0.1", initial="exp")


def run(*, use_mpmath=False, precision=50, method="newton", problem=None):
    """Return an evaluable solution; the physical definition is reused unchanged."""
    case = make_case()
    solution = solve(case.make_problem() if problem is None else problem, 8,
                     solver="collocation", use_mpmath=use_mpmath, precision=precision,
                     method=method, tolerance=tolerance(use_mpmath, precision))
    if not solution.converged:
        raise RuntimeError("The collocation example did not converge")
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
                          "collocation-"+("mpmath" if use_mpmath else "numpy"))


if __name__ == "__main__":
    main()
