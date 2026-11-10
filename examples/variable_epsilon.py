"""Use the mixed solver for a smooth or piecewise-smooth epsilon coefficient."""

from overscreening_breakdown import solve
from overscreening_breakdown.manufactured import rational_case, transmission_case

if __package__:
    from ._common import backend_choices, parser, report, sampled_error, tolerance, write_outputs
else:
    from _common import backend_choices, parser, report, sampled_error, tolerance, write_outputs


def make_case(coefficient="smooth"):
    if coefficient == "smooth":
        return rational_case(coefficient="smooth", nonlinear=True)
    if coefficient == "jump":
        return transmission_case()
    raise ValueError("coefficient must be smooth or jump")


def run(*, use_mpmath=False, precision=50, coefficient="smooth", problem=None,
        max_refinements=3):
    """Return an evaluable mixed solution using declared coefficient interfaces."""
    case = make_case(coefficient)
    solution = solve(case.make_problem() if problem is None else problem,
                     16 if coefficient == "jump" else 4, solver="galerkin",
                     use_mpmath=use_mpmath, precision=precision,
                     tolerance=tolerance(use_mpmath, precision), max_refinements=max_refinements,
                     # At N=16 the float64 residual comparison reaches an
                     # assembly-roundoff floor around 1e-11. Keep the Newton
                     # target separate from this quadrature consistency target.
                     quadrature_atol="1e-9" if coefficient == "jump" and not use_mpmath else None)
    if not solution.converged:
        raise RuntimeError("The variable-epsilon example failed its nonlinear or quadrature check")
    return solution


def main(argv=None):
    arguments = parser(__doc__)
    arguments.add_argument("--coefficient", choices=("smooth", "jump"), default="smooth")
    options = arguments.parse_args(argv)
    case = make_case(options.coefficient)
    problem = case.make_problem()
    for use_mpmath in backend_choices(options.backend):
        solution = run(problem=problem, use_mpmath=use_mpmath, precision=options.precision,
                       coefficient=options.coefficient)
        report(solution, sampled_error(solution, case))
        print("epsilon interfaces: {}".format(solution.physical.epsilon.interfaces))
        if options.output_dir is not None:
            write_outputs(solution, options.output_dir,
                          "variable-"+options.coefficient+"-"+("mpmath" if use_mpmath else "numpy"))


if __name__ == "__main__":
    main()
