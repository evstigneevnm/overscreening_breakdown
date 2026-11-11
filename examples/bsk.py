"""Describe a BSK problem immutably and solve it with either discretization."""

from overscreening_breakdown import BSKProblem, solve

if __package__:
    from ._common import backend_choices, parser, report, tolerance, write_outputs
else:
    from _common import backend_choices, parser, report, tolerance, write_outputs


def make_problem():
    """Every parameter belongs to the physical equation, before normalization."""
    return BSKProblem(
        sigma="2", mu="0.1", gamma="0.5", delta="1", epsilon=1,
        boundary_conditions=(("0.05", None, None, 0, None), (None,)*5),
    )


def run(*, solver="galerkin", use_mpmath=False, precision=50, problem=None,
        verbosity=0, progress_callback=None, progress_interval=1.0):
    problem = make_problem() if problem is None else problem
    # The Gaussian source requires quadrature refinement independently of
    # Newton accuracy. This bounded example requests a separate consistency
    # target; using 50-digit arithmetic does not certify 50-digit integration.
    quadrature = (dict(quadrature_atol="1e-12", quadrature_rtol="1e-12")
                  if solver == "galerkin" else {})
    solution = solve(problem, 8, solver=solver, use_mpmath=use_mpmath,
                     precision=precision, tolerance=tolerance(use_mpmath, precision),
                     verbosity=verbosity, progress_callback=progress_callback,
                     progress_interval=progress_interval, **quadrature)
    if not solution.converged:
        raise RuntimeError("The BSK example failed its nonlinear or quadrature check")
    return solution


def main(argv=None):
    arguments = parser(__doc__)
    arguments.add_argument("--solver", choices=("collocation", "galerkin", "both"), default="both")
    options = arguments.parse_args(argv)
    methods = ("collocation", "galerkin") if options.solver == "both" else (options.solver,)
    problem = make_problem()
    for use_mpmath in backend_choices(options.backend):
        for method in methods:
            solution = run(solver=method, problem=problem, use_mpmath=use_mpmath,
                           precision=options.precision, verbosity=options.verbosity)
            report(solution)
            print("u(0)={}, delta={}".format(solution.u(0), solution.physical.delta))
            if method == "galerkin":
                print("quadrature consistency targets: atol=rtol=1e-12")
            if options.output_dir is not None:
                write_outputs(solution, options.output_dir,
                              "bsk-"+method+"-"+("mpmath" if use_mpmath else "numpy"))


if __name__ == "__main__":
    main()
