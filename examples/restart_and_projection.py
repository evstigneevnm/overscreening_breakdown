"""Save typed solutions, restart the same solver, and explicitly project once."""

from pathlib import Path
import tempfile

from overscreening_breakdown import load_solution, project_collocation, solve
from overscreening_breakdown.manufactured import rational_case

if __package__:
    from ._common import backend_choices, parser, report, tolerance, write_outputs
else:
    from _common import backend_choices, parser, report, tolerance, write_outputs


def run(directory, *, use_mpmath=True, precision=50, verbosity=0,
        progress_callback=None, progress_interval=1.0):
    """Return the collocation source and the restored two-field solution."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    case = rational_case(nonlinear=True, beta="0.1", initial="exp")
    problem = case.make_problem()
    target = tolerance(use_mpmath, precision)
    options = dict(use_mpmath=use_mpmath, precision=precision, tolerance=target,
                   verbosity=verbosity, progress_callback=progress_callback,
                   progress_interval=progress_interval)
    source = solve(problem, 8, solver="collocation", **options)
    if not source.converged:
        raise RuntimeError("The collocation source did not converge")
    source.save(directory/"collocation.json")
    source_restart = load_solution(directory/"collocation.json", problem=problem,
                                  solver="collocation")
    restarted = solve(problem, 8, solver="collocation", initial_guess=source_restart, **options)
    if not restarted.converged:
        raise RuntimeError("The saved collocation restart did not converge")
    projected = project_collocation(restarted, N=4,
                                    trace_tolerance=target if use_mpmath else "1e-8")
    projected.save(directory/"projected.json")
    if projected.converged:
        raise RuntimeError("A projection must not claim nonlinear convergence")
    mixed = solve(problem, 4, solver="galerkin", initial_guess=projected, **options)
    if not mixed.converged:
        raise RuntimeError("The projected Galerkin restart did not converge")
    mixed.save(directory/"galerkin.json")
    restored = load_solution(directory/"galerkin.json", problem=problem, solver="galerkin")
    report(source)
    report(restored)
    print("restored u(1)={}, w(1)={}".format(restored.u(1), restored.w(1)))
    return source, restored


def main(argv=None):
    options = parser(__doc__).parse_args(argv)
    for use_mpmath in backend_choices(options.backend):
        backend = "mpmath" if use_mpmath else "numpy"
        if options.output_dir is None:
            with tempfile.TemporaryDirectory(prefix="overscreening-restart-", dir="/tmp") as directory:
                run(directory, use_mpmath=use_mpmath, precision=options.precision,
                    verbosity=options.verbosity)
        else:
            directory = options.output_dir/backend
            _, restored = run(directory, use_mpmath=use_mpmath, precision=options.precision,
                              verbosity=options.verbosity)
            write_outputs(restored, directory, "restored-galerkin")


if __name__ == "__main__":
    main()
