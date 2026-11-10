"""Manufactured studies, with the historical full plot sweep explicitly opt-in.

Default arguments are the bounded benchmark runner's arguments. Invoke
``--legacy-full-sweep`` alone to retain the original expensive plotting sweep.
"""

import sys


def _legacy_full_sweep():
    from overscreening_breakdown.manufactured import legacy_problems
    from overscreening_breakdown.manufactured.legacy_runner import (
        linear_manufactured_solutions_solver, nonlinear_manufactured_solutions_solver,
    )

    for use_mp in (False, True):
        linear = linear_manufactured_solutions_solver(
            default_folder="./figures", show_figures=False, use_mpmath=use_mp, prec=100)
        nonlinear = nonlinear_manufactured_solutions_solver(
            default_folder="./figures", show_figures=False, use_mpmath=use_mp, prec=100)
        for number in range(1, 8):
            linear.solve(getattr(legacy_problems, "problem_{}".format(number))())
        for number in range(1, 8):
            nonlinear.solve(getattr(legacy_problems, "nonlinear_problem_{}".format(number))())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--legacy-full-sweep":
        if len(argv) != 1:
            raise SystemExit("--legacy-full-sweep must be used alone")
        _legacy_full_sweep()
        return 0
    from benchmarks.run_manufactured import main as run
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
