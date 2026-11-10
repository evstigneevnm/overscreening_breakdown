"""Common solve, explicit conversion, exact snapshot and optional plots.

Run ``python -B -m examples.common_solver_example`` from the repository root.
Add ``--output-dir /tmp/common-solver-example`` to retain the JSON and plots.
The example uses the same physical epsilon=1 problem for both solvers.
"""

import argparse
from pathlib import Path
import tempfile

from examples.flux_problem_collocation import make_problem
from solvers import load_solution, project_collocation, solve


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    problem = make_problem()
    options = dict(use_mpmath=True, precision=50, tolerance="1e-35")
    collocation = solve(problem, 8, **options)
    mixed = solve(problem, 4, solver="galerkin", **options)
    if not collocation.converged or not mixed.converged:
        raise RuntimeError("the comparison problem did not converge")
    # Trace normalization is explicit and recorded in the projected snapshot.
    # The tolerance here permits roundoff amplified by the collocation system.
    restart = project_collocation(collocation, N=4, trace_tolerance="1e-35")
    resumed = solve(problem, 4, solver="galerkin", initial_guess=restart, **options)
    if not resumed.converged:
        raise RuntimeError("the projected mixed restart did not converge")
    path = directory/"galerkin.json"
    resumed.save(path)
    restored = load_solution(path, problem=problem, solver="galerkin")
    norm = restored.h1_norm(field="u", plan=restored.plan)
    print("collocation: {} unknowns; Galerkin: {} unknowns".format(
        collocation.unknown_count, restored.unknown_count))
    print("restored u(1)={}, w(1)={}, physical H1(u)={}".format(
        restored.u(1), restored.w(1), norm))
    return collocation, restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="overscreening-common-", dir="/tmp") as directory:
            run(directory)
    else:
        _, result = run(args.output_dir)
        from matplotlib import pyplot as plt
        from solution_plotting import plot_coefficients, plot_fields
        for name, plotter in (("fields", plot_fields), ("coefficients", plot_coefficients)):
            figure = plotter(result)
            figure.savefig(args.output_dir/(name+".pdf"))
            plt.close(figure)


if __name__ == "__main__":
    main()
