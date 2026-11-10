"""Presentation and command-line support shared by the bounded examples."""

import argparse
from pathlib import Path


def tolerance(use_mpmath, precision):
    return "1e-{}".format(max(10, precision-15)) if use_mpmath else "1e-11"


def sampled_error(solution, case):
    arithmetic = solution.arithmetic
    errors = []
    for text in ("0", "0.125", "1", "4", "10"):
        x = arithmetic.scalar(text)
        errors.extend((abs(solution.u(x)-case.u(x, arithmetic)),
                       abs(solution.w(x)-case.w(x, arithmetic))))
    return max(errors)


def report(solution, error=None):
    print("{}: nonlinear_converged={}, quadrature_converged={}, {} unknowns{}".format(
        solution.solver_type, solution.nonlinear_converged,
        solution.quadrature_converged, solution.unknown_count,
        "" if error is None else ", max sampled field error={}".format(error)))


def write_outputs(solution, directory, prefix):
    from overscreening_breakdown import plot_coefficients, plot_fields
    from matplotlib import pyplot as plt

    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    solution.save(directory/(prefix+".json"))
    for name, plotter in (("fields", plot_fields), ("coefficients", plot_coefficients)):
        figure = plotter(solution)
        figure.savefig(directory/(prefix+"-"+name+".pdf"))
        plt.close(figure)


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--backend", choices=("numpy", "mpmath", "both"), default="both")
    result.add_argument("--precision", type=int, default=50)
    result.add_argument("--output-dir", type=Path)
    return result


def backend_choices(name):
    return (False, True) if name == "both" else (name == "mpmath",)
