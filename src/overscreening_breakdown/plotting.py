"""Plot common solution fields without changing their numerical arithmetic.

Field evaluations and sampling coordinates retain the solution's arithmetic.
Conversion to float64 occurs only when supplying display data to Matplotlib.
The caller owns displaying, saving and closing every returned Figure.
"""

import numpy as np
from matplotlib import pyplot as plt


def _axes(ax, count):
    if ax is None:
        figure, axes = plt.subplots(count, 1, figsize=(7, 2.7*count), squeeze=False)
        return figure, axes[:, 0]
    axes = np.asarray(ax, dtype=object).reshape(-1)
    if len(axes) != count:
        raise ValueError("expected {} plotting axes".format(count))
    figure = axes[0].figure
    if any(axis.figure is not figure for axis in axes):
        raise ValueError("all plotting axes must belong to the same figure")
    return figure, axes


def _display_values(values):
    result = np.array([float(value) for value in values], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise OverflowError("plot coordinates and values must fit finite float64 display data")
    return result


def _title(solution):
    return "{}, N={}, {} unknowns".format(
        solution.solver_type.capitalize(), solution.N, solution.unknown_count)


def _record_solution(figure, solution):
    owners = getattr(figure, "_overscreening_solutions", ())
    if all(owner is not solution for owner in owners):
        figure._overscreening_solutions = tuple(owners)+(solution,)
    return figure


def plot_fields(solution, *, x=None, ax=None):
    """Return three physical-coordinate panels for u, w and physical flux J.

    ``x`` is a one-dimensional sequence of physical sample points.  By default,
    use 201 native-arithmetic points on the finite domain, or [0,10*L] for a
    half-line problem.  ``ax`` may supply three axes from one Figure.
    """
    solution.check_precision()
    arithmetic = solution.arithmetic
    if x is None:
        left, right = solution.domain
        if right == arithmetic.inf:
            right = 10*solution.L
        points = [left+(right-left)*arithmetic.scalar(index)/200 for index in range(201)]
    else:
        inputs = np.asarray(x, dtype=object)
        if inputs.ndim != 1 or not inputs.size:
            raise ValueError("physical plot samples must be a nonempty one-dimensional sequence")
        points = [arithmetic.scalar(value) for value in inputs]
    native = np.array(points, dtype=object if arithmetic.use_mpmath else np.float64)
    fields = (("u", solution.u(native)), ("w", solution.w(native)),
              ("physical flux J", solution.flux(native)))
    display_x = _display_values(points)
    figure, axes = _axes(ax, 3)
    for axis, (label, values) in zip(axes, fields):
        axis.plot(display_x, _display_values(values), label=label)
        axis.set_xlabel("physical coordinate x")
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(True, alpha=0.25)
    figure.suptitle(_title(solution))
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return _record_solution(figure, solution)


def plot_coefficients(solution, *, ax=None):
    """Return separate Galerkin coefficient panels, or one collocation panel."""
    solution.check_precision()
    if solution.solver_type == "galerkin":
        lift = (r"$u-\ell$ coefficients" if getattr(solution.basis.mapping, "mapping_type", None) == "affine_interval"
                else r"$u-\ell_L$ coefficients")
        fields = ((lift, solution.u_coefficients),
                  (r"$w$ coefficients", solution.w_coefficients))
    elif solution.solver_type == "collocation":
        fields = ((r"$u$ coefficients", solution.coefficients),)
    else:
        raise ValueError("coefficient plots support collocation and galerkin solutions")
    figure, axes = _axes(ax, len(fields))
    for axis, (label, coefficients) in zip(axes, fields):
        axis.plot(np.arange(len(coefficients)), _display_values(coefficients),
                  marker="o", label=label)
        axis.set_xlabel("coefficient index")
        axis.set_ylabel("coefficient value")
        axis.legend()
        axis.grid(True, alpha=0.25)
    figure.suptitle(_title(solution))
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return _record_solution(figure, solution)
