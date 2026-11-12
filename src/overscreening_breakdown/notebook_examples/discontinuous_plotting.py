"""Figures for the guided discontinuous-material Galerkin notebook.

Numerical fields and differences are evaluated in their native arithmetic.
Only display arrays are converted to float64. Every coefficient and flux
curve is drawn one material interval at a time, with no line across a jump.
These helpers neither solve problems nor display or save the returned figures.
"""

from collections.abc import Mapping
from numbers import Integral

import numpy as np
from matplotlib import pyplot as plt

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.layered import (
    LayeredMaterial, SUPPORTED_INTERFACE_COUNTS, preset_material,
)


def _samples(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
        raise ValueError("use at least two sample points in each material layer")
    return int(value)


def _segments(arithmetic, interfaces, x_max, samples):
    samples = _samples(samples)
    right = arithmetic.scalar(x_max)
    if not arithmetic.isfinite(right) or right <= 0:
        raise ValueError("the plot limit x_max must be finite and positive")
    zero = arithmetic.scalar(0)
    interfaces = tuple(arithmetic.scalar(value) for value in interfaces)
    cuts = (zero,)+tuple(value for value in interfaces if zero < value < right)+(right,)
    for left, end in zip(cuts, cuts[1:]):
        # A left piece must not accidentally evaluate the right trace at its
        # final interface. The next piece starts at that interface instead.
        denominator = samples if end in interfaces else samples-1
        yield [left+(end-left)*arithmetic.scalar(index)/denominator
               for index in range(samples)]


def _display(values):
    display = np.array([float(value) for value in values], dtype=np.float64)
    if not np.all(np.isfinite(display)):
        raise OverflowError("plot coordinates and fields must fit finite display values")
    return display


def _positive_display(values, arithmetic):
    display = np.array([float(value) if arithmetic.isfinite(value) and value > 0 else np.nan
                        for value in values], dtype=np.float64)
    display[~np.isfinite(display) | (display <= 0)] = np.nan
    return display


def _guides(axis, interfaces, x_max):
    for interface in interfaces:
        if 0 < interface <= x_max:
            axis.axvline(float(interface), color="0.65", linestyle=":", linewidth=0.8)


def _draw_pieces(axis, segments, callback, *, label, color="C0", linestyle="-"):
    for index, points in enumerate(segments):
        axis.plot(_display(points), _display(callback(point) for point in points),
                  color=color, linestyle=linestyle,
                  label=label if index == 0 else "_nolegend_")


def _finish_axis(axis, *, xlabel="model coordinate x", ylabel=None):
    axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=8)


def _experiment(result):
    solution = result.solution
    check = getattr(solution, "check_precision", None)
    if callable(check):
        check()
    return solution, solution.arithmetic


def _domain(solution):
    a = solution.arithmetic
    physical = getattr(solution, "physical", None)
    domain = getattr(solution, "domain", getattr(physical, "domain", (0, a.inf)))
    if len(domain) != 2 or a.scalar(domain[0]) != 0:
        raise ValueError("guided plots require a domain starting at zero")
    return tuple(a.scalar(value) for value in domain)


def _plot_limit(solution, x_max=None):
    a = solution.arithmetic
    _, end = _domain(solution)
    right = (end if a.isfinite(end) else a.scalar(8)) if x_max is None else a.scalar(x_max)
    if not a.isfinite(right) or right <= 0:
        raise ValueError("the plot limit x_max must be finite and positive")
    if a.isfinite(end) and right > end:
        raise ValueError("the plot limit x_max cannot exceed the finite endpoint R")
    return right


def _same_domain(first, second):
    # Compare declarations at their common precision; field evaluation still
    # uses each solution's full native owner and coordinates independently.
    first_owner, second_owner = first.arithmetic, second.arithmetic
    first_bits = first_owner.context.prec if first_owner.use_mpmath else 53
    second_bits = second_owner.context.prec if second_owner.use_mpmath else 53
    a = first_owner if first_bits <= second_bits else second_owner
    return tuple(a.scalar(value) for value in _domain(first)) == tuple(
        a.scalar(value) for value in _domain(second))


def _scalar_label(value, arithmetic):
    return (arithmetic.context.nstr(value, n=6) if arithmetic.use_mpmath
            else format(value, ".6g"))


def _resolution_label(solution):
    a = solution.arithmetic
    _, end = _domain(solution)
    unknowns = getattr(solution, "unknown_count", None)
    count = "{} unknowns".format(unknowns) if unknowns is not None else "unknown count not recorded"
    size = "N={} ({})".format(solution.N, count)
    return "R={}, {}".format(_scalar_label(end, a), size) if a.isfinite(end) else size


def _title(result, description):
    solution = result.solution
    model = str(getattr(result, "model", "calculation")).replace("_", " ")
    status = ("equations and integration verified" if getattr(solution, "converged", False)
              else "unverified calculation")
    return "{}: {}\n{} — {}".format(model, description, _resolution_label(solution), status)


def _newton_tolerance_label(result, arithmetic):
    numerical = getattr(result, "settings", {}).get("numerical", {})
    requested = numerical.get("newton_atol", numerical.get("tolerance"))
    if requested is None:
        return "Requested Newton absolute residual tolerance was not recorded"
    value = arithmetic.scalar(requested)
    display = (arithmetic.context.nstr(value, n=6) if arithmetic.use_mpmath
               else format(value, ".6g"))
    return "Newton absolute residual tolerance = {}".format(display)


def plot_material(material, *, x_max="8", samples=60, ax=None, accuracy="standard"):
    """Preview a layer table or declared custom coefficient on a finite window.

    The shown window does not determine a solver's physical domain.
    ``high_precision`` uses an independent 50-digit context for sampling.
    """
    if accuracy not in ("standard", "strict_integration", "high_precision"):
        raise ValueError("accuracy must be standard, strict_integration, or high_precision")
    arithmetic = MpmathArithmetic(50) if accuracy == "high_precision" else NumpyArithmetic()
    description = material.coefficient if isinstance(material, LayeredMaterial) else material
    bind = getattr(description, "bind", None)
    if not callable(bind):
        raise TypeError("provide a LayeredMaterial or a declared Coefficient")
    coefficient = bind(arithmetic)
    segments = list(_segments(arithmetic, coefficient.interfaces, x_max, samples))
    if ax is None:
        figure, axis = plt.subplots(figsize=(7, 3.5))
    else:
        axis, figure = ax, ax.figure
    _draw_pieces(axis, segments, coefficient, label="epsilon(x)")
    _guides(axis, coefficient.interfaces, arithmetic.scalar(x_max))
    count = len(coefficient.interfaces)
    kind = getattr(material, "kind", "custom")
    axis.set_title("{} interface{}: {} material".format(count, "" if count == 1 else "s", kind))
    _finish_axis(axis, ylabel="coefficient epsilon")
    if ax is None:
        figure.tight_layout()
    return figure


def plot_material_gallery(kind="constant", *, x_max="8", samples=40, materials=None):
    """Preview supplied material definitions or the six half-line presets."""
    if materials is None:
        materials = {count: preset_material(count, kind=kind) for count in SUPPORTED_INTERFACE_COUNTS}
    if not isinstance(materials, Mapping) or not materials:
        raise ValueError("materials must be a nonempty mapping of interface counts to definitions")
    columns = min(3, len(materials))
    rows = (len(materials)+columns-1)//columns
    figure, axes = plt.subplots(rows, columns, figsize=(13 if columns == 3 else 7, 3.25*rows), squeeze=False)
    for (_, material), axis in zip(materials.items(), axes.flat):
        plot_material(material, x_max=x_max, samples=samples, ax=axis)
    for axis in tuple(axes.flat)[len(materials):]:
        axis.set_visible(False)
    figure.suptitle("Material presets: {} layers".format(kind))
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def plot_experiment(result, *, x_max=None, samples=60):
    """Show epsilon, potential u, independent field w, and model flux J."""
    solution, arithmetic = _experiment(result)
    x_max = _plot_limit(solution, x_max)
    coefficient = solution.physical.epsilon
    segments = list(_segments(arithmetic, coefficient.interfaces, x_max, samples))
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    axes = list(axes.flat)
    callbacks = (coefficient, solution.u, solution.w, solution.flux)
    labels = ("coefficient epsilon", "potential u", "auxiliary field w", "model flux J")
    case = getattr(result, "case", None)
    for index, (axis, callback, label) in enumerate(zip(axes, callbacks, labels)):
        _draw_pieces(axis, segments, callback, label="computed" if index else "epsilon(x)")
        if case is not None and index:
            field = ("u", "w", "flux")[index-1]
            reference = getattr(case, field)
            _draw_pieces(axis, segments, lambda x, f=reference: f(x, arithmetic),
                         label="known answer", color="black", linestyle="--")
        _guides(axis, coefficient.interfaces, arithmetic.scalar(x_max))
        axis.set_title(label)
        _finish_axis(axis, ylabel=label)
    figure.suptitle(_title(result, "material and fields"))
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def plot_errors(result, *, x_max=None, samples=60):
    """Show native-arithmetic absolute errors against a manufactured case."""
    solution, arithmetic = _experiment(result)
    x_max = _plot_limit(solution, x_max)
    case = getattr(result, "case", None)
    if case is None:
        raise ValueError("a known-answer manufactured case is needed for an error plot")
    interfaces = solution.physical.epsilon.interfaces
    segments = list(_segments(arithmetic, interfaces, x_max, samples))
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.7), squeeze=False)
    for axis, field in zip(axes.flat, ("u", "w", "flux")):
        numerical, reference = getattr(solution, field), getattr(case, field)
        plotted = []
        native = []
        for index, points in enumerate(segments):
            # Both values are converted into this owner before subtraction;
            # float conversion occurs only after the small error is formed.
            errors = [abs(arithmetic.scalar(numerical(x))-arithmetic.scalar(reference(x, arithmetic)))
                      for x in points]
            native.extend(errors)
            display = _positive_display(errors, arithmetic)
            plotted.extend(display)
            axis.plot(_display(points), display, color="C0",
                      label="absolute error" if index == 0 else "_nolegend_")
        if not any(np.isfinite(value) for value in plotted):
            axis.set_ylim(0.1, 10)
            message = ("All sampled errors are exactly zero." if all(value == 0 for value in native)
                       else "Errors outside the log display range are omitted.")
            axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", fontsize=8)
        axis.set_yscale("log")
        _guides(axis, interfaces, arithmetic.scalar(x_max))
        axis.set_title("{} error".format("model flux" if field == "flux" else field))
        _finish_axis(axis, ylabel="absolute error")
    figure.suptitle(_title(result, "errors against the known answer"))
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return figure


def _domain_comparison_signature(result):
    solution, a = _experiment(result)
    settings = getattr(result, "settings", {})
    if not all(name in settings for name in ("material", "physical", "numerical")):
        raise ValueError("R-only sensitivity requires recorded fixed physical and numerical settings")
    return dict(
        model=getattr(result, "model", None), material=settings["material"],
        physical={key: value for key, value in settings["physical"].items() if key not in ("domain", "R")},
        numerical={key: value for key, value in settings["numerical"].items() if key != "newton_atol"},
        arithmetic=(a.use_mpmath, a.dps, a.backend), N=solution.N,
    )


def plot_domain_comparison(results, *, samples=60):
    """Compare finite domains at fixed physics, arithmetic, N and solver targets.

    Each field is shown only on its own interval. These curves measure domain
    sensitivity; spatial accuracy is assessed in a separate resolution study.
    Metadata checks cannot certify equality of arbitrary executable callbacks.
    """
    if not isinstance(results, Mapping) or not results:
        raise ValueError("provide a nonempty mapping from R values to experiment results")
    prepared, signature = [], None
    for endpoint, result in results.items():
        solution, a = _experiment(result)
        _, end = _domain(solution)
        if not a.isfinite(end) or a.scalar(endpoint) != end:
            raise ValueError("domain comparison keys must match each finite endpoint R")
        current = _domain_comparison_signature(result)
        if signature is not None and current != signature:
            raise ValueError("R-only sensitivity requires fixed physics, material, arithmetic, N and numerical settings")
        signature = current
        segments = tuple(_segments(a, solution.physical.epsilon.interfaces, end, samples))
        prepared.append((result, solution, a, end, segments))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    interfaces = set()
    for index, (result, solution, a, end, segments) in enumerate(prepared):
        verified = solution.converged
        label = "{}; {}".format(_resolution_label(solution), "verified" if verified else "unverified partial result")
        interfaces.update(float(point) for point in solution.physical.epsilon.interfaces)
        for axis, field in zip(axes.flat, ("u", "w", "flux")):
            _draw_pieces(axis, segments, getattr(solution, field), label=label,
                         color="C{}".format(index % 10), linestyle="-" if verified else ":")
    maximum = max(float(item[3]) for item in prepared)
    for axis, label in zip(axes.flat, ("potential u", "curvature field w", "model flux J")):
        _guides(axis, sorted(interfaces), maximum)
        axis.set_title(label)
        _finish_axis(axis, ylabel=label)
    first = prepared[0][1]
    figure.suptitle("Domain sensitivity at fixed N={} ({} unknowns)\nR={}; status is shown for each calculation".format(
        first.N, first.unknown_count, ", ".join(_scalar_label(item[3], item[2]) for item in prepared)))
    figure.tight_layout(rect=(0, 0, 1, 0.85))
    return figure


def plot_convergence(result):
    """Separate fixed-equation Newton histories from quadrature verification.

    Newton segments break at every homotopy parameter and quadrature-plan
    change. Ratios below the horizontal line at one meet the recorded target.
    """
    solution, arithmetic = _experiment(result)
    history = getattr(solution, "quadrature_history", ())
    plans = [item.newton_result for item in history if item.newton_result is not None]
    if not plans and getattr(solution, "newton_result", None) is not None:
        plans = [solution.newton_result]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    newton_axis, quadrature_axis = axes[0]
    plan_offset = 0
    newton_values = []
    for plan_index, plan in enumerate(plans):
        stage_offset = 0
        color = "C{}".format(plan_index % 10)
        for stage_index, stage in enumerate(plan.stages, 1):
            target = arithmetic.scalar(stage.target_norm)
            if not arithmetic.isfinite(target) or target <= 0:
                raise ValueError("Newton convergence plots require positive recorded targets")
            records = [record for record in plan.history if record.stage == stage_index]
            attempts = [plan_offset+stage_offset]
            norms = [arithmetic.scalar(stage.initial_norm)/target]
            attempts.extend(plan_offset+record.iteration for record in records)
            norms.extend(arithmetic.scalar(record.residual_norm)/target for record in records)
            end = plan_offset+stage_offset+stage.iterations
            if attempts[-1] != end:
                attempts.append(end)
                norms.append(arithmetic.scalar(stage.final_norm)/target)
            newton_values.extend(norms)
            newton_axis.plot(
                attempts, _positive_display(norms, arithmetic), color=color,
                linestyle="-" if stage.homotopy == 0 else "--", marker="o", markersize=3,
                label="plan {}, q={:.3g}".format(plan_index+1, float(stage.homotopy)))
            if not stage.converged:
                newton_axis.scatter([end], _positive_display([norms[-1]], arithmetic),
                                    color=color, marker="v", s=25)
            stage_offset += stage.iterations
        reference = getattr(plan, "reference_result", None)
        if reference is not None and reference.target_norm > 0:
            initial = plan.stages[0].iterations if plan.stages else 0
            ratio = arithmetic.scalar(reference.residual_norm)/arithmetic.scalar(reference.target_norm)
            newton_axis.scatter([plan_offset+initial], _positive_display([ratio], arithmetic),
                                color=color, marker="D", s=25,
                                label="plan {} reference".format(plan_index+1))
        plan_offset += plan.iterations
    if not plans:
        newton_axis.text(0.5, 0.5, "No Newton history was recorded.",
                         transform=newton_axis.transAxes, ha="center")
    elif not np.any(np.isfinite(_positive_display(newton_values, arithmetic))):
        message = ("All recorded residuals are exactly zero."
                   if all(value == 0 for value in newton_values)
                   else "Nonfinite or out-of-range residuals are omitted.")
        newton_axis.text(0.5, 0.5, message, transform=newton_axis.transAxes,
                         ha="center", fontsize=8)
    newton_axis.axhline(1, color="0.4", linestyle=":", label="acceptance target")
    newton_axis.set_yscale("log")
    newton_axis.set_title("Equation solve, all quadrature plans (Euclidean residual)\n"
                          +_newton_tolerance_label(result, arithmetic))
    _finish_axis(newton_axis, xlabel="attempted Newton iteration", ylabel="residual / stage target")

    quadrature_values = []
    for field, label, marker in (("residual", "residual integration check", "o"),
                                 ("jacobian", "derivative integration check", "s")):
        orders, ratios = [], []
        for item in history:
            difference = getattr(item, field+"_difference", None)
            target = getattr(item, field+"_target", None)
            if difference is None or target is None or target <= 0:
                continue
            orders.append(item.order)
            ratios.append(arithmetic.scalar(difference)/arithmetic.scalar(target))
        if orders:
            quadrature_values.extend(ratios)
            quadrature_axis.plot(orders, _positive_display(ratios, arithmetic),
                                 marker=marker, label=label)
    if not quadrature_values:
        quadrature_axis.text(0.5, 0.5, "No integration verification was recorded.",
                            transform=quadrature_axis.transAxes, ha="center")
    elif not np.any(np.isfinite(_positive_display(quadrature_values, arithmetic))):
        message = ("Recorded integration changes are exactly zero."
                   if all(value == 0 for value in quadrature_values)
                   else "Nonfinite or out-of-range changes are omitted.")
        quadrature_axis.text(0.5, 0.5, message, transform=quadrature_axis.transAxes,
                            ha="center", fontsize=8)
    quadrature_axis.axhline(1, color="0.4", linestyle=":", label="verification target")
    quadrature_axis.set_yscale("log")
    quadrature_axis.set_title("Independent numerical integration check")
    _finish_axis(quadrature_axis, xlabel="quadrature points per panel", ylabel="change / target")
    figure.suptitle(_title(result, "numerical checks; values below one pass"))
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return figure
