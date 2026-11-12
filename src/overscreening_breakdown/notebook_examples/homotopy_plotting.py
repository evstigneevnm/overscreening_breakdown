"""Plots of recorded continuation decisions and independently computed fields.

These helpers use stored solver results. They never run a solve, display a
figure, or write a file. Native field evaluation precedes display conversion.
"""

from collections.abc import Mapping

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from overscreening_breakdown.notebook_examples.discontinuous_plotting import (
    _display, _draw_pieces, _experiment, _finish_axis, _guides, _segments, _title,
)


def plot_homotopy_path(experiment):
    """Show accepted states, rejected trials, and rollback in each stored plan.

    The unsuccessful initial physical attempt is a separate marker. A verified
    reference starts its own accepted path at q=1; no line joins the failed
    physical attempt to that reference. Each quadrature plan has its own path.
    """
    # Load the stored-history adapter only when called to keep imports
    # independent of the orchestration module.
    from overscreening_breakdown.notebook_examples.discontinuous_galerkin import homotopy_stage_rows

    _experiment(experiment)
    rows = homotopy_stage_rows(experiment)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    plans = tuple(dict.fromkeys(row["plan"] for row in rows))
    for index, plan in enumerate(plans):
        color = "C{}".format(index % 10)
        records = [row for row in rows if row["plan"] == plan]
        attempts, accepted_parameters = [], []
        labels = set()

        def label_once(label):
            if label in labels:
                return "_nolegend_"
            labels.add(label)
            return label

        for row in records:
            attempt, parameter = row["attempt"], _display([row["q"]])[0]
            kind, accepted = row["kind"], row["accepted"]
            if kind == "initial":
                axis.scatter(
                    [attempt], [parameter], marker="o" if accepted else "x",
                    color=color if accepted else "0.35", s=45, zorder=4,
                    label="plan {}: initial physical attempt ({})".format(
                        plan, "accepted" if accepted else "failed"),
                )
            elif kind == "reference":
                axis.scatter(
                    [attempt], [parameter], marker="D", s=45, zorder=5,
                    facecolors=color if accepted else "none",
                    edgecolors=color if accepted else "red",
                    label="plan {}: {} reference".format(plan, "verified" if accepted else "unverified"),
                )
            elif not accepted:
                axis.scatter([attempt], [parameter], marker="x", color="red", s=65, zorder=6,
                             label=label_once("plan {}: rejected trial".format(plan)))
                if row["accepted_q"] is not None:
                    previous = _display([row["accepted_q"]])[0]
                    axis.plot([attempt, attempt], [parameter, previous], color="red",
                              linestyle=":", linewidth=1.2,
                              label=label_once("plan {}: rollback to accepted state".format(plan)))
            if row["accepted_q"] is not None:
                attempts.append(attempt)
                accepted_parameters.append(row["accepted_q"])
        if attempts:
            axis.step(attempts, _display(accepted_parameters), where="post", color=color,
                      marker="o", markersize=4, label="plan {}: accepted state".format(plan))
    if not rows:
        axis.text(0.5, 0.5, "No Newton stage history was recorded.",
                  transform=axis.transAxes, ha="center")
    axis.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    axis.set_ylim(-0.08, 1.08)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_title("q=1: screened reference; q=0: physical problem")
    _finish_axis(axis, xlabel="stage attempt within each quadrature plan", ylabel="homotopy parameter q")
    figure.suptitle(_title(experiment, "accepted continuation path and rejected trials"))
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    return figure


def plot_homotopy_comparison(results, *, x_max="8", samples=60):
    """Compare u, w, and model flux for labelled experiment results.

    A failed solve is explicitly labelled as an unverified partial result and
    uses dotted curves. Coefficient jumps split every field curve. When a
    known answer is supplied, it is evaluated in its result's native context.
    """
    if not isinstance(results, Mapping) or not results:
        raise ValueError("provide a nonempty mapping of labels to experiment results")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3), squeeze=False)
    axes = tuple(axes.flat)
    interfaces, resolutions = set(), []
    reference_result = None
    for index, (label, result) in enumerate(results.items()):
        solution, arithmetic = _experiment(result)
        coefficient = solution.physical.epsilon
        segments = list(_segments(arithmetic, coefficient.interfaces, x_max, samples))
        interfaces.update(float(value) for value in coefficient.interfaces)
        if solution.N not in resolutions:
            resolutions.append(solution.N)
        if reference_result is None and getattr(result, "case", None) is not None:
            reference_result = result
        verified = getattr(solution, "converged", False)
        curve_label = "{} ({})".format(label, "verified" if verified else "unverified partial result")
        for axis, field in zip(axes, ("u", "w", "flux")):
            _draw_pieces(axis, segments, getattr(solution, field), label=curve_label,
                         color="C{}".format(index % 10), linestyle="-" if verified else ":")
    if reference_result is not None:
        solution, arithmetic = _experiment(reference_result)
        segments = list(_segments(arithmetic, solution.physical.epsilon.interfaces, x_max, samples))
        for axis, field in zip(axes, ("u", "w", "flux")):
            callback = getattr(reference_result.case, field)
            _draw_pieces(axis, segments, lambda x, f=callback: f(x, arithmetic),
                         label="known answer", color="black", linestyle="--")
    labels = ("potential u", "curvature field w", "model flux J")
    for axis, label in zip(axes, labels):
        _guides(axis, sorted(interfaces), float(x_max))
        axis.set_title(label)
        _finish_axis(axis, ylabel=label)
    resolution = ", ".join("N={} ({} unknowns)".format(N, 2*N-1) for N in resolutions)
    figure.suptitle("Continuation comparison: {}\nVerification status is shown for each calculation".format(resolution))
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    return figure
