"""Guided layered-material experiments, without notebook-specific dependencies.

Physical inputs remain separate from numerical presets. The helpers deliberately
distinguish algebraic convergence, quadrature verification, and spatial checks.
"""

import csv
from dataclasses import dataclass, field, fields, is_dataclass, replace
from decimal import Decimal
from importlib.metadata import version
import json
import re
from numbers import Integral
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from overscreening_breakdown import BSKProblem, solve
from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient, ConstantCoefficient
from overscreening_breakdown.diagnostics import ProgressReporter
from overscreening_breakdown.layered import LayeredMaterial, SUPPORTED_INTERFACE_COUNTS, preset_material
from overscreening_breakdown.manufactured.transmission import layered_transmission_case
from overscreening_breakdown.norms import physical_l2_norm
from overscreening_breakdown.quadrature import QuadratureFactory


ACCURACY_PRESETS = MappingProxyType({
    "standard": MappingProxyType(dict(
        L="1", use_mpmath=False, precision=50,
        quadrature_order=None, quadrature_atol="1e-8", quadrature_rtol="1e-8",
        max_refinements=3)),
    "strict_integration": MappingProxyType(dict(
        L="1", use_mpmath=False, precision=50,
        quadrature_order=None, quadrature_atol="1e-9", quadrature_rtol="1e-9",
        max_refinements=3)),
    "high_precision": MappingProxyType(dict(
        L="1", use_mpmath=True, precision=50,
        quadrature_order=None, quadrature_atol="1e-12", quadrature_rtol="1e-12",
        max_refinements=3)),
})
MODELS = ("BSK", "manufactured_linear", "manufactured_nonlinear")


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_value(value, arithmetic=None):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, arithmetic) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item, arithmetic) for item in value]
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name), arithmetic) for item in fields(value)}
    if hasattr(value, "_mpf_"):
        owner = arithmetic.context if arithmetic is not None and arithmetic.use_mpmath else value.context
        return owner.nstr(value, n=owner.dps+3)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return _json_value(value.item(), arithmetic)
    raise TypeError("experiment settings contain an unsupported value")


def _coefficient(material):
    if isinstance(material, LayeredMaterial):
        return material.coefficient
    if isinstance(material, (Coefficient, ConstantCoefficient)):
        return material
    raise ValueError("choose a layer table or an explicitly bounded Coefficient")


def _material_settings(material):
    if isinstance(material, LayeredMaterial):
        return dict(type="layers", reconstructible=True, **material.to_dict())
    if isinstance(material, ConstantCoefficient):
        return dict(type="constant", reconstructible=True, value=str(material.value))
    return dict(type="custom_function", reconstructible=False,
                interfaces=[str(value) for value in material.interfaces],
                lower_bound=str(material.lower_bound), upper_bound=str(material.upper_bound),
                wall_value=str(material.wall_value),
                note="Supply the original coefficient function again when loading the solution.")


@dataclass(frozen=True)
class ExperimentResult:
    model: str
    accuracy: str
    material: object
    solution: object
    case: object
    settings: object
    elapsed_seconds: float
    _options: object = field(repr=False)
    problem: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ResolutionComparison:
    coarse: ExperimentResult
    fine: ExperimentResult
    relative_changes: object


def _resolution(value, name="N"):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
        raise ValueError("{} must be an integer of at least two".format(name))
    return int(value)


def _numerical_options(accuracy, numerical):
    if accuracy not in ACCURACY_PRESETS:
        raise ValueError("accuracy must be standard, strict_integration, or high_precision")
    options = dict(ACCURACY_PRESETS[accuracy])
    if numerical is not None:
        if not isinstance(numerical, Mapping):
            raise ValueError("advanced numerical settings must be a dictionary")
        if "N" in numerical:
            raise ValueError("provide N explicitly, outside the numerical dictionary")
        if "tolerance" in numerical or "newton_atol" in numerical:
            raise ValueError("provide newton_atol explicitly, outside the numerical dictionary")
        options.update(numerical)
    if "solver" in options or "verbosity" in options or "problem" in options:
        raise ValueError("solver and verbosity are selected by the experiment interface")
    return options


def _newton_tolerance(value, options):
    """Validate the threshold in the selected arithmetic without float narrowing."""
    arithmetic = (MpmathArithmetic(options["precision"]) if options["use_mpmath"]
                  else NumpyArithmetic())
    message = "newton_atol must be finite and strictly positive in the selected arithmetic"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            native = arithmetic.scalar(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not arithmetic.isfinite(native) or native <= 0:
        raise ValueError(message)
    requested = str(value) if isinstance(value, (Decimal, np.generic)) else _json_value(value, arithmetic)
    return native, requested


def _check_material(material, actual, domain=None):
    """Check a supplied material declaration against the physical coefficient."""
    a = actual.arithmetic
    expected = _coefficient(material).bind(a)
    if (expected.interfaces != actual.interfaces or any(
        getattr(expected, name) != getattr(actual, name)
        for name in ("lower_bound", "upper_bound", "wall_value")
    )):
        raise ValueError("material declaration does not match the supplied problem epsilon")
    left, right = ((a.scalar(0), a.inf) if domain is None
                   else tuple(a.scalar(value) for value in domain))
    if any(not left < point < right for point in actual.interfaces):
        raise ValueError("material interfaces must lie strictly inside the supplied problem domain")
    one = a.scalar(1)
    starts = (left,)+actual.interfaces
    samples = set(starts)
    if a.isfinite(right):
        boundaries = starts+(right,)
        samples.add(right)
        samples.update((start+end)/2 for start, end in zip(boundaries, boundaries[1:]))
    else:
        samples.update((start+end)/2 for start, end in zip(starts, starts[1:]))
        tail = starts[-1]
        samples.update((tail+one, tail+2*one, tail+8*one))
    for x in samples:
        if expected(x) != actual(x):
            raise ValueError("material values do not match the supplied problem epsilon")


def _solution_domain(solution):
    a = solution.arithmetic
    domain = getattr(solution, "domain", getattr(solution.physical, "domain", (0, a.inf)))
    return tuple(a.scalar(value) for value in domain)


def _field_plan(solution, order):
    """Build a physical rule in the solution's actual geometry."""
    a = solution.arithmetic
    domain = _solution_domain(solution)
    geometry = (dict(domain=domain) if a.isfinite(domain[1]) else dict(L=solution.L))
    return QuadratureFactory(a, interfaces=solution.physical.epsilon.interfaces,
                             **geometry).plan(order, solution.plan.subdivisions)


def _defined_physical_settings(problem, solution):
    """Record the actual supplied definition without serializing callbacks."""
    physical, a = solution.physical, solution.arithmetic
    left = physical.boundary_conditions[0]
    values = dict(delta=physical.delta,
                  domain=tuple(value if a.isfinite(value) else str(value) for value in physical.domain),
                  boundary_conditions=physical.boundary_conditions,
                  wall_potential=left[0], wall_third_derivative=left[3])
    if isinstance(problem, BSKProblem):
        values.update((name, a.scalar(getattr(problem, name))) for name in ("sigma", "mu", "gamma"))
    epsilon = physical.epsilon
    definition = dict(
        description_type="{}.{}".format(type(problem).__module__, type(problem).__qualname__),
        physical=values,
        epsilon=dict(lower_bound=epsilon.lower_bound, upper_bound=epsilon.upper_bound,
                     wall_value=epsilon.wall_value, interfaces=epsilon.interfaces),
        callbacks_serialized=False,
        note="Reuse the original problem definition when loading or refining; model is a display label.",
    )
    return values, definition


def run_defined_problem(problem, *, N, newton_atol, material, case=None, model="custom",
                        accuracy="standard", verbosity=0, numerical=None):
    """Solve the caller's physical definition and retain it for later refinement.

    The problem supplies its source, derivative, coefficient, boundary values,
    and initial guess. ``model`` labels the experiment and never selects or
    reconstructs those inputs. A supplied material must agree with its epsilon
    declarations and representative samples; this is not a proof of callback
    equality everywhere. ``case`` optionally supplies known reference fields.
    """
    N = _resolution(N)
    options = _numerical_options(accuracy, numerical)
    options["tolerance"], requested_atol = _newton_tolerance(newton_atol, options)
    if options["quadrature_order"] is None:
        options["quadrature_order"] = 2*N+4
    _coefficient(material)
    a = MpmathArithmetic(options["precision"]) if options["use_mpmath"] else NumpyArithmetic()
    raw_domain = getattr(problem, "domain", (0, "inf"))
    domain = tuple(a.scalar(value) for value in raw_domain)
    finite_domain = a.isfinite(domain[1])
    if finite_domain:
        if numerical is not None and "L" in numerical:
            raise ValueError("finite Galerkin geometry is set by the problem domain; L is a half-line parameter")
        options.pop("L", None)
    description = getattr(problem, "epsilon", None)
    if callable(getattr(description, "bind", None)):
        _check_material(material, description.bind(a), domain)
    reporter = ProgressReporter(verbosity=verbosity)
    if reporter.verbosity:
        arithmetic = ("mpmath ({} decimal digits)".format(options["precision"])
                      if options["use_mpmath"] else
                      "NumPy float64 ({} decimal digits)".format(NumpyArithmetic.dps))
        print("{} | Galerkin N={}, {} unknowns | {} | newton_atol={} (absolute Euclidean residual).".format(
            model, N, 2*N if finite_domain else 2*N-1, arithmetic, requested_atol), flush=True)
    started = perf_counter()
    solution = solve(problem, N, solver="galerkin", verbosity=verbosity, **options)
    elapsed = perf_counter()-started
    _check_material(material, solution.physical.epsilon, solution.physical.domain)
    physical, definition = _defined_physical_settings(problem, solution)
    settings = dict(model=model, accuracy=accuracy, material=_material_settings(material),
                    physical=physical, definition=definition,
                    numerical=dict(N=N, newton_atol=requested_atol, **{
                        key: value for key, value in options.items()
                        if key not in ("initial_guess", "progress_callback")}),
                    warm_start=options.get("initial_guess") is not None,
                    progress_observer_attached=options.get("progress_callback") is not None)
    return ExperimentResult(model, accuracy, material, solution, case,
                            _freeze(_json_value(settings, solution.arithmetic)), elapsed,
                            MappingProxyType(dict(N=N, **options)), problem=problem)


def run_experiment(material, *, N, newton_atol, model="BSK", accuracy="standard", wall_potential="0.05",
                   delta="1", gamma="0.5", sigma="2", mu="0.1", verbosity=0,
                   numerical=None):
    """Construct a catalog problem, then delegate to run_defined_problem.

    This convenience interface fixes the catalog's boundary and initial data.
    Use run_defined_problem with an explicit BSKProblem or FluxProblem when
    defining the physical source, boundary values, or initial guess yourself.
    """
    if model not in MODELS:
        raise ValueError("model must be BSK, manufactured_linear, or manufactured_nonlinear")
    coefficient = _coefficient(material)
    if model == "BSK":
        problem = BSKProblem(
            delta=delta, gamma=gamma, sigma=sigma, mu=mu, epsilon=coefficient,
            boundary_conditions=((wall_potential, None, None, 0, None),
                                 (0, None, None, None, None)),
        )
        case = None
        physical = dict(delta=str(delta), gamma=str(gamma), sigma=str(sigma), mu=str(mu),
                        wall_potential=str(wall_potential), wall_third_derivative="0")
    else:
        if not isinstance(material, LayeredMaterial):
            raise ValueError("known-answer examples need a layer table; custom functions can use BSK")
        case = layered_transmission_case(material, nonlinear=model == "manufactured_nonlinear", delta=delta)
        problem = case.make_problem()
        physical = dict(delta=str(delta), wall_potential="1", wall_third_derivative="-6",
                        reaction_beta="0.125" if model == "manufactured_nonlinear" else "0")
    result = run_defined_problem(
        problem, N=N, newton_atol=newton_atol, material=material, case=case, model=model,
        accuracy=accuracy, verbosity=verbosity, numerical=numerical,
    )
    return replace(result, settings=_freeze({**result.settings, "physical": physical}))


def _number(value, arithmetic, digits=5):
    if arithmetic.use_mpmath:
        return arithmetic.context.nstr(value, n=digits)
    return format(float(value), ".{}g".format(digits))


def report_experiment(result, *, resolution=None):
    """Return human-readable status without claiming spatial accuracy."""
    solution, a = result.solution, result.solution.arithmetic
    history = [item.newton_result for item in solution.quadrature_history]
    count = len(solution.physical.epsilon.interfaces)
    lines = ["{} | {} interfaces | {}".format(result.model, count, result.accuracy),
             "Equations solved: {}".format("yes" if solution.nonlinear_converged else "NO"),
             "Newton absolute residual tolerance: {} (Euclidean).".format(
                 result.settings["numerical"]["newton_atol"]),
             "Numerical integration verified: {}".format("yes" if solution.quadrature_converged else "NO"),
             "Resolution sensitivity: {}".format(
                 "not checked (solve and integration checks do not measure spatial error)"
                 if resolution is None else "compared N={} and N={} (not an error bound)".format(
                     resolution.coarse.solution.N, resolution.fine.solution.N)),
             "Basis size N={}, {} unknowns; {} arithmetic{}.".format(
                 solution.N, solution.unknown_count, "mpmath" if a.use_mpmath else "NumPy",
                 " ({} digits)".format(a.dps) if a.use_mpmath else ""),
             "Newton attempts: {}; reference used: {}; elapsed: {:.2f} s.".format(
                 sum(item.iterations for item in history),
                 "yes" if any(item.reference_result is not None for item in history) else "no",
                 result.elapsed_seconds)]
    domain = _solution_domain(solution)
    if a.isfinite(domain[1]):
        lines.insert(1, "Physical domain: [{}, {}].".format(_number(domain[0], a, 7), _number(domain[1], a, 7)))
    final = solution.newton_result
    if final.homotopy == 0:
        lines.append("Final physical residual norm: {} (Euclidean).".format(
            _number(final.residual_norm, a, 7)))
    else:
        lines.append("Final Newton residual norm at q={}: {} (Euclidean); physical endpoint not reached.".format(
            _number(final.homotopy, a, 7), _number(final.residual_norm, a, 7)))
    if not solution.converged:
        reason = solution.newton_result.reason if not solution.nonlinear_converged else "quadrature_not_converged"
        lines.append("Result is not verified: {}. Inspect the diagnostics before using these fields.".format(reason))
    if resolution is not None:
        lines.append("Comparison Newton absolute tolerances: coarse={}, fine={}.".format(
            resolution.coarse.settings["numerical"]["newton_atol"],
            resolution.fine.settings["numerical"]["newton_atol"]))
        lines.append("Relative field changes: "+", ".join(
            "{}={}".format(name, _number(value, a)) for name, value in resolution.relative_changes.items()))
        if not resolution.fine.solution.converged:
            lines.append("The finer solve is not verified; the comparison is provisional.")
    return "\n".join(lines)


def material_table(material, *, accuracy="standard"):
    """Return the region table, including both traces at each interface."""
    from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
    if accuracy not in ACCURACY_PRESETS:
        raise ValueError("accuracy must be standard, strict_integration, or high_precision")
    a = MpmathArithmetic(50) if accuracy == "high_precision" else NumpyArithmetic()
    bound = _coefficient(material).bind(a)
    if not isinstance(material, LayeredMaterial):
        return "Custom coefficient | interfaces: {} | bounds: [{}, {}]".format(
            ", ".join(str(value) for value in bound.interfaces), bound.lower_bound, bound.upper_bound)
    layers = material.bind(a)
    lines = ["Layer | starts at x | ends at x | epsilon at layer start"]
    for index, (start, value) in enumerate(material.rows):
        end = material.rows[index+1][0] if index+1 < len(material.rows) else "infinity"
        lines.append("{} | {} | {} | {}".format(index+1, start, end, value))
    lines.append("Interface | epsilon on left | epsilon on right")
    for index, point in enumerate(layers.interfaces):
        lines.append("{} | {} | {}".format(_number(point, a), _number(layers.trace(index, "left"), a),
                                           _number(layers.trace(index, "right"), a)))
    return "\n".join(lines)


def manufactured_errors(result, *, order=None):
    """Estimate physical L2 errors using an independently refined split rule."""
    if result.case is None:
        raise ValueError("the BSK model has no supplied exact solution")
    solution, a = result.solution, result.solution.arithmetic
    plan = _field_plan(solution, order or max(2*solution.plan.order, 2*solution.N+8))
    return {name: solution.l2_error(lambda x, field=name: getattr(result.case, field)(x, a),
                                    field=name, plan=plan) for name in ("u", "w", "flux")}


def report_verification(result):
    """Summarize independently integrated errors in a known-answer experiment."""
    errors = manufactured_errors(result)
    a = result.solution.arithmetic
    return ("Known-answer check: estimated physical L2 errors\n"
            +"\n".join("  {}: {}".format(name, _number(value, a, 7)) for name, value in errors.items())
            +"\nThese measure spatial and arithmetic error together; the integration is numerical.")


def manufactured_error_metrics(result, *, order=None):
    """Estimate absolute and relative physical L2 errors in native arithmetic.

    Each field's error and reference norm use the same independent positive
    rule, split at every material interface. ``relative`` is a ratio, not a
    percentage, and is None when the computed reference norm is zero.
    """
    if result.case is None:
        raise ValueError("the BSK model has no supplied exact solution")
    solution, a = result.solution, result.solution.arithmetic
    selected_order = max(2*solution.plan.order, 2*solution.N+8) if order is None else order
    plan = _field_plan(solution, selected_order)
    metrics = {}
    for field_name in ("u", "w", "flux"):
        def reference(x, field=field_name):
            return getattr(result.case, field)(x, a)

        absolute = solution.l2_error(reference, field=field_name, plan=plan)
        reference_norm = physical_l2_norm(reference, plan)
        relative = None if reference_norm == 0 else absolute/reference_norm
        metrics[field_name] = dict(absolute=absolute, reference_norm=reference_norm, relative=relative)
    return _freeze(metrics)


def _relative_percent(value, arithmetic):
    return ("undefined (zero reference norm)" if value is None
            else _number(100*value, arithmetic, 7))


def report_error_table(result, *, order=None):
    """Return absolute and percentage errors for a known-answer experiment."""
    metrics = manufactured_error_metrics(result, order=order)
    a = result.solution.arithmetic
    lines = ["Known-answer errors in the physical L2(dx) norm",
             "Field | Absolute L2 | Relative L2 (%)"]
    for field_name, values in metrics.items():
        lines.append("{} | {} | {}".format(
            field_name, _number(values["absolute"], a, 7),
            _relative_percent(values["relative"], a)))
    if not getattr(result.solution, "converged", False):
        lines.append("PROVISIONAL: the equations or numerical integration checks did not converge.")
    lines.append("Relative error is 100 times error/reference norm; zero reference norms are undefined.")
    lines.append("Errors use numerical integration and include spatial and arithmetic error together.")
    return "\n".join(lines)


def run_manufactured_convergence(material, *, N_values, newton_atol, model,
                                 accuracy="standard", delta="1", verbosity=0, numerical=None):
    """Solve one manufactured problem at explicitly selected increasing N values.

    All sizes and shared numerical declarations are checked before the first
    solve. Each solve starts independently; no previous result is used as a
    restart. Newton and integration targets remain fixed across the sequence.
    """
    if model not in ("manufactured_linear", "manufactured_nonlinear"):
        raise ValueError("convergence checks require manufactured_linear or manufactured_nonlinear")
    if isinstance(N_values, (str, bytes)):
        raise ValueError("N_values must be a nonempty sequence of increasing integers")
    try:
        resolutions = tuple(_resolution(value, "each N value") for value in N_values)
    except TypeError as error:
        raise ValueError("N_values must be a nonempty sequence of increasing integers") from error
    if not resolutions or any(left >= right for left, right in zip(resolutions, resolutions[1:])):
        raise ValueError("N_values must be nonempty and strictly increasing")
    if not isinstance(material, LayeredMaterial):
        raise ValueError("known-answer convergence checks need a layer table")
    options = _numerical_options(accuracy, numerical)
    _newton_tolerance(newton_atol, options)
    explicit_order = options["quadrature_order"]
    if explicit_order is not None and (
        isinstance(explicit_order, bool) or not isinstance(explicit_order, Integral)
        or explicit_order < resolutions[-1]+1
    ):
        raise ValueError("explicit quadrature_order must be an integer of at least max(N_values)+1")
    ProgressReporter(verbosity=verbosity)
    return tuple(run_experiment(
        material, N=N, newton_atol=newton_atol, model=model, accuracy=accuracy,
        delta=delta, verbosity=verbosity, numerical=numerical,
    ) for N in resolutions)


def _convergence_signature(result):
    if result.case is None:
        raise ValueError("convergence tables require known-answer manufactured results")
    a = result.solution.arithmetic
    check = getattr(result.solution, "check_precision", None)
    if callable(check):
        check()
    domain = _solution_domain(result.solution)
    geometry = (("affine_interval", domain) if a.isfinite(domain[1])
                else ("rational_half_line", result.solution.L))
    return dict(
        model=result.model, material=result.settings["material"], physical=result.settings["physical"],
        arithmetic=(a.use_mpmath, a.dps, a.backend), geometry=geometry,
        numerical={key: value for key, value in result.settings["numerical"].items()
                   if key not in ("N", "quadrature_order", "newton_atol")},
    )


def report_convergence_table(results, *, order=None):
    """Compare compatible manufactured experiments without assigning a rate.

    Physics, material declarations, arithmetic, and solver/integration targets
    must match. N and quadrature order may differ. The requested spelling of
    newton_atol is excluded from compatibility; its canonical tolerance is kept.
    """
    results = tuple(results)
    if not results:
        raise ValueError("provide at least one manufactured result")
    expected = _convergence_signature(results[0])
    previous = 1
    for result in results:
        N = _resolution(result.solution.N)
        if N <= previous:
            raise ValueError("convergence table N values must be strictly increasing")
        previous = N
        if _convergence_signature(result) != expected:
            raise ValueError("convergence results must share physics, material, arithmetic and numerical targets")
    summary = ["N | Unknowns | Newton atol | Residual norm (Euclidean) | Status"]
    absolute = ["N | Absolute L2(u) | Absolute L2(w) | Absolute L2(flux)"]
    relative = ["N | Relative L2(u) (%) | Relative L2(w) (%) | Relative L2(flux) (%)"]
    for result in results:
        solution, a = result.solution, result.solution.arithmetic
        metrics = manufactured_error_metrics(result, order=order)
        final = solution.newton_result
        residual = _number(final.residual_norm, a, 7)
        if final.homotopy != 0:
            residual += " at q={}".format(_number(final.homotopy, a, 7))
        summary.append("{} | {} | {} | {} | {}".format(
            solution.N, solution.unknown_count, result.settings["numerical"]["newton_atol"], residual,
            "verified" if solution.converged else "PROVISIONAL"))
        absolute.append("{} | {}".format(solution.N, " | ".join(
            _number(values["absolute"], a, 7) for values in metrics.values())))
        relative.append("{} | {}".format(solution.N, " | ".join(
            _relative_percent(values["relative"], a) for values in metrics.values())))
    lines = ["Manufactured N convergence: {}".format(results[0].model),
             "Physical inputs and Newton/integration targets are held fixed; N and quadrature order may vary.",
             "\n".join(summary), "\n".join(absolute), "\n".join(relative),
             "PROVISIONAL rows have unverified equation or integration solves.",
             "Residuals are physical at q=0; a displayed nonzero q identifies a continuation residual.",
             "Relative errors divide by the reference norm; zero reference norms are undefined.",
             "These numerically integrated errors do not impose a convergence rate or an accuracy threshold."]
    return "\n\n".join(lines)


def _homotopy_plans(experiment):
    solution = experiment.solution
    plans = tuple(item.newton_result for item in solution.quadrature_history
                  if item.newton_result is not None)
    return plans or (() if solution.newton_result is None else (solution.newton_result,))


def homotopy_stage_rows(experiment):
    """Return native immutable stage observations, including rejected trials.

    Plan numbers start at one; attempt numbers are zero-based row indices
    within each plan. accepted_q is the last verified parameter after that
    attempt, or None when this plan has no accepted state yet.
    """
    rows, a = [], experiment.solution.arithmetic
    for plan_index, result in enumerate(_homotopy_plans(experiment), 1):
        accepted_q, attempt = None, 0

        def append(kind, q, accepted, target, residual, iterations, reason, **extra):
            nonlocal accepted_q, attempt
            if accepted:
                accepted_q = q
            rows.append(MappingProxyType(dict(
                kind=kind, plan=plan_index, attempt=attempt, q=q, accepted_q=accepted_q,
                accepted=bool(accepted), target_norm=target, residual_norm=residual,
                iterations=iterations, reason=reason, **extra)))
            attempt += 1

        if result.stages:
            first = result.stages[0]
            append("initial", first.homotopy, first.converged, first.target_norm,
                   first.final_norm, first.iterations, first.reason)
        reference = result.reference_result
        if reference is not None:
            append("reference", a.scalar(1), reference.converged, reference.target_norm,
                   reference.residual_norm, 0, reference.reason, linear_solves=reference.linear_solves)
        for stage in result.stages[1:]:
            append("continuation", stage.homotopy, stage.converged, stage.target_norm,
                   stage.final_norm, stage.iterations, stage.reason)
    return tuple(rows)


def report_homotopy_stages(experiment):
    """Describe accepted states and rollback using stored solver observations."""
    rows, a = homotopy_stage_rows(experiment), experiment.solution.arithmetic
    if not rows:
        return "No Newton or reference stage history was recorded."
    lines = ["Plan | Kind | q tried | Accepted q | Newton iterations | Residual | Target | Status | Reason"]
    for row in rows:
        status = "accepted" if row["accepted"] else ("rejected" if row["kind"] == "continuation" else "failed")
        reason = row["reason"]
        if "linear_solves" in row:
            reason += " ({} reference linear solves)".format(row["linear_solves"])
        lines.append("{} | {} | {} | {} | {} | {} | {} | {} | {}".format(
            row["plan"], row["kind"], _number(row["q"], a, 7),
            "none" if row["accepted_q"] is None else _number(row["accepted_q"], a, 7),
            row["iterations"], _number(row["residual_norm"], a, 7),
            _number(row["target_norm"], a, 7), status, reason))
    for index, plan in enumerate(_homotopy_plans(experiment), 1):
        if not plan.converged:
            lines.append("Plan {} stopped: {} (final q={}).".format(
                index, plan.reason, _number(plan.homotopy, a, 7)))
    lines.append("Rejected continuation trials keep the last accepted q. Each quadrature plan is checked separately.")
    return "\n".join(lines)


def report_homotopy_comparison(results):
    """Compare requested budgets with actual work, without an efficiency claim."""
    if not isinstance(results, Mapping) or not results:
        raise ValueError("provide a nonempty mapping of labels to experiments")
    lines = ["Run | Newton atol | Stage cap | Total cap | Newton attempts | Reference used | Accepted positive-q stages | Final q | Final residual | Integration verified | Converged"]
    for label, experiment in results.items():
        solution, a = experiment.solution, experiment.solution.arithmetic
        plans, rows = _homotopy_plans(experiment), homotopy_stage_rows(experiment)
        final = solution.newton_result
        references = [plan.reference_result for plan in plans if plan.reference_result is not None]
        reference = ("verified" if any(item.converged for item in references) else
                     "failed" if references or any(plan.reason.startswith("reference_") for plan in plans) else "no")
        positive = sum(row["kind"] == "continuation" and row["accepted"] and 0 < row["q"] < 1 for row in rows)
        options = experiment._options
        lines.append("{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}".format(
            label, experiment.settings["numerical"]["newton_atol"],
            options.get("max_iterations", "solver default"),
            getattr(options.get("continuation"), "max_total_iterations", "solver default"),
            sum(plan.iterations for plan in plans) if plans else "not recorded", reference, positive,
            _number(final.homotopy, a, 7) if final is not None else "not recorded",
            _number(final.residual_norm, a, 7) if final is not None else "not recorded",
            "yes" if solution.quadrature_converged else "no", "yes" if solution.converged else "no"))
    lines.append("Newton totals include rejected attempts across all quadrature plans; reference linear solves are additional work.")
    lines.append("Final residuals are physical only at q=0. Several continuation stages can use more work than one direct solve.")
    return "\n".join(lines)


def compare_resolution(result, *, fine_N, newton_atol, verbosity=0):
    """Compare resolutions with an explicit fine-solve absolute Newton target."""
    fine_N = _resolution(fine_N, "fine_N")
    if fine_N <= result.solution.N:
        raise ValueError("fine_N must be greater than the current N")
    if result.problem is None:
        raise ValueError("resolution comparison requires the original unbound problem definition")
    options = dict(result._options)
    options.pop("N")
    options.pop("tolerance")
    options["quadrature_order"] = max(2*fine_N+4, result.solution.plan.order)
    options["initial_guess"] = lambda x, a: result.solution.u(x)
    fine = run_defined_problem(
        result.problem, N=fine_N, newton_atol=newton_atol, material=result.material, case=result.case,
        model=result.model, accuracy=result.accuracy, verbosity=verbosity, numerical=options,
    )
    # Catalog spelling is display metadata only. The original problem above
    # supplies every physical callback and parameter for this new solve.
    fine = replace(fine, settings=_freeze({**fine.settings, "physical": result.settings["physical"]}))
    solution, a = fine.solution, fine.solution.arithmetic
    plan = _field_plan(solution, 2*solution.plan.order)
    changes = {}
    for field_name in ("u", "w", "flux"):
        old = getattr(result.solution, field_name)
        difference = solution.l2_error(lambda x, callback=old: a.scalar(callback(x)), field=field_name, plan=plan)
        scale = solution.l2_norm(field=field_name, plan=plan)
        changes[field_name] = difference/scale if scale != 0 else (a.scalar(0) if difference == 0 else a.scalar("inf"))
    return ResolutionComparison(result, fine, MappingProxyType(changes))


def run_preset_sweep(*, N_by_interfaces, newton_atol,
                     interface_counts=SUPPORTED_INTERFACE_COUNTS, kind="constant",
                     models=MODELS, accuracy="standard", wall_potential="0.05", delta="1",
                     gamma="0.5", sigma="2", mu="0.1", verbosity=0, numerical=None):
    """Solve selected presets at explicitly chosen resolutions for each count.

    Every selected count, its resolution, and the absolute Newton tolerance
    are checked before any solve.
    Additional mapping entries are ignored, so one table can serve subsets.
    """
    if not isinstance(N_by_interfaces, Mapping):
        raise ValueError("N_by_interfaces must map interface counts to explicit N values")
    counts, models = tuple(interface_counts), tuple(models)
    resolutions = {}
    for count in counts:
        if (isinstance(count, bool) or not isinstance(count, Integral)
                or count not in SUPPORTED_INTERFACE_COUNTS):
            raise ValueError("choose interface counts from 1, 2, 3, 5, 7, or 10")
        if count not in N_by_interfaces:
            raise ValueError("N_by_interfaces is missing N for {} interfaces".format(count))
        resolutions[count] = _resolution(N_by_interfaces[count], "N for {} interfaces".format(count))
    if any(model not in MODELS for model in models):
        raise ValueError("model must be BSK, manufactured_linear, or manufactured_nonlinear")
    options = _numerical_options(accuracy, numerical)
    _newton_tolerance(newton_atol, options)
    materials = {count: preset_material(count, kind=kind) for count in counts}
    return [run_experiment(materials[count], N=resolutions[count], newton_atol=newton_atol,
                           model=model, accuracy=accuracy,
                           wall_potential=wall_potential, delta=delta, gamma=gamma,
                           sigma=sigma, mu=mu, verbosity=verbosity, numerical=numerical)
            for count in counts for model in models]


def _validate_case_name(name):
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
        raise ValueError("case names must be portable identifiers containing letters, digits, '.', '_' or '-'")
    return name


def _figure_solution_metadata(solution):
    a, physical = solution.arithmetic, solution.physical
    mapping = getattr(solution, "mapping", getattr(getattr(solution, "basis", None), "mapping", None))
    return _json_value(dict(
        solver_type=solution.solver_type, N=solution.N, unknown_count=solution.unknown_count,
        mapping_type=getattr(mapping, "mapping_type", None),
        domain=tuple(value if a.isfinite(value) else str(value) for value in solution.domain),
        arithmetic=dict(backend="mpmath" if a.use_mpmath else "numpy", implementation=a.backend,
                        precision_dps=a.dps, precision_bits=a.context.prec if a.use_mpmath else 53),
        delta=physical.delta, boundary_conditions=physical.boundary_conditions,
        epsilon=dict(lower_bound=physical.epsilon.lower_bound, upper_bound=physical.epsilon.upper_bound,
                     wall_value=physical.epsilon.wall_value, interfaces=physical.epsilon.interfaces),
    ), a)


def _preflight_export(result, *, figures=None, resolution=None, case_name=None):
    """Validate ownership and snapshots without creating paths or evaluating fields."""
    solution = result.solution
    solution.check_precision()
    if case_name is not None:
        _validate_case_name(case_name)
    if resolution is not None:
        if getattr(getattr(resolution, "coarse", None), "solution", None) is not solution:
            raise ValueError("the resolution comparison must belong to this solution")
        fine = resolution.fine.solution
        fine.check_precision()
        if (fine.solver_type != solution.solver_type or fine.domain != solution.domain
                or fine.N <= solution.N or fine.physical.delta != solution.physical.delta
                or fine.physical.boundary_conditions != solution.physical.boundary_conditions
                or any(getattr(fine.physical.epsilon, name) != getattr(solution.physical.epsilon, name)
                       for name in ("interfaces", "lower_bound", "upper_bound", "wall_value"))):
            raise ValueError("the resolution comparison must retain the physical problem and increase N")
        if not isinstance(resolution.relative_changes, Mapping):
            raise ValueError("resolution changes must be a mapping of finite nonnegative values")
        for value in resolution.relative_changes.values():
            value = fine.arithmetic.scalar(value)
            if not fine.arithmetic.isfinite(value) or value < 0:
                raise ValueError("resolution changes must be finite and nonnegative")
    if figures is not None and not isinstance(figures, Mapping):
        raise ValueError("figures must be a mapping of filenames to figures")
    normalized, provenance = {}, {}
    for label, figure in (() if figures is None else figures.items()):
        name = str(label)
        if (not name or Path(name).name != name or name in (".", "..") or "\\" in name
                or any(ord(character) < 32 or ord(character) == 127 for character in name)
                or name in ("settings", "solution", "profiles") or name in normalized):
            raise ValueError("figure names must be unique simple filenames and avoid reserved export names")
        if not callable(getattr(figure, "savefig", None)):
            raise ValueError("each exported figure must provide savefig(path)")
        normalized[name] = figure
        if not hasattr(figure, "_overscreening_solutions"):
            provenance[name] = dict(ownership="untagged", note="Figure contents are the caller's responsibility.")
            continue
        owners = figure._overscreening_solutions
        if not isinstance(owners, tuple) or not owners:
            raise ValueError("tagged figures must record their solution owners")
        if not any(owner is solution for owner in owners):
            raise ValueError("figure '{}' belongs to a different solution".format(name))
        for owner in owners:
            owner.check_precision()
        provenance[name] = dict(
            ownership="comparison" if len(owners) > 1 else "single_solution",
            solutions=[_figure_solution_metadata(owner) for owner in owners],
        )
    return MappingProxyType(normalized), _freeze(provenance)


def export_experiment(result, directory, *, figures=None, samples=101, x_max=None, resolution=None,
                      case_name=None):
    """Write settings, a typed solution, native decimal profiles, and optional PDFs.

    Custom executable coefficient functions are never serialized. Their
    declarations are exported, with an instruction to supply the function
    again when loading the typed solution. Tagged package figures must include
    this solution among their owners; comparison figures record every owner.
    Untagged user figures and material previews remain the caller's responsibility.
    """
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples < 2:
        raise ValueError("export samples must be an integer of at least two")
    figures, provenance = _preflight_export(
        result, figures=figures, resolution=resolution, case_name=case_name)
    solution, a = result.solution, result.solution.arithmetic
    domain = _solution_domain(solution)
    left = domain[0]
    finite = a.isfinite(domain[1])
    right = a.scalar(domain[1] if x_max is None and finite else "8" if x_max is None else x_max)
    if not a.isfinite(right) or right <= left:
        raise ValueError("export x_max must be finite and greater than the left domain endpoint")
    if finite and right > domain[1]:
        raise ValueError("export x_max lies outside the finite problem domain")
    directory = Path(directory).expanduser()
    paths = {name: directory/filename for name, filename in (
        ("settings", "settings.json"), ("solution", "solution.json"), ("profiles", "profiles.csv"))}
    settings = _json_value(result.settings, a)
    if case_name is not None:
        settings["case_name"] = case_name
    settings["figure_provenance"] = _json_value(provenance, a)
    settings["versions"] = {name: version(name) for name in (
        "overscreening-breakdown", "numpy", "scipy", "mpmath")}
    settings["diagnostics"] = dict(
        converged=solution.converged, nonlinear_converged=solution.nonlinear_converged,
        quadrature_converged=solution.quadrature_converged, resolution_checked=resolution is not None,
        elapsed_seconds=result.elapsed_seconds, interface_value_convention="right trace",
        note="Numerical convergence and integration checks do not certify spatial accuracy.")
    if resolution is not None:
        settings["resolution"] = dict(
            coarse_N=resolution.coarse.solution.N, fine_N=resolution.fine.solution.N,
            coarse_newton_atol=resolution.coarse.settings["numerical"]["newton_atol"],
            fine_newton_atol=resolution.fine.settings["numerical"]["newton_atol"],
            fine_verified=resolution.fine.solution.converged,
            relative_changes=_json_value(resolution.relative_changes, a),
            note="A two-resolution comparison is a sensitivity check, not a rigorous error bound.")
    encoded_settings = json.dumps(settings, indent=2, allow_nan=False)+"\n"
    directory.mkdir(parents=True, exist_ok=True)
    paths["settings"].write_text(encoded_settings)
    solution.save(paths["solution"])

    def decimal(value):
        return a.context.nstr(value, n=a.dps+3) if a.use_mpmath else format(float(value), ".17g")

    points = [left+(right-left)*a.scalar(index)/(int(samples)-1) for index in range(int(samples))]
    points = sorted(set(points+list(point for point in solution.physical.epsilon.interfaces if point <= right)))
    with paths["profiles"].open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "epsilon", "u", "w", "flux"))
        for x in points:
            writer.writerow(decimal(value) for value in (
                x, solution.physical.epsilon(x), solution.u(x), solution.w(x), solution.flux(x)))
    for name, figure in figures.items():
        path = directory/(name+".pdf")
        figure.savefig(path)
        paths[name] = path
    return paths


from .discontinuous_plotting import (  # noqa: E402
    plot_material, plot_material_gallery, plot_experiment, plot_errors, plot_convergence,
)
