"""Run one bounded manufactured-solution study and retain native diagnostics.

Run from an installed checkout with ``python -m benchmarks.run_manufactured``.
The default is one small case, not the historical Cartesian parameter sweep.
"""

import argparse
from contextlib import nullcontext, redirect_stdout
import csv
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.continuation import ContinuationOptions, ScreenedLinearReference
from overscreening_breakdown.layered import SUPPORTED_INTERFACE_COUNTS, segment_material
from overscreening_breakdown.quadrature import QuadratureFactory
from overscreening_breakdown.solvers import solve


SAMPLE_POINTS = ("0", "0.125", "0.5", "1", "2", "4", "8")


class _Tee:
    """Retain a benchmark's solver log while streaming requested progress."""

    def __init__(self, captured, live):
        self.captured, self.live = captured, live

    def write(self, text):
        count = self.captured.write(text)
        self.live.write(text)
        return count

    def flush(self):
        self.captured.flush()
        self.live.flush()


def _positive_integer(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _nonnegative_integer(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def parser(*, precision_study=False):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--case", action="append", dest="cases", help=(
        "repeatable: linear:1..8, nonlinear:1..7, rational:constant/scaled/smooth/"
        "exponential, rational-nonlinear:<coefficient>, transmission, "
        "finite:constant/smooth/layered, finite-nonlinear:constant/smooth/layered"))
    result.add_argument("--solver", choices=("collocation", "galerkin", "legacy_collocation"),
                        default="galerkin")
    result.add_argument("--N", nargs="+", type=_positive_integer, default=[8], dest="resolutions")
    result.add_argument("--L", default="1", help="mapping length; decimal strings retain precision")
    result.add_argument("--R", default="4", help="physical interval [0,R] for finite cases")
    result.add_argument("--delta", default="2", help="positive delta for finite cases")
    result.add_argument("--beta", default="0.125",
                        help="nonnegative cubic strength for finite-nonlinear cases")
    result.add_argument("--interfaces", type=int, choices=SUPPORTED_INTERFACE_COUNTS, default=3,
                        help="interface count for finite layered cases")
    result.add_argument("--material-style", choices=("constant", "graded"), default="constant",
                        help="within-layer profile for finite layered cases")
    result.add_argument("--arithmetic", choices=("numpy", "mpmath"),
                        default="mpmath" if precision_study else "numpy")
    result.add_argument("--backend", choices=("auto", "python", "gmpy"), default="auto",
                        help="verify the actual process mpmath backend; does not change it")
    if precision_study:
        result.add_argument("--precisions", nargs="+", type=_positive_integer, default=[30, 50, 100])
    else:
        result.add_argument("--study", choices=("single", "resolution", "quadrature"), default="single")
        result.add_argument("--precision", type=_positive_integer, default=50)
    result.add_argument("--method", choices=("newton", "root", "root_no_jac", "nonlin_solve"),
                        default="newton")
    result.add_argument("--tolerance", default="auto" if precision_study else "1e-10",
                        help="positive decimal value, or auto for working-precision guard digits")
    result.add_argument("--max-iterations", type=_nonnegative_integer, default=50)
    result.add_argument("--no-globalization", action="store_true")
    result.add_argument("--verbosity", type=int, choices=range(4), default=0,
                        help="solver progress: 0 quiet, 1 phases, 2 steps, 3 detailed trials")
    result.add_argument("--continuation-step", default="0.25",
                        help="initial reference-to-physical parameter decrement (at most 0.5)")
    result.add_argument("--max-continuation-stages", type=_nonnegative_integer, default=100,
                        help="continuation attempts across all plans, including rejected stages")
    result.add_argument("--max-total-iterations", type=_nonnegative_integer, default=1000,
                        help="Newton attempts across all plans, including initial physical solves")
    result.add_argument("--reference-strength", default="1",
                        help="positive normalized reaction in the screened linear reference")
    result.add_argument("--quadrature-order", nargs="+", type=_positive_integer)
    result.add_argument("--subdivisions", type=_positive_integer, default=1)
    result.add_argument("--quadrature-atol")
    result.add_argument("--quadrature-rtol")
    result.add_argument("--max-refinements", type=_nonnegative_integer,
                        default=0 if precision_study else 3)
    result.add_argument("--diagnostic-order", type=_positive_integer, default=64,
                        help="independent points per panel for physical error quadrature")
    result.add_argument("--diagnostic-endpoint", default="8",
                        help="upper diagnostic endpoint for half-line collocation; finite cases use [0,R]")
    result.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    return result


def _finite_parameters(selector, options):
    family, _, coefficient = selector.partition(":")
    if family not in ("finite", "finite-nonlinear"):
        return None
    if coefficient not in ("constant", "smooth", "layered"):
        raise ValueError("finite coefficient must be constant, smooth, or layered")
    if options is None:
        options = parser().parse_args([])
    for name in ("R", "delta", "beta"):
        raw = getattr(options, name)
        try:
            value = Decimal(raw)
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValueError("finite {} must be a finite decimal number".format(name)) from error
        if not value.is_finite() or (value < 0 if name == "beta" else value <= 0):
            constraint = "nonnegative" if name == "beta" else "strictly positive"
            raise ValueError("finite {} must be finite and {}".format(name, constraint))
    layered = coefficient == "layered"
    material = (segment_material(options.interfaces, R=options.R, kind=options.material_style)
                if layered else None)
    return {"R": options.R, "delta": options.delta,
            "beta": options.beta if family == "finite-nonlinear" else "0",
            "nonlinear": family == "finite-nonlinear", "coefficient": coefficient,
            "interfaces": options.interfaces if layered else 0,
            "material_style": options.material_style if layered else None,
            "material": material.to_dict() if material is not None else None}


def _case(selector, options=None):
    from overscreening_breakdown.manufactured import (
        finite_segment_case, linear_case, nonlinear_case, rational_case, transmission_case,
    )
    parameters = _finite_parameters(selector, options)
    if parameters is not None:
        from overscreening_breakdown.layered import LayeredMaterial
        coefficient = (LayeredMaterial(**parameters["material"])
                       if parameters["material"] is not None else parameters["coefficient"])
        return finite_segment_case(
            coefficient, R=parameters["R"], delta=parameters["delta"],
            nonlinear=parameters["nonlinear"], beta=parameters["beta"],
        )
    family, separator, argument = selector.partition(":")
    if family == "linear" and separator:
        return linear_case(int(argument))
    if family == "nonlinear" and separator:
        return nonlinear_case(int(argument))
    if family in ("rational", "rational-nonlinear") and separator:
        return rational_case(coefficient=argument, nonlinear=family == "rational-nonlinear")
    if selector == "transmission":
        return transmission_case()
    raise ValueError("unknown case selector {!r}".format(selector))


def _decimal(value, arithmetic):
    """Decimal output without a float64 intermediate, with roundtrip guard digits."""
    if arithmetic.use_mpmath:
        return arithmetic.context.nstr(arithmetic.scalar(value), n=arithmetic.dps+8,
                                       strip_zeros=False)
    return repr(float(value))


def _portable(value, arithmetic):
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if is_dataclass(value):
        # Traverse native diagnostic objects without deepcopying their
        # context-bound mpmath matrices into a foreign arithmetic context.
        return {item.name: _portable(getattr(value, item.name), arithmetic)
                for item in fields(value)}
    if isinstance(value, dict):
        return {str(k): _portable(v, arithmetic) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(v, arithmetic) for v in value]
    if isinstance(value, np.ndarray):
        return _portable(value.tolist(), arithmetic)
    if hasattr(value, "rows") and hasattr(value, "cols"):
        if value.cols == 1:
            return [_portable(value[row, 0], arithmetic) for row in range(value.rows)]
        return [[_portable(value[row, column], arithmetic) for column in range(value.cols)]
                for row in range(value.rows)]
    return _decimal(value, arithmetic)


def _metadata():
    packages = {}
    for name in ("overscreening-breakdown", "numpy", "scipy", "mpmath", "gmpy2"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, check=False)
        revision_value = revision.stdout.strip() if revision.returncode == 0 else None
        dirty = bool(status.stdout) if status.returncode == 0 else None
    except OSError:
        revision_value, dirty = None, None
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(), "versions": packages,
            "mpmath_backend": mp.libmp.BACKEND,
            "revision": revision_value, "working_tree_dirty": dirty}


def _native_scalar(value):
    return np.asarray(value).item()


def _diagnostic_domain(arithmetic, domain=None, endpoint=None):
    left, right = (arithmetic.scalar(value) for value in
                   ((0, "inf") if domain is None else domain))
    if arithmetic.isfinite(right) or endpoint is None:
        return (left, right), False
    endpoint = arithmetic.scalar(endpoint)
    if not arithmetic.isfinite(endpoint) or endpoint <= left:
        raise ValueError("diagnostic endpoint must be finite and positive")
    return (left, endpoint), True


def _sample_points(arithmetic, domain):
    left, right = domain
    if right == arithmetic.inf:
        return tuple(arithmetic.scalar(point) for point in SAMPLE_POINTS)
    # Preserve the old [0,8] samples when that window is selected, and use
    # the same fractions on other finite intervals. Retain exact endpoints.
    return tuple(left if point == "0" else right if point == "8" else
                 left+(arithmetic.scalar(point)/8)*(right-left) for point in SAMPLE_POINTS)


def _errors(case, arithmetic, evaluator, *, interfaces, L, order, include_flux,
            endpoint=None, domain=None):
    diagnostic_domain, window = _diagnostic_domain(arithmetic, domain, endpoint)
    native_interfaces = tuple(arithmetic.scalar(point) for point in interfaces)
    split = tuple(point for point in native_interfaces
                  if diagnostic_domain[0] < point < diagnostic_domain[1])
    plan = QuadratureFactory(arithmetic, L=L, interfaces=split, domain=diagnostic_domain).plan(order)
    scalar = arithmetic.scalar
    sqrt = arithmetic.sqrt
    integrate = plan.integrate
    suffix = "_finite_interval" if window else ""
    sample_points = _sample_points(arithmetic, diagnostic_domain)
    errors = {}
    fields = ("u", "w", "flux") if include_flux else ("u", "w")
    for field in fields:
        reference = getattr(case, field)
        difference = lambda x: evaluator(field, x)-reference(x, arithmetic)
        errors[field+"_l2"+suffix] = sqrt(integrate(lambda x: difference(x)**2))
        errors[field+"_sample_max"] = max(abs(difference(scalar(x))) for x in sample_points)
        if field != "flux":
            derivative = getattr(case, field+"_dx")
            errors[field+"_h1"+suffix] = sqrt(integrate(
                lambda x: difference(x)**2+(evaluator(field+"_dx", x)-derivative(x, arithmetic))**2))
    return errors


def _newton_metadata(result):
    if result is None:
        return None
    return {"reason": result.reason, "residual_norm": result.residual_norm,
            "homotopy": result.homotopy, "history": result.history, "stages": result.stages,
            "iterations": result.iterations, "adjoint_steps": result.adjoint_steps,
            "adjoint_evaluations": result.adjoint_evaluations,
            "recovery_reasons": result.recovery_reasons,
            "reference_result": result.reference_result,
            "continuation_attempts": result.continuation_attempts}


def _continuation_options(options):
    return ContinuationOptions(
        reference=ScreenedLinearReference(strength=options.reference_strength),
        initial_step=options.continuation_step,
        max_stages=options.max_continuation_stages,
        max_total_iterations=options.max_total_iterations,
    )


def _diagnostics(*args, **kwargs):
    started = perf_counter()
    try:
        arithmetic = args[1]
        domain, window = _diagnostic_domain(arithmetic, kwargs.get("domain"), kwargs.get("endpoint"))
        samples = _sample_points(arithmetic, domain)
        native_interfaces = tuple(arithmetic.scalar(point) for point in kwargs["interfaces"])
        split = tuple(point for point in native_interfaces if domain[0] < point < domain[1])
        return {"errors": _errors(*args, **kwargs), "diagnostic_success": True,
                "diagnostic_sample_points": samples, "diagnostic_interfaces": split,
                "diagnostic_mapping": "affine_interval" if arithmetic.isfinite(domain[1]) else "rational_half_line",
                "diagnostic_scope": "truncated_half_line" if window else "physical_domain",
                "diagnostic_seconds": perf_counter()-started}
    except Exception as error:
        return {"errors": None, "diagnostic_success": False,
                "diagnostic_error_type": type(error).__name__, "diagnostic_error": str(error),
                "diagnostic_traceback": traceback.format_exc(),
                "diagnostic_seconds": perf_counter()-started}


def _common(case, options, N, precision, quadrature_order):
    started = perf_counter()
    solution = solve(
        case.make_problem(), N, solver=options.solver, L=options.L,
        use_mpmath=options.arithmetic == "mpmath", precision=precision,
        method=options.method, tolerance=options.tolerance,
        max_iterations=options.max_iterations, use_globalization=not options.no_globalization,
        continuation=_continuation_options(options), quadrature_order=quadrature_order,
        subdivisions=options.subdivisions, quadrature_atol=options.quadrature_atol,
        quadrature_rtol=options.quadrature_rtol, max_refinements=options.max_refinements,
        verbosity=options.verbosity,
    )
    solve_seconds = perf_counter()-started
    arithmetic = solution.arithmetic
    finite = arithmetic.isfinite(solution.domain[1])
    endpoint = options.diagnostic_endpoint if options.solver == "collocation" and not finite else None
    diagnostics = _diagnostics(
        case, arithmetic, lambda field, x: getattr(solution, field)(x),
        interfaces=solution.physical.epsilon.interfaces, L=options.L,
        order=options.diagnostic_order, include_flux=True,
        endpoint=endpoint, domain=solution.domain)
    plan = getattr(solution, "plan", None)
    data = {"method": solution.method, "unknown_count": solution.unknown_count,
            "converged": bool(solution.converged),
            "nonlinear_converged": bool(solution.nonlinear_converged),
            "quadrature_converged": getattr(solution, "quadrature_converged", None),
            "final_quadrature_order": plan.order if plan is not None else None,
            "final_subdivisions": plan.subdivisions if plan is not None else None,
            "newton": _newton_metadata(solution.newton_result),
            "scipy": getattr(solution, "scipy_result", None),
            "wall_u_defect": solution.u(0)-solution.physical.boundary_conditions[0][0],
            "far_u_trace": None if finite else solution.u(arithmetic.inf),
            "right_u_trace": solution.u(solution.domain[1]) if finite else None,
            "right_u_defect": (solution.u(solution.domain[1])-solution.physical.boundary_conditions[1][0]
                               if finite and solution.physical.boundary_conditions[1][0] is not None else None),
            "diagnostic_domain": ["0", _decimal(solution.domain[1], arithmetic) if finite else
                                  options.diagnostic_endpoint if endpoint is not None else "inf"],
            "mapping_type": "affine_interval" if finite else "rational_half_line",
            "initialization_policy": (("linear_endpoint_lift_zero_w" if options.solver == "galerkin"
                                       else "wall_exponential") if finite else None),
            "quadrature_history": getattr(solution, "quadrature_history", ()),
            "physical": {"delta": solution.physical.delta,
                         "domain": solution.physical.domain,
                         "boundary_conditions": solution.physical.boundary_conditions,
                         "epsilon_lower_bound": solution.physical.epsilon.lower_bound,
                         "epsilon_upper_bound": solution.physical.epsilon.upper_bound,
                         "epsilon_wall_value": solution.physical.epsilon.wall_value,
                         "epsilon_interfaces": solution.physical.epsilon.interfaces},
            "coefficients": tuple(solution.coefficients), "solve_seconds": solve_seconds,
            **diagnostics}
    return data, arithmetic


def _legacy(case, options, N, precision):
    from overscreening_breakdown.collocation.spectral1d import (
        collocation_discretization, solve_nonlinear_problem,
    )
    use_mp = options.arithmetic == "mpmath"
    arithmetic = MpmathArithmetic(precision) if use_mp else NumpyArithmetic()
    problem = case.legacy_problem()
    started = perf_counter()
    if callable(getattr(problem, "right_hand_side", None)):
        discretization = solve_nonlinear_problem(
            N, problem=problem, L=arithmetic.scalar(options.L), use_mpmath=use_mp, prec=precision,
            use_method=options.method, tolerance=arithmetic.scalar(options.tolerance),
            total_iterations=options.max_iterations,
            use_globalization=not options.no_globalization,
            continuation=_continuation_options(options),
            verbosity=options.verbosity)
        coefficients = discretization.solve_problem()
        converged = discretization.is_converged()
        result = discretization.get_newton_result()
        scipy_result = discretization.get_scipy_result()
        method = options.method
    else:
        discretization = collocation_discretization(
            N, domain=problem.get_domain(), L=arithmetic.scalar(options.L),
            use_mpmath=use_mp, prec=precision)
        problem.set_use_mp_math(discretization.get_mpmath_ref())
        discretization.set_boundary_conditions(problem.get_boundary_conditions())
        matrix = discretization.bilinear_form(discretization.psi, 0)
        derivatives = (discretization.psi, discretization.dpsi, discretization.ddpsi,
                       discretization.dddpsi, discretization.ddddpsi)
        for factor, derivative in zip(problem.operator(), derivatives):
            if factor is not None:
                matrix += discretization.bilinear_form(derivative, factor)
        matrix = discretization.set_boundary_bilinear_form_tau_method(matrix)
        rhs = discretization.linear_functional(lambda x, t: problem.rhs_in_domain(x))
        rhs = discretization.linear_functional_boundary(rhs)
        coefficients = discretization.solve(matrix, rhs)
        residual = matrix*coefficients-rhs if use_mp else matrix@coefficients-rhs
        converged = arithmetic.norm(residual) <= arithmetic.scalar(options.tolerance)*(1+arithmetic.norm(rhs))
        result, scipy_result, method = None, None, "direct"
    solve_seconds = perf_counter()-started

    def evaluate(field, x):
        order = {"u": 0, "u_dx": 1, "w": 2, "w_dx": 3}[field]
        if order and (x == 0 or x == arithmetic.inf):
            endpoint = (discretization.psi_with_derivative_value_at_right if x == 0 else
                        discretization.psi_with_derivative_value_at_left)
            value = sum(_native_scalar(endpoint(index, order))*coefficient
                        for index, coefficient in enumerate(coefficients))
        else:
            y = discretization.arg_from_domain_to_basis(x)
            value = _native_scalar(discretization.solution_with_derivative_in_basis(coefficients, y, order))
        return arithmetic.scalar(-value if order > 1 else value)

    diagnostics = _diagnostics(case, arithmetic, evaluate, interfaces=(), L=options.L,
                               order=options.diagnostic_order, include_flux=False,
                               endpoint=options.diagnostic_endpoint)
    return {"method": method, "unknown_count": len(coefficients),
            "converged": bool(converged),
            "nonlinear_converged": bool(converged) if result is not None or scipy_result is not None else None,
            "quadrature_converged": None, "newton": _newton_metadata(result),
            "scipy": scipy_result,
            "operator": tuple(problem.operator()),
            "boundary_conditions": problem.get_boundary_conditions(),
            "wall_u_defect": evaluate("u", arithmetic.scalar(0))-arithmetic.scalar(problem.get_boundary_conditions()[0][0]),
            "far_u_trace": evaluate("u", arithmetic.inf),
            "diagnostic_domain": ["0", options.diagnostic_endpoint],
            "coefficients": tuple(coefficients), "solve_seconds": solve_seconds,
            **diagnostics}, arithmetic


def _validate(options, argument_parser, *, precision_study):
    if options.backend != "auto" and options.backend != mp.libmp.BACKEND:
        argument_parser.error("actual mpmath backend is {!r}; start a new process with the desired backend".format(mp.libmp.BACKEND))
    orders = options.quadrature_order or [None]
    if precision_study:
        if options.arithmetic != "mpmath":
            argument_parser.error("the precision study requires --arithmetic mpmath")
        if len(options.resolutions) != 1 or len(orders) != 1:
            argument_parser.error("the precision study holds N and quadrature order fixed")
        if options.max_refinements != 0:
            argument_parser.error("the precision study requires --max-refinements 0")
    else:
        if options.study != "resolution" and len(options.resolutions) != 1:
            argument_parser.error("multiple N values require --study resolution")
        if options.study != "quadrature" and len(orders) != 1:
            argument_parser.error("multiple quadrature orders require --study quadrature")
        if options.study == "quadrature" and options.solver != "galerkin":
            argument_parser.error("quadrature studies require --solver galerkin")
        if options.study == "quadrature" and options.max_refinements != 0:
            argument_parser.error("use --max-refinements 0 to hold each quadrature rule fixed")
    if options.arithmetic == "mpmath" and options.method != "newton":
        argument_parser.error("mpmath supports newton only")
    if options.solver == "galerkin" and options.method != "newton":
        argument_parser.error("Galerkin supports newton only")


def _write(output_dir, metadata, records):
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"format_version": 1, "metadata": metadata,
              "diagnostic_note": "Errors use independent fixed quadrature, not rigorous bounds. Finite problems use their full physical segment. Half-line collocation uses the stated finite window because a nonzero far trace prevents physical integrability. Half-line Galerkin errors assume integrable reference fields. Sampled maxima use only the recorded diagnostic_sample_points.",
              "records": records}
    (output_dir/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    csv_records = [{"versions": metadata["versions"], "platform": metadata["platform"],
                    "python": metadata["python"], "revision": metadata["revision"],
                    "working_tree_dirty": metadata["working_tree_dirty"],
                    "created_utc": metadata["created_utc"], "study": metadata["study"],
                    "run_options": metadata["options"], **row} for row in records]
    columns = sorted({key for row in csv_records for key in row})
    with (output_dir/"results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in csv_records:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def main(argv=None, *, precision_study=False):
    argument_parser = parser(precision_study=precision_study)
    options = argument_parser.parse_args(argv)
    _validate(options, argument_parser, precision_study=precision_study)
    selectors = options.cases or ["rational:constant"]
    try:
        cases = [(selector, _case(selector, options), _finite_parameters(selector, options))
                 for selector in selectors]
    except (TypeError, ValueError) as error:
        argument_parser.error(str(error))
    for selector, case, parameters in cases:
        if options.solver not in case.supported_solvers and options.solver != "legacy_collocation":
            argument_parser.error("{} supports {}; select the solver explicitly".format(selector, case.supported_solvers))
        if options.solver == "legacy_collocation" and (
                parameters is not None or selector.startswith(("rational", "transmission"))):
            argument_parser.error("{} has no historical legacy operator".format(selector))
    metadata = _metadata()
    metadata["command"] = list(sys.argv if argv is None else argv)
    metadata["study"] = "precision" if precision_study else options.study
    metadata["options"] = {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(options).items()}
    records, failed = [], False
    precisions = options.precisions if precision_study else [options.precision]
    for selector, case, parameters in cases:
        for N in options.resolutions:
            for precision in precisions:
                for order in options.quadrature_order or [None]:
                    run_options = argparse.Namespace(**vars(options))
                    if run_options.tolerance == "auto":
                        run_options.tolerance = "1e-{}".format(max(8, precision-12) if options.arithmetic == "mpmath" else 10)
                    row = {"case": selector, "case_identifier": case.identifier,
                           "case_name": case.name, "case_notes": case.notes,
                           "supported_solvers": case.supported_solvers, "solver": options.solver,
                           "N": N, "L": None if parameters is not None else options.L,
                           "case_parameters": parameters,
                           "arithmetic": options.arithmetic,
                           "requested_precision": precision if options.arithmetic == "mpmath" else None,
                           "mpmath_backend": mp.libmp.BACKEND,
                           "requested_quadrature_order": order,
                           "tolerance": run_options.tolerance,
                           "diagnostic_quadrature_order": options.diagnostic_order}
                    log = io.StringIO()
                    try:
                        # Legacy callbacks use global mp; retain one scoped precision
                        # through assembly and diagnostics and restore it afterwards.
                        scope = mp.workdps(precision) if options.arithmetic == "mpmath" else nullcontext()
                        output = _Tee(log, sys.stdout) if options.verbosity else log
                        with scope, redirect_stdout(output):
                            data, arithmetic = (_legacy(case, run_options, N, precision)
                                                if options.solver == "legacy_collocation" else
                                                _common(case, run_options, N, precision, order))
                            row.update(_portable(data, arithmetic))
                            row["precision_bits"] = arithmetic.context.prec if arithmetic.use_mpmath else 53
                            row["actual_decimal_precision"] = arithmetic.dps
                        failed |= not row["converged"] or not row["diagnostic_success"]
                    except Exception as error:
                        row.update(converged=False, error_type=type(error).__name__, error=str(error),
                                   traceback=traceback.format_exc())
                        failed = True
                    row["solver_log"] = log.getvalue()
                    records.append(row)
                    _write(options.output_dir, metadata, records)
                    print("{} {} N={} precision={} converged={} diagnostics={}".format(
                        selector, options.solver, N, precision if options.arithmetic == "mpmath" else "float64",
                        row["converged"], row.get("diagnostic_success", False)), flush=True)
    print("results: {}".format(options.output_dir.resolve()), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
