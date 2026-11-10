"""Data-only, solver-discriminated solution snapshots.

The JSON format stores finite scalars as exact binary values, rather than
decimal display strings. mpmath coefficients retain their full mantissa and
the owner's exact working precision in bits. Loading never changes global
mpmath precision and never executes callback code from a file.

The original physical description must be supplied again. Its numerical
metadata and coefficient samples are compared at the saved precision before
an explicitly requested arithmetic conversion. Sampling cannot certify the
identity of arbitrary callbacks; source callbacks are the caller's choice.
Final convergence flags and projection provenance are retained, but Newton
iteration objects and quadrature iteration histories are not serialized.
Legacy pickle coefficient files remain with the existing trusted collocation
reader and must be explicitly imported before using this format.
"""

from decimal import Decimal, InvalidOperation
import json
from numbers import Integral
from pathlib import Path

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.problems.flux import BoundFluxProblem
from overscreening_breakdown.quadrature import QuadratureFactory


FORMAT = "overscreening_solution"
FORMAT_VERSION = 2
BASIS_IDS = {"collocation": "cosine_tau", "galerkin": "rational_chebyshev_mixed"}


def _keys(value, required, name):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError("invalid {} fields".format(name))
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def _number(value):
    """Encode a scalar without changing its original binary or decimal value."""
    if isinstance(value, bool):
        raise TypeError("boolean values are not numerical metadata")
    if hasattr(value, "_mpf_"):
        sign, mantissa, exponent, bits = value._mpf_
        if bits < 0:
            if value == mp.inf:
                return {"type": "infinity", "value": 1}
            raise ValueError("only positive infinity is supported in domain metadata")
        return {"type": "mpf", "value": [int(sign), str(mantissa), int(exponent), int(bits)]}
    if isinstance(value, (float, np.floating)):
        if isinstance(value, np.floating) and value.dtype.itemsize > 8 and np.isfinite(value):
            numerator, denominator = value.as_integer_ratio()
            sign, mantissa = int(numerator < 0), abs(numerator)
            exponent = 1-denominator.bit_length()
            while mantissa and mantissa % 2 == 0:
                mantissa //= 2
                exponent += 1
            if not mantissa:
                sign, exponent = 0, 0
            return {"type": "mpf", "value": [sign, str(mantissa), exponent, mantissa.bit_length()]}
        value = float(value)
        if np.isposinf(value):
            return {"type": "infinity", "value": 1}
        if not np.isfinite(value):
            raise ValueError("saved numerical values must be finite")
        return {"type": "float64", "value": value.hex()}
    if isinstance(value, Integral):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, (str, Decimal)):
        try:
            decimal = Decimal(value)
        except InvalidOperation as error:
            raise ValueError("numerical decimal metadata is invalid") from error
        if not decimal.is_finite():
            raise ValueError("numerical decimal metadata must be finite")
        return {"type": "decimal", "value": str(value)}
    raise TypeError("unsupported numerical metadata type")


def _decode_number(value, arithmetic, *, raw=False, allow_infinity=False):
    _keys(value, ("type", "value"), "numerical value")
    kind, payload = value["type"], value["value"]
    if kind == "infinity":
        if not allow_infinity or type(payload) is not int or payload != 1:
            raise ValueError("infinity is permitted only as a right domain endpoint")
        return arithmetic.scalar(arithmetic.inf)
    if kind == "mpf":
        if not isinstance(payload, list) or len(payload) != 4:
            raise ValueError("invalid mpmath binary tuple")
        sign, mantissa, exponent, bits = payload
        if (isinstance(sign, bool) or not isinstance(sign, int) or sign not in (0, 1)
                or not isinstance(mantissa, str) or not mantissa.isdecimal()
                or isinstance(exponent, bool) or not isinstance(exponent, int)):
            raise ValueError("invalid mpmath binary tuple")
        bits = _integer(bits, "mpmath mantissa bit count")
        mantissa = int(mantissa)
        if (mantissa.bit_length() != bits or (mantissa and mantissa % 2 == 0)
                or (not mantissa and (sign != 0 or exponent != 0))):
            raise ValueError("mpmath binary tuple must be canonical")
        context = arithmetic.context if arithmetic.use_mpmath else mp.mp.clone()
        exact = context.make_mpf((sign, mp.libmp.MPZ(mantissa), exponent, bits))
        return exact if raw else arithmetic.scalar(exact)
    if kind == "float64":
        if not isinstance(payload, str):
            raise ValueError("invalid float64 hexadecimal value")
        try:
            exact = float.fromhex(payload)
        except (ValueError, OverflowError) as error:
            raise ValueError("invalid float64 hexadecimal value") from error
        if not np.isfinite(exact):
            raise ValueError("float64 values must be finite")
        return exact if raw else arithmetic.scalar(exact)
    if kind == "integer":
        if not isinstance(payload, str):
            raise ValueError("invalid integer metadata")
        try:
            exact = int(payload)
        except ValueError as error:
            raise ValueError("invalid integer metadata") from error
        return exact if raw else arithmetic.scalar(exact)
    if kind == "decimal":
        if not isinstance(payload, str):
            raise ValueError("invalid decimal metadata")
        try:
            exact = Decimal(payload)
        except InvalidOperation as error:
            raise ValueError("invalid decimal metadata") from error
        if not exact.is_finite():
            raise ValueError("decimal metadata must be finite")
        return payload if raw else arithmetic.scalar(payload)
    raise ValueError("unsupported numerical value encoding")


def _metadata(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, (float, np.floating, Decimal)) or hasattr(value, "_mpf_"):
        return _number(value)
    if isinstance(value, (tuple, list)):
        return [_metadata(item) for item in value]
    raise TypeError("solution metadata must contain only portable scalar values and sequences")


def _decode_metadata(value, arithmetic):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return _decode_number(value, arithmetic)
    if isinstance(value, list):
        return tuple(_decode_metadata(item, arithmetic) for item in value)
    raise ValueError("invalid portable solution metadata")


def _projection_info(solution):
    result = []
    seen = set()
    for key, value in getattr(solution, "projection_info", ()):
        if not isinstance(key, str) or key in seen:
            raise ValueError("projection metadata keys must be distinct strings")
        seen.add(key)
        result.append([key, _metadata(value)])
    return result


def _coefficient_kind(epsilon):
    return "constant" if epsilon._function is None else "declared"


def _physical_metadata(physical):
    epsilon = physical.epsilon
    return {
        "delta": _number(physical.delta),
        "boundary_conditions": [[None if value is None else _number(value)
                                 for value in row] for row in physical.boundary_conditions],
        "epsilon": {
            "kind": _coefficient_kind(epsilon),
            "is_identity": bool(epsilon.is_identity),
            "lower_bound": _number(epsilon.lower_bound),
            "upper_bound": _number(epsilon.upper_bound),
            "wall_value": _number(epsilon.wall_value),
            "interfaces": [_number(value) for value in epsilon.interfaces],
        },
    }


def _sample_coordinates(solution):
    if solution.solver_type == "galerkin":
        plan = solution.plan
        if plan is None:
            plan = QuadratureFactory(
                solution.arithmetic, L=solution.L,
                interfaces=solution.physical.epsilon.interfaces,
            ).plan(solution.N+1)
        return plan.physical_nodes
    left, right = solution.physical.domain
    return (left, solution.L) if not solution.arithmetic.isfinite(right) else (left, right/2, right)


def _raw_length(solution):
    if solution.solver_type == "galerkin":
        return solution.basis.mapping.raw_L
    return getattr(solution, "raw_L", solution.L)


def _quadrature_metadata(plan):
    if plan is None:
        return None
    plan.check_precision()
    return {
        "order": int(plan.order),
        "subdivisions": int(plan.subdivisions),
        "raw_interfaces": [_number(value) for value in plan.raw_interfaces],
        "panels": [[_number(left), _number(right)] for left, right in plan.panels],
    }


def save_solution(solution, path):
    """Write a data-only snapshot; callbacks must be supplied again on load."""
    solution.check_precision()
    solver = solution.solver_type
    if solver not in BASIS_IDS:
        raise ValueError("unsupported solver type")
    arithmetic, physical = solution.arithmetic, solution.physical
    if not isinstance(physical, BoundFluxProblem) or physical.arithmetic is not arithmetic:
        raise TypeError("solution must contain its bound physical problem")
    fields = {"u": [_number(value) for value in solution.u_coefficients]}
    if solver == "galerkin":
        fields["w"] = [_number(value) for value in solution.w_coefficients]
    samples = [{"x": _number(x), "value": _number(physical.epsilon(x))}
               for x in _sample_coordinates(solution)]
    nonlinear = getattr(solution, "nonlinear_converged", solution.converged)
    record = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "solver_type": solver,
        "basis": {"id": BASIS_IDS[solver], "version": 1, "N": int(solution.N),
                  "unknown_count": int(solution.unknown_count)},
        "arithmetic": {
            "backend": "mpmath" if arithmetic.use_mpmath else "numpy",
            "implementation": arithmetic.backend,
            "precision_bits": arithmetic.context.prec if arithmetic.use_mpmath else 53,
            "precision_dps": int(arithmetic.dps),
        },
        "geometry": {"L": _number(solution.L), "raw_L": _number(_raw_length(solution)),
                     "domain": [_number(value) for value in physical.domain]},
        "physical": _physical_metadata(physical),
        "epsilon_samples": samples,
        "callback_verification": "coefficient_samples_only",
        "coefficients": fields,
        "quadrature": _quadrature_metadata(getattr(solution, "plan", None)),
        "status": {
            "method": solution.method, "converged": bool(solution.converged),
            "nonlinear_converged": bool(nonlinear),
            "quadrature_converged": (bool(solution.quadrature_converged)
                                      if solver == "galerkin" else None),
        },
        "projection_info": _projection_info(solution),
    }
    # Complete encoding before opening the destination, so unsupported values
    # cannot replace an existing file with a partially encoded snapshot.
    text = json.dumps(record, indent=2, allow_nan=False)+"\n"
    Path(path).write_text(text, encoding="utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field: {}".format(key))
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonstandard JSON numerical constant: {}".format(value))


def _read_record(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "load_solution reads JSON snapshots only; use the existing trusted "
            "collocation load_data reader for legacy coefficient files, then import explicitly"
        ) from error


def _saved_arithmetic(record):
    _keys(record, ("backend", "implementation", "precision_bits", "precision_dps"), "arithmetic")
    bits = _integer(record["precision_bits"], "working bit precision", 1)
    digits = _integer(record["precision_dps"], "working decimal precision", 1)
    if record["backend"] == "numpy":
        if bits != 53 or digits != 15 or record["implementation"] != "numpy":
            raise ValueError("invalid NumPy arithmetic metadata")
        return NumpyArithmetic()
    if record["backend"] != "mpmath" or record["implementation"] not in ("python", "gmpy"):
        raise ValueError("unsupported arithmetic backend")
    arithmetic = MpmathArithmetic(digits)
    arithmetic.context.prec = bits
    if arithmetic.dps != digits:
        raise ValueError("working bit and decimal precision metadata disagree")
    return arithmetic


def _decode_sequence(values, arithmetic, name, allow_infinity=False):
    if not isinstance(values, list):
        raise ValueError("{} must be a numerical list".format(name))
    return tuple(_decode_number(value, arithmetic, allow_infinity=allow_infinity) for value in values)


def _check_physical(record, physical, arithmetic):
    _keys(record, ("delta", "boundary_conditions", "epsilon"), "physical problem")
    if _decode_number(record["delta"], arithmetic) != physical.delta:
        raise ValueError("saved delta does not match the supplied problem")
    rows = record["boundary_conditions"]
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("invalid saved boundary conditions")
    boundaries = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 5:
            raise ValueError("invalid saved boundary conditions")
        boundaries.append(tuple(None if value is None else _decode_number(value, arithmetic)
                                for value in row))
    if tuple(boundaries) != physical.boundary_conditions:
        raise ValueError("saved boundary data do not match the supplied problem")
    epsilon = record["epsilon"]
    _keys(epsilon, ("kind", "is_identity", "lower_bound", "upper_bound", "wall_value", "interfaces"),
          "coefficient")
    if (epsilon["kind"] != _coefficient_kind(physical.epsilon)
            or not isinstance(epsilon["is_identity"], bool)
            or epsilon["is_identity"] != physical.epsilon.is_identity):
        raise ValueError("saved coefficient description does not match the supplied problem")
    for field in ("lower_bound", "upper_bound", "wall_value"):
        if _decode_number(epsilon[field], arithmetic) != getattr(physical.epsilon, field):
            raise ValueError("saved epsilon {} does not match the supplied problem".format(field))
    if _decode_sequence(epsilon["interfaces"], arithmetic, "epsilon interfaces") != physical.epsilon.interfaces:
        raise ValueError("saved epsilon interfaces do not match the supplied problem")


def _restore_plan(record, basis):
    if record is None:
        return None
    _keys(record, ("order", "subdivisions", "raw_interfaces", "panels"), "quadrature")
    order = _integer(record["order"], "quadrature order", basis.N+1)
    subdivisions = _integer(record["subdivisions"], "quadrature subdivisions", 1)
    interfaces = record["raw_interfaces"]
    if not isinstance(interfaces, list):
        raise ValueError("quadrature raw interfaces must be a list")
    raw_interfaces = tuple(_decode_number(value, basis.arithmetic, raw=True) for value in interfaces)
    plan = QuadratureFactory(
        basis.arithmetic, L=basis.mapping.raw_L, interfaces=raw_interfaces,
    ).plan(order, subdivisions)
    panels = record["panels"]
    if not isinstance(panels, list):
        raise ValueError("invalid saved quadrature panels")
    decoded = tuple(_decode_sequence(panel, basis.arithmetic, "quadrature panel") for panel in panels)
    if decoded != plan.panels:
        raise ValueError("saved quadrature panels disagree with its geometry")
    return plan


def _check_samples(record, physical, arithmetic, expected_coordinates):
    if not isinstance(record, list) or len(record) != len(expected_coordinates):
        raise ValueError("invalid saved coefficient samples")
    for sample, coordinate in zip(record, expected_coordinates):
        _keys(sample, ("x", "value"), "coefficient sample")
        x = _decode_number(sample["x"], arithmetic)
        if x != coordinate:
            raise ValueError("saved coefficient sample coordinates disagree with its quadrature")
        expected = _decode_number(sample["value"], arithmetic)
        if physical.epsilon(x) != expected:
            raise ValueError("supplied epsilon callback disagrees with saved coefficient samples")


def load_solution(path, *, problem, solver=None, use_mpmath=None, precision=None):
    """Restore fields without solving, checking the supplied problem first.

    ``solver`` is an optional expected solver name, never a conversion request.
    A collocation snapshot cannot load as Galerkin, and vice versa. Conversion
    requires the explicit collocation-to-Galerkin projection API.

    By default the saved backend and exact bit precision are restored. Explicit
    ``use_mpmath``/``precision`` overrides can round coefficients and metadata;
    compatibility is checked in the saved arithmetic before that conversion.
    Changing backend or working bits clears convergence flags. Exact restores
    retain the saved flags as provenance, not as a new residual certificate.
    Source callbacks are supplied by ``problem`` and are not identity-certified.
    """
    record = _read_record(path)
    _keys(record, ("format", "format_version", "solver_type", "basis", "arithmetic", "geometry",
                   "physical", "epsilon_samples", "callback_verification", "coefficients",
                   "quadrature", "status", "projection_info"), "solution")
    if (record["format"] != FORMAT or type(record["format_version"]) is not int
            or record["format_version"] != FORMAT_VERSION):
        raise ValueError("unsupported solution format or format version")
    saved_solver = record["solver_type"]
    if not isinstance(saved_solver, str) or saved_solver not in BASIS_IDS:
        raise ValueError("unsupported saved solver type")
    if solver is not None and solver != saved_solver:
        raise ValueError("saved solver type is incompatible; use explicit projection for conversion")
    if isinstance(problem, BoundFluxProblem) or not callable(getattr(problem, "bind", None)):
        raise TypeError("supply the original unbound physical problem description")
    source = _saved_arithmetic(record["arithmetic"])
    physical = problem.bind(source)
    geometry = record["geometry"]
    _keys(geometry, ("L", "raw_L", "domain"), "geometry")
    length = _decode_number(geometry["L"], source)
    raw_length = _decode_number(geometry["raw_L"], source, raw=True)
    if length <= 0 or source.scalar(raw_length) != length:
        raise ValueError("saved mapping length is invalid or inconsistent")
    domain = _decode_sequence(geometry["domain"], source, "domain", allow_infinity=True)
    if len(domain) != 2 or domain != physical.domain:
        raise ValueError("saved domain does not match the supplied problem")
    _check_physical(record["physical"], physical, source)
    if record["callback_verification"] != "coefficient_samples_only":
        raise ValueError("unsupported callback verification metadata")
    basis_record = record["basis"]
    _keys(basis_record, ("id", "version", "N", "unknown_count"), "basis")
    if (basis_record["id"] != BASIS_IDS[saved_solver]
            or type(basis_record["version"]) is not int or basis_record["version"] != 1):
        raise ValueError("saved basis does not match its solver type")
    N = _integer(basis_record["N"], "basis N", 2 if saved_solver == "galerkin" else 1)
    unknowns = _integer(basis_record["unknown_count"], "unknown count", 1)
    fields = record["coefficients"]
    _keys(fields, ("u", "w") if saved_solver == "galerkin" else ("u",), "coefficient fields")
    u = _decode_sequence(fields["u"], source, "u coefficients")
    w = _decode_sequence(fields["w"], source, "w coefficients") if saved_solver == "galerkin" else ()
    expected = 2*N-1 if saved_solver == "galerkin" else N+sum(
        value is not None for row in physical.boundary_conditions for value in row
    )
    if (unknowns != expected or len(u)+len(w) != unknowns
            or (saved_solver == "galerkin" and (len(u) != N-1 or len(w) != N))):
        raise ValueError("coefficient counts disagree with the saved discretization")
    expected_kind = "mpf" if source.use_mpmath else "float64"
    for value in fields["u"] + (fields["w"] if saved_solver == "galerkin" else []):
        if value["type"] != expected_kind:
            raise ValueError("coefficient encoding disagrees with the saved arithmetic")
        if source.use_mpmath and value["value"][3] > source.context.prec:
            raise ValueError("coefficient mantissa exceeds the saved working precision")
    status = record["status"]
    _keys(status, ("method", "converged", "nonlinear_converged", "quadrature_converged"), "status")
    methods = ("newton", "projection") if saved_solver == "galerkin" else (
        "newton", "root", "root_no_jac", "nonlin_solve"
    )
    if status["method"] not in methods or any(not isinstance(status[field], bool)
                                             for field in ("converged", "nonlinear_converged")):
        raise ValueError("invalid saved method or convergence flags")
    if ((saved_solver == "galerkin" and not isinstance(status["quadrature_converged"], bool))
            or (saved_solver == "collocation" and status["quadrature_converged"] is not None)):
        raise ValueError("invalid quadrature convergence flag")
    projection = record["projection_info"]
    if not isinstance(projection, list):
        raise ValueError("invalid projection metadata")
    projection_info, seen = [], set()
    for pair in projection:
        if (not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str)
                or pair[0] in seen):
            raise ValueError("invalid projection metadata entry")
        seen.add(pair[0])
        projection_info.append((pair[0], _decode_metadata(pair[1], source)))
    if saved_solver == "collocation" and projection_info:
        raise ValueError("collocation snapshots cannot contain mixed projection metadata")
    if saved_solver == "galerkin":
        basis = GalerkinBasis(N, source, L=raw_length)
        plan = _restore_plan(record["quadrature"], basis)
        from overscreening_breakdown.galerkin.solution import GalerkinSolution
        source_solution = GalerkinSolution(
            basis=basis, physical=physical, coefficients=source.vector(u+w), plan=plan,
            method=status["method"], converged=status["converged"],
            nonlinear_converged=status["nonlinear_converged"],
            quadrature_converged=status["quadrature_converged"], projection_info=projection_info,
        )
    else:
        if record["quadrature"] is not None:
            raise ValueError("collocation snapshots cannot contain mixed quadrature metadata")
        from overscreening_breakdown.collocation.solution import CollocationSolution
        source_solution = CollocationSolution(
            physical=physical, coefficients=source.vector(u), N=N, L=raw_length,
            method=status["method"], converged=status["converged"],
            nonlinear_converged=status["nonlinear_converged"],
        )
    _check_samples(record["epsilon_samples"], physical, source, _sample_coordinates(source_solution))
    if use_mpmath is not None and not isinstance(use_mpmath, bool):
        raise ValueError("use_mpmath must be a bool or None")
    target_mp = source.use_mpmath if use_mpmath is None else use_mpmath
    if precision is not None:
        precision = _integer(precision, "requested decimal precision", 1)
        if not target_mp:
            raise ValueError("precision requires use_mpmath=True")
    if target_mp == source.use_mpmath and precision is None:
        return source_solution
    if target_mp:
        target = MpmathArithmetic(source.dps if precision is None else precision)
        if precision is None:
            target.context.prec = source.context.prec if source.use_mpmath else 53
    else:
        target = NumpyArithmetic()
    changed = (target.use_mpmath != source.use_mpmath
               or (target.use_mpmath and target.context.prec != source.context.prec))
    converged = status["converged"] and not changed
    nonlinear_converged = status["nonlinear_converged"] and not changed
    target_physical = problem.bind(target)
    coefficients = target.vector(u+w)
    if saved_solver == "galerkin":
        target_basis = GalerkinBasis(N, target, L=raw_length)
        # A changed arithmetic reconstructs the rule from raw geometry instead
        # of requiring its newly rounded panels to equal the saved panels.
        target_plan = None
        if record["quadrature"] is not None:
            quad = record["quadrature"]
            interfaces = tuple(_decode_number(value, target, raw=True) for value in quad["raw_interfaces"])
            target_plan = QuadratureFactory(target, L=raw_length, interfaces=interfaces).plan(
                quad["order"], quad["subdivisions"]
            )
        return GalerkinSolution(
            basis=target_basis, physical=target_physical, coefficients=coefficients, plan=target_plan,
            method=status["method"], converged=converged,
            nonlinear_converged=nonlinear_converged,
            quadrature_converged=status["quadrature_converged"] and not changed,
            projection_info=tuple((key, _decode_metadata(value, target)) for key, value in projection),
        )
    return CollocationSolution(
        physical=target_physical, coefficients=coefficients, N=N, L=raw_length,
        method=status["method"], converged=converged,
        nonlinear_converged=nonlinear_converged,
    )
