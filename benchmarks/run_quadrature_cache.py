"""Measure first quadrature construction and reuse of one factory's cached plan.

Run from an installed checkout with ``python -m benchmarks.run_quadrature_cache``.
Every repetition creates a fresh arithmetic owner and factory. The cached batch
then reuses precisely that owner, geometry, order and subdivisions; it does not
measure repeated public solves or reset operating-system caches.
"""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter
import traceback

import mpmath as mp

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.quadrature import QuadratureFactory


def _positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arithmetic", choices=("numpy", "mpmath"), default="numpy")
    result.add_argument("--precision", type=_positive_integer, default=50,
                        help="decimal digits for mpmath; NumPy always uses float64")
    result.add_argument("--backend", choices=("auto", "python", "gmpy"), default="auto",
                        help="verify the process mpmath backend, without changing it")
    result.add_argument("--L", default="1", help="positive half-line mapping length")
    result.add_argument("--R", help="positive right endpoint; selects the finite interval [0,R]")
    result.add_argument("--interfaces", nargs="*", default=[],
                        help="strictly increasing physical split positions, as decimal strings")
    result.add_argument("--order", type=_positive_integer, default=12,
                        help="Gauss-Legendre nodes per panel")
    result.add_argument("--subdivisions", type=_positive_integer, default=1)
    result.add_argument("--lookups", type=_positive_integer, default=1000,
                        help="cached lookups in each timed batch, including identity checks")
    result.add_argument("--repetitions", type=_positive_integer, default=3)
    result.add_argument("--output-dir", type=Path, default=Path("benchmark-results/quadrature-cache"))
    return result


def _validate(options, argument_parser):
    if options.backend != "auto" and options.backend != mp.libmp.BACKEND:
        argument_parser.error("requested mpmath backend {} but this process uses {}".format(
            options.backend, mp.libmp.BACKEND))
    try:
        length = Decimal(options.L)
        right = Decimal(options.R) if options.R is not None else Decimal("Infinity")
        interfaces = [Decimal(value) for value in options.interfaces]
    except (InvalidOperation, ValueError) as error:
        argument_parser.error("geometry must contain decimal numbers: {}".format(error))
    if not length.is_finite() or length <= 0:
        argument_parser.error("L must be finite and strictly positive")
    if options.R is not None and (not right.is_finite() or right <= 0):
        argument_parser.error("R must be finite and strictly positive")
    previous = Decimal(0)
    for position in interfaces:
        if not position.is_finite() or not previous < position < right:
            argument_parser.error("interfaces must be finite, strictly increasing and inside the domain")
        previous = position


def _trial(options, *, clock=perf_counter):
    """Keep preparation, first build and actual cache hits as separate intervals."""
    started = clock()
    arithmetic = (MpmathArithmetic(options.precision) if options.arithmetic == "mpmath"
                  else NumpyArithmetic())
    owner_seconds = clock() - started
    started = clock()
    factory = QuadratureFactory(
        arithmetic, L=options.L, interfaces=options.interfaces,
        domain=("0", options.R) if options.R is not None else None,
    )
    factory_seconds = clock() - started
    started = clock()
    plan = factory.plan(options.order, subdivisions=options.subdivisions)
    plan_build_seconds = clock() - started
    started = clock()
    for _ in range(options.lookups):
        if factory.plan(options.order, subdivisions=options.subdivisions) is not plan:
            raise RuntimeError("identical quadrature requests did not reuse the cached plan")
    cached_lookup_seconds = clock() - started
    plan.check_precision()
    return {
        "success": True,
        "owner_seconds": owner_seconds,
        "factory_seconds": factory_seconds,
        "plan_build_seconds": plan_build_seconds,
        "cached_lookup_seconds": cached_lookup_seconds,
        "cached_lookup_seconds_per_call": cached_lookup_seconds / options.lookups,
        "cache_identity_verified": True,
        "lookup_count": options.lookups,
        "panel_count": len(plan.panels),
        "node_count": len(plan.nodes),
        "precision_bits": arithmetic.context.prec if arithmetic.use_mpmath else 53,
        "precision_dps": int(arithmetic.dps),
        "arithmetic_backend": arithmetic.backend,
    }


def _versions():
    result = {}
    for name in ("overscreening-breakdown", "numpy", "mpmath", "gmpy2"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def main(argv=None):
    argument_parser = parser()
    options = argument_parser.parse_args(argv)
    _validate(options, argument_parser)
    records = []
    for repetition in range(options.repetitions):
        try:
            row = _trial(options)
        except Exception as error:
            row = {"success": False, "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc()}
        row["repetition"] = repetition
        records.append(row)
    successful = [row for row in records if row["success"]]
    timing_fields = ("owner_seconds", "factory_seconds", "plan_build_seconds",
                     "cached_lookup_seconds", "cached_lookup_seconds_per_call")
    summary = {"attempted": len(records), "successful": len(successful),
               "failed": len(records)-len(successful), "timings": {}}
    for field in timing_fields:
        values = [row[field] for row in successful]
        summary["timings"][field] = ({"median": median(values), "minimum": min(values),
                                     "maximum": max(values)} if values else None)
    report = {
        "format_version": 1,
        "measurement": "quadrature_factory_cache",
        "cache_semantics": (
            "Each repetition creates a new arithmetic owner and factory. The first plan "
            "build starts with empty factory caches. Its cached batch reuses the same owner, "
            "factory, precision, geometry, order and subdivisions, checking object identity "
            "on every lookup. Timings use a monotonic wall clock; lookup timings include "
            "loop and identity-check overhead. Process startup/imports and export are excluded. "
            "Operating-system and library caches are not reset. This is not a repeated solve."
        ),
        "metadata": {"created_utc": datetime.now(timezone.utc).isoformat(),
                     "python": platform.python_version(), "platform": platform.platform(),
                     "versions": _versions(), "mpmath_backend": mp.libmp.BACKEND,
                     "command": list(argv) if argv is not None else sys.argv[1:]},
        "configuration": {"arithmetic": options.arithmetic,
                          "requested_precision": options.precision,
                          "domain": ["0", options.R if options.R is not None else "inf"],
                          "mapping": "affine" if options.R is not None else "rational_half_line",
                          "L": options.L if options.R is None else None,
                          "interfaces": options.interfaces, "order": options.order,
                          "subdivisions": options.subdivisions, "lookups": options.lookups,
                          "repetitions": options.repetitions},
        "records": records, "summary": summary,
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    destination = options.output_dir / "quadrature-cache.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Quadrature cache study: {}/{} successful; {}".format(
        len(successful), len(records), destination))
    return int(len(successful) != len(records))


if __name__ == "__main__":
    raise SystemExit(main())
