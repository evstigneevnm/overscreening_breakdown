"""Residual certification for NumPy collocation results returned by SciPy."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ScipySolveResult:
    """Separate SciPy termination from the independently verified root.

    ``nfev`` and ``njev`` are SciPy's reported evaluation counts, excluding
    the extra residual evaluation used here. Residual norms include every
    assembled equation, including tau boundary rows. ``target_norm`` is the
    selected RMS residual threshold, not SciPy's step tolerance.
    """

    method: str
    converged: bool
    reason: str
    scipy_success: bool
    status: Optional[int]
    message: str
    nfev: Optional[int]
    njev: Optional[int]
    residual_norm: np.float64
    residual_max: np.float64
    target_norm: np.float64
    homotopy: np.float64 = np.float64(0)


def _vector(values, name, shape=None):
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("float64"):
        raise TypeError("{} must be a NumPy float64 vector".format(name))
    if values.ndim != 1 or not len(values) or (shape is not None and values.shape != shape):
        raise ValueError("{} must be a nonempty vector matching the coefficient shape".format(name))
    return values


def verify_scipy_result(result, *, residual, target_norm, method):
    """Certify a finite q=0 residual independently of SciPy's success flag.

    ``residual(state)`` must explicitly evaluate the full zero-homotopy
    discrete system. Its callback receives a copy, and cached ``result.fun``
    is never read. Nonfinite states/residuals and numerical overflow return
    failed diagnostics; malformed arrays and callback programming errors
    propagate. Arbitrary-precision vectors are not narrowed to float64.
    """
    if method not in ("root", "root_no_jac", "nonlin_solve"):
        raise ValueError("SciPy verification requires root, root_no_jac, or nonlin_solve")
    if (isinstance(target_norm, (bool, np.bool_)) or np.ndim(target_norm) != 0
            or np.iscomplexobj(target_norm) or hasattr(target_norm, "_mpf_")):
        raise ValueError("target_norm must be a finite nonnegative float64 scalar")
    target_norm = np.float64(target_norm)
    if not np.isfinite(target_norm) or target_norm < 0:
        raise ValueError("target_norm must be finite and nonnegative")
    state = _vector(result.x, "SciPy coefficients")
    infinity = np.float64("inf")

    def optional_integer(name):
        value = getattr(result, name, None)
        return None if value is None else int(value)

    def finish(reason, rms=infinity, maximum=infinity):
        return ScipySolveResult(
            method=method, converged=reason == "converged", reason=reason,
            scipy_success=bool(getattr(result, "success", False)),
            status=optional_integer("status"), message=str(getattr(result, "message", "")),
            nfev=optional_integer("nfev"), njev=optional_integer("njev"),
            residual_norm=np.float64(rms), residual_max=np.float64(maximum),
            target_norm=target_norm,
        )

    if not np.all(np.isfinite(state)):
        return finish("nonfinite_state")
    try:
        values = residual(state.copy())
    except OverflowError:
        return finish("residual_overflow")
    values = _vector(values, "SciPy verification residual", state.shape)
    if not np.all(np.isfinite(values)):
        return finish("nonfinite_residual")
    maximum = np.max(np.abs(values))
    if maximum == 0:
        rms = np.float64(0)
    else:
        # Scale before squaring: the RMS never exceeds the largest finite
        # entry, even when the unscaled Euclidean norm would overflow. Terms
        # that underflow after this scaling cannot affect float64's sum.
        with np.errstate(under="ignore"):
            scaled = values/maximum
            rms = maximum*np.sqrt(np.mean(scaled*scaled, dtype=np.float64))
    return finish("converged" if rms <= target_norm else "residual_too_large", rms, maximum)
