"""Real arithmetic owned by an individual solver or bound physical problem.

Use decimal strings, integers, or ``Decimal`` values for parameters whose
decimal precision matters. A Python float already contains a rounded binary
value; selecting mpmath cannot recover its original decimal spelling.

Each ``MpmathArithmetic`` owns a cloned context. Its public ``context`` is an
escape hatch for additional mpmath operations; changing that context's
precision changes this arithmetic, but never another owner's context or the
global ``mpmath.mp`` context.
"""

from decimal import Decimal
from numbers import Complex, Integral, Real

import mpmath as mp
import numpy as np


def _require_real_scalar(value):
    if (np.ndim(value) != 0 or np.iscomplexobj(value)
            or (isinstance(value, Complex) and not isinstance(value, Real))
            or hasattr(value, "_mpc_")):
        raise TypeError("A real scalar value is required")


class _RealArithmetic:
    """Elementary functions shared by both real numerical backends."""

    def exp(self, value):
        return self.context.exp(value)

    def sinh(self, value):
        return self.context.sinh(value)

    def cosh(self, value):
        return self.context.cosh(value)

    def tanh(self, value):
        return self.context.tanh(value)

    def sqrt(self, value):
        return self.context.sqrt(value)

    def log(self, value):
        return self.context.log(value)

    def sin(self, value):
        return self.context.sin(value)

    def cos(self, value):
        return self.context.cos(value)

    def atan2(self, y, x):
        """Return the quadrant-aware angle in the owner's arithmetic."""
        if self.use_mpmath:
            return self.context.atan2(y, x)
        return self.context.arctan2(y, x)

    @property
    def pi(self):
        return self.context.pi

    @property
    def inf(self):
        return self.context.inf

    def isfinite(self, value):
        """Return a Python bool for a real scalar."""
        return bool(self.context.isfinite(value))


class NumpyArithmetic(_RealArithmetic):
    """Float64 arithmetic with one-dimensional NumPy vectors."""

    use_mpmath = False
    backend = "numpy"
    dps = np.finfo(np.float64).precision
    context = np

    def scalar(self, value):
        _require_real_scalar(value)
        return np.float64(value)

    def vector(self, values):
        values = list(values)
        if np.ndim(values) != 1:
            raise ValueError("A vector requires a one-dimensional sequence")
        return np.array([self.scalar(value) for value in values], dtype=np.float64)

    def matrix(self, rows):
        if hasattr(rows, "rows") and hasattr(rows, "cols"):
            rows = [[rows[i, j] for j in range(rows.cols)]
                    for i in range(rows.rows)]
        rows = [list(row) for row in rows]
        if not rows:
            return np.empty((0, 0), dtype=np.float64)
        columns = len(rows[0])
        if any(len(row) != columns for row in rows):
            raise ValueError("Matrix rows must have equal length")
        return np.array([[self.scalar(value) for value in row] for row in rows],
                        dtype=np.float64)

    def zeros(self, rows, cols=None):
        """Return a zero vector, or a matrix when ``cols`` is supplied."""
        return np.zeros(rows if cols is None else (rows, cols), dtype=np.float64)

    def norm(self, values):
        """Euclidean vector norm, or Frobenius matrix norm."""
        return np.linalg.norm(values)

    def solve(self, matrix, rhs):
        return np.linalg.solve(matrix, rhs)


class MpmathArithmetic(_RealArithmetic):
    """Arbitrary precision arithmetic with an independent mpmath context.

    Vectors are mpmath column matrices. Construct entries through ``scalar``
    so foreign-context mpf inputs are rounded to the owner's working precision.
    """

    use_mpmath = True
    backend = mp.libmp.BACKEND

    def __init__(self, dps=50):
        if isinstance(dps, bool) or not isinstance(dps, Integral) or dps < 1:
            raise ValueError("dps must be a positive integer")
        self._context = mp.mp.clone()
        self._context.dps = int(dps)

    @property
    def context(self):
        return self._context

    @property
    def dps(self):
        return self._context.dps

    def scalar(self, value):
        _require_real_scalar(value)
        if isinstance(value, Decimal):
            value = str(value)
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            # ctx.convert handles every NumPy floating format without first
            # narrowing it to a Python float. ctx.mpf then applies owner rounding.
            value = self._context.convert(value)
        return self._context.mpf(value)

    def vector(self, values):
        entries = [self.scalar(value) for value in values]
        if not entries:
            return self._context.matrix(0, 1)
        return self._context.matrix(entries)

    def matrix(self, rows):
        if hasattr(rows, "rows") and hasattr(rows, "cols"):
            rows = [[rows[i, j] for j in range(rows.cols)]
                    for i in range(rows.rows)]
        rows = [list(row) for row in rows]
        if not rows:
            return self._context.matrix(0, 0)
        columns = len(rows[0])
        if any(len(row) != columns for row in rows):
            raise ValueError("Matrix rows must have equal length")
        result = self._context.matrix(len(rows), columns)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                result[i, j] = self.scalar(value)
        return result

    def zeros(self, rows, cols=None):
        """Return a zero column vector, or a matrix if ``cols`` is supplied."""
        return self._context.zeros(rows, 1 if cols is None else cols)

    def norm(self, values):
        """Euclidean vector norm, or Frobenius matrix norm."""
        return self._context.norm(values)

    def solve(self, matrix, rhs):
        return self._context.lu_solve(matrix, rhs)
