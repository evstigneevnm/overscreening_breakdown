"""Native RMS-merit geometry for adjoint fallback in collocation.

Tau boundary rows define an affine trace space B*c=b. A direction is projected
onto its homogeneous tangent space, retaining the current nonzero traces. The
differential matrix is not used as a descent metric.
"""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from overscreening_breakdown.arithmetic import arithmetic_signature, require_precision


@dataclass(frozen=True, init=False, eq=False)
class CollocationMeritGeometry:
    """RMS gradient and Euclidean projection onto homogeneous tau traces.

    ``boundary_rows`` is a sequence of rows or a native rectangular matrix;
    rows must be nonzero and independent. ``unknown_count`` is the full square
    collocation system size, including its boundary equations. Normalizing each
    row preserves its nullspace and avoids squaring disparate row magnitudes.
    All matrices and products remain in ``arithmetic``; no inverse is formed.
    """

    arithmetic: Any = field(repr=False, compare=False)
    unknown_count: int
    normalized_boundary_rows: tuple
    precision_signature: tuple
    _gram: Any = field(repr=False, compare=False)

    def __init__(self, arithmetic, boundary_rows, unknown_count):
        if (isinstance(unknown_count, bool) or not isinstance(unknown_count, Integral)
                or unknown_count < 1):
            raise ValueError("unknown_count must be a positive integer")
        if hasattr(boundary_rows, "rows") and hasattr(boundary_rows, "cols"):
            if boundary_rows.cols != unknown_count:
                raise ValueError("Boundary rows must match the collocation coefficient count")
            rows = [[boundary_rows[i, j] for j in range(boundary_rows.cols)]
                    for i in range(boundary_rows.rows)]
        else:
            rows = [list(row) for row in boundary_rows]
        normalized = []
        for row in rows:
            if len(row) != unknown_count:
                raise ValueError("Boundary rows must match the collocation coefficient count")
            values = tuple(arithmetic.scalar(value) for value in row)
            if not all(arithmetic.isfinite(value) for value in values):
                raise ValueError("Boundary rows must be finite")
            scale = max(abs(value) for value in values)
            if scale == 0:
                raise ValueError("Boundary rows must be nonzero")
            scaled = tuple(value/scale for value in values)
            length = arithmetic.sqrt(arithmetic.dot(scaled, scaled))
            normalized.append(tuple(value/length for value in scaled))
        normalized = tuple(normalized)
        gram = arithmetic.matrix([[arithmetic.dot(left, right) for right in normalized]
                                  for left in normalized])
        object.__setattr__(self, "arithmetic", arithmetic)
        object.__setattr__(self, "unknown_count", int(unknown_count))
        object.__setattr__(self, "normalized_boundary_rows", normalized)
        object.__setattr__(self, "precision_signature", arithmetic_signature(arithmetic))
        object.__setattr__(self, "_gram", gram)

    def check_precision(self):
        require_precision(self.arithmetic, self.precision_signature,
                          rebuild="rebuild the collocation merit geometry")

    def dot(self, left, right):
        self.check_precision()
        return self.arithmetic.dot(left, right)

    def merit_gradient(self, residual, jacobian):
        """Gradient of ||R||_2^2/(2*m), before the trace projection."""
        self.check_precision()
        if len(residual) != self.unknown_count:
            raise ValueError("Residual size must match the square collocation system")
        gradient = self.arithmetic.transpose_matvec(jacobian, residual)
        if len(gradient) != self.unknown_count:
            raise ValueError("Jacobian columns must match the collocation coefficient count")
        return gradient/self.arithmetic.scalar(self.unknown_count)

    def adjoint_direction(self, gradient):
        """Return -g+B.T*solve(B*B.T,B*g), with B's rows normalized."""
        self.check_precision()
        gradient = self.arithmetic.vector(gradient)
        if len(gradient) != self.unknown_count:
            raise ValueError("Gradient size must match the collocation coefficient count")
        rows = self.normalized_boundary_rows
        if not rows:
            return -gradient
        rhs = self.arithmetic.vector(self.arithmetic.dot(row, gradient) for row in rows)
        multipliers = self.arithmetic.solve(self._gram, rhs)
        return -gradient+self.arithmetic.transpose_matvec(rows, multipliers)
