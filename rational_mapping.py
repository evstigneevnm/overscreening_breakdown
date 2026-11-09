"""Context-bound rational map between [-1, 1] and the physical half-line.

The wall is y=-1 and infinity is y=1. Derivatives and quadrature Jacobians
refer to the unweighted physical measure dx. A map snapshots its precision;
construct a new map from raw_L if its arithmetic context changes precision.
"""

from dataclasses import dataclass, field
from typing import Any


def arithmetic_signature(arithmetic):
    """Identify the numerical backend and exact working precision in bits."""
    bits = arithmetic.context.prec if arithmetic.use_mpmath else 53
    return (arithmetic.use_mpmath, arithmetic.backend, bits)


@dataclass(frozen=True, init=False)
class RationalHalfLineMap:
    arithmetic: Any = field(repr=False, compare=False)
    L: Any
    raw_L: Any = field(repr=False)
    precision_signature: tuple

    def __init__(self, arithmetic, L=1):
        length = arithmetic.scalar(L)
        if not arithmetic.isfinite(length) or length <= 0:
            raise ValueError("mapping length L must be finite and strictly positive")
        object.__setattr__(self, "arithmetic", arithmetic)
        object.__setattr__(self, "L", length)
        object.__setattr__(self, "raw_L", L)
        object.__setattr__(self, "precision_signature", arithmetic_signature(arithmetic))

    def check_precision(self):
        if arithmetic_signature(self.arithmetic) != self.precision_signature:
            raise RuntimeError("arithmetic precision changed; rebuild the rational map or plan")

    def reference_coordinate(self, y):
        self.check_precision()
        y = self.arithmetic.scalar(y)
        if not self.arithmetic.isfinite(y) or not -1 <= y <= 1:
            raise ValueError("reference coordinate must be finite and in [-1, 1]")
        return y

    def physical_coordinate(self, x):
        self.check_precision()
        x = self.arithmetic.scalar(x)
        if x == self.arithmetic.inf:
            return x
        if not self.arithmetic.isfinite(x) or x < 0:
            raise ValueError("physical coordinate must be nonnegative and finite or infinity")
        return x

    def to_reference(self, x):
        x = self.physical_coordinate(x)
        one = self.arithmetic.scalar(1)
        if x == self.arithmetic.inf:
            return one
        # Ratios avoid overflow in x+L for large but representable arguments.
        if x >= self.L:
            ratio = self.L/x
            return (one-ratio)/(one+ratio)
        ratio = x/self.L
        return (ratio-one)/(ratio+one)

    def to_physical(self, y):
        y = self.reference_coordinate(y)
        one = self.arithmetic.scalar(1)
        if y == one:
            return self.arithmetic.inf
        return self._finite_result(self.L*((one+y)/(one-y)), "physical coordinate")

    def jacobian(self, y):
        """Return dx/dy; its value at the infinite endpoint is infinity."""
        y = self.reference_coordinate(y)
        distance = self.arithmetic.scalar(1)-y
        if distance == 0:
            return self.arithmetic.inf
        return self._finite_result((self.L/distance)*(2/distance), "mapping Jacobian")

    def inverse_jacobian(self, y):
        """Return dy/dx evaluated at reference coordinate y."""
        y = self.reference_coordinate(y)
        distance = self.arithmetic.scalar(1)-y
        return self._finite_result((distance*(distance/2))/self.L, "inverse mapping Jacobian")

    def _finite_result(self, value, name):
        if not self.arithmetic.isfinite(value):
            raise OverflowError("{} is not finite in the selected arithmetic".format(name))
        return value
