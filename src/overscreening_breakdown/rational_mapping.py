"""Context-bound rational map between [-1, 1] and the physical half-line.

The wall is y=-1 and infinity is y=1. Derivatives and quadrature Jacobians
refer to the unweighted physical measure dx. A map snapshots its precision;
construct a new map from raw_L if its arithmetic context changes precision.
"""

from dataclasses import dataclass, field
from typing import Any

from overscreening_breakdown.arithmetic import arithmetic_signature, require_precision


@dataclass(frozen=True, init=False)
class RationalHalfLineMap:
    arithmetic: Any = field(repr=False, compare=False)
    L: Any
    raw_L: Any = field(repr=False)
    precision_signature: tuple
    mapping_type = "rational_half_line"

    @property
    def domain(self):
        self.check_precision()
        return (self.arithmetic.scalar(0), self.arithmetic.scalar(self.arithmetic.inf))

    @property
    def raw_domain(self):
        return (0, "inf")

    def __init__(self, arithmetic, L=1):
        length = arithmetic.scalar(L)
        if not arithmetic.isfinite(length) or length <= 0:
            raise ValueError("mapping length L must be finite and strictly positive")
        object.__setattr__(self, "arithmetic", arithmetic)
        object.__setattr__(self, "L", length)
        object.__setattr__(self, "raw_L", L)
        object.__setattr__(self, "precision_signature", arithmetic_signature(arithmetic))

    def check_precision(self):
        require_precision(self.arithmetic, self.precision_signature,
                          rebuild="rebuild the rational map or plan")

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

    def to_angle(self, x):
        """Return ``t`` in [0, pi] with ``x = L cot(t/2)**2``.

        This evaluates the half-angle directly, without first rounding the
        reference coordinate to an endpoint.  At finite precision an angle
        extremely close to pi can still round to pi.
        """
        x = self.physical_coordinate(x)
        arithmetic = self.arithmetic
        zero, two = arithmetic.scalar(0), arithmetic.scalar(2)
        pi = arithmetic.scalar(arithmetic.pi)
        if x == zero:
            return pi
        if x == arithmetic.inf:
            return zero
        if x == self.L:
            return pi/two
        return two*arithmetic.atan2(arithmetic.sqrt(self.L), arithmetic.sqrt(x))

    def from_angle(self, t):
        """Return ``x = L cot(t/2)**2`` without squaring an extreme ratio.

        The exact endpoint conventions are t=0 -> infinity and t=pi -> zero.
        An interior angle whose physical value overflows the selected arithmetic
        raises ``OverflowError``; representational underflow can yield zero.
        """
        self.check_precision()
        arithmetic = self.arithmetic
        t = arithmetic.scalar(t)
        zero, two = arithmetic.scalar(0), arithmetic.scalar(2)
        pi = arithmetic.scalar(arithmetic.pi)
        if not arithmetic.isfinite(t) or not zero <= t <= pi:
            raise ValueError("angle must be finite and in [0, pi]")
        if t == zero:
            return arithmetic.inf
        if t == pi:
            return zero
        midpoint = pi/two
        if t == midpoint:
            return self.L
        sine, cosine = arithmetic.sin(t/two), arithmetic.cos(t/two)
        if not arithmetic.use_mpmath:
            tiny = arithmetic.context.finfo(arithmetic.context.float64).tiny
            root_tiny = arithmetic.sqrt(tiny)
            if sine >= root_tiny and cosine >= root_tiny:
                sine_squared, cosine_squared = sine*sine, cosine*cosine
                if self.L >= 1 or cosine_squared >= tiny/self.L:
                    # Preserve the legacy operation order when all intermediate
                    # values are normal. One-ulp changes to collocation nodes
                    # can affect finite-difference nonlinear solver decisions.
                    return self._finite_result(
                        (self.L*cosine_squared)/sine_squared,
                        "physical coordinate",
                    )
        if t < midpoint:
            ratio = sine/cosine
            if ratio == zero:
                raise OverflowError("angle half-ratio underflowed in the selected arithmetic")
            value = (self.L/ratio)/ratio
        else:
            ratio = cosine/sine
            value = (self.L*ratio)*ratio
        return self._finite_result(value, "physical coordinate")

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
