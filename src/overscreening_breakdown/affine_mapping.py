"""Owned-arithmetic geometry for a finite physical interval.

Reference coordinates increase from -1 at the left endpoint to 1 at the
right endpoint. Chebyshev angles retain the collocation convention y=cos(t):
t=0 is the right endpoint and t=pi is the left endpoint.
"""

from dataclasses import dataclass, field
from typing import Any

from overscreening_breakdown.rational_mapping import arithmetic_signature


@dataclass(frozen=True, init=False)
class AffineIntervalMap:
    arithmetic: Any = field(repr=False, compare=False)
    domain: tuple
    raw_domain: tuple = field(repr=False)
    left: Any
    right: Any
    length: Any
    precision_signature: tuple
    mapping_type = "affine_interval"

    def __init__(self, arithmetic, domain):
        try:
            raw = tuple(domain)
        except TypeError as error:
            raise ValueError("finite domain must contain two endpoints") from error
        if len(raw) != 2:
            raise ValueError("finite domain must contain two endpoints")
        left, right = (arithmetic.scalar(value) for value in raw)
        if not arithmetic.isfinite(left) or not arithmetic.isfinite(right) or left >= right:
            raise ValueError("finite endpoints must be distinct and increasing at the selected precision")
        length = right-left
        if not arithmetic.isfinite(length) or length <= 0:
            raise ValueError("finite interval length must remain positive and finite")
        for name, value in (
            ("arithmetic", arithmetic), ("raw_domain", raw), ("domain", (left, right)),
            ("left", left), ("right", right), ("length", length),
            ("precision_signature", arithmetic_signature(arithmetic)),
        ):
            object.__setattr__(self, name, value)

    def check_precision(self):
        if arithmetic_signature(self.arithmetic) != self.precision_signature:
            raise RuntimeError("arithmetic precision changed; rebuild the affine map or plan")

    def reference_coordinate(self, y):
        self.check_precision()
        y = self.arithmetic.scalar(y)
        if not self.arithmetic.isfinite(y) or not -1 <= y <= 1:
            raise ValueError("reference coordinate must be finite and in [-1, 1]")
        return y

    def physical_coordinate(self, x):
        self.check_precision()
        x = self.arithmetic.scalar(x)
        if not self.arithmetic.isfinite(x) or not self.left <= x <= self.right:
            raise ValueError("physical coordinate lies outside the finite interval")
        return x

    def _sum_is_safe(self, first, second):
        if self.arithmetic.use_mpmath or first == 0 or second == 0 or (first < 0) != (second < 0):
            return True
        largest = self.arithmetic.context.finfo(self.arithmetic.context.float64).max
        return abs(first) <= largest-abs(second)

    def to_reference(self, x):
        x = self.physical_coordinate(x)
        if x == self.left:
            return self.arithmetic.scalar(-1)
        if x == self.right:
            return self.arithmetic.scalar(1)
        # Preserve the legacy finite collocation operation order in its
        # ordinary range. Direct distances avoid overflowing large endpoints.
        if self._sum_is_safe(self.left, self.right) and self._sum_is_safe(x, x):
            return (2*x-(self.right+self.left))/self.length
        return 2*((x-self.left)/self.length)-1

    def to_physical(self, y):
        y = self.reference_coordinate(y)
        if y == -1:
            return self.left
        if y == 1:
            return self.right
        term = self.length*y
        if self._sum_is_safe(self.left, self.right):
            total = self.right+self.left
            if self._sum_is_safe(term, total):
                return (term+total)/2
        return self.left*((1-y)/2)+self.right*((1+y)/2)

    def jacobian(self, y):
        self.reference_coordinate(y)
        value = self.length/2
        if value <= 0 or not self.arithmetic.isfinite(value):
            raise OverflowError("affine Jacobian is not finite and positive")
        return value

    def inverse_jacobian(self, y):
        self.reference_coordinate(y)
        value = self.arithmetic.scalar(2)/self.length
        if value <= 0 or not self.arithmetic.isfinite(value):
            raise OverflowError("inverse affine Jacobian is not finite and positive")
        return value

    def to_angle(self, x):
        x = self.physical_coordinate(x)
        arithmetic = self.arithmetic
        pi = arithmetic.scalar(arithmetic.pi)
        if x == self.left:
            return pi
        if x == self.right:
            return arithmetic.scalar(0)
        y = self.to_reference(x)
        if -1 < y < 1:
            inverse_cosine = arithmetic.context.acos if arithmetic.use_mpmath else arithmetic.context.arccos
            return inverse_cosine(y)
        # A finite endpoint distance can survive after y rounds to +/-1.
        return 2*arithmetic.atan2(arithmetic.sqrt(self.right-x), arithmetic.sqrt(x-self.left))

    def from_angle(self, t):
        self.check_precision()
        arithmetic = self.arithmetic
        t = arithmetic.scalar(t)
        pi = arithmetic.scalar(arithmetic.pi)
        if not arithmetic.isfinite(t) or not 0 <= t <= pi:
            raise ValueError("angle must be finite and in [0, pi]")
        if t == 0:
            return self.right
        if t == pi:
            return self.left
        value = self.to_physical(arithmetic.cos(t))
        if value == self.left:
            return self.left+self.length*arithmetic.cos(t/2)**2
        if value == self.right:
            return self.right-self.length*arithmetic.sin(t/2)**2
        return value
