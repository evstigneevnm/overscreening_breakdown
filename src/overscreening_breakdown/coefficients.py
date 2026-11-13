"""Positive scalar coefficient descriptions and solver-bound evaluations.

Descriptions retain their original numeric inputs until ``bind(arithmetic)``
converts them in the solver's arithmetic context.  Bounds, interfaces, and the
right wall trace are declarations by the caller: sampling cannot establish
piecewise smoothness, and no coefficient derivatives are required.  Each
evaluation is nevertheless checked against the declared positive bounds.

Frozen descriptions protect their metadata, not mutable state captured by user
callbacks.  Callers are responsible for keeping that state unchanged while a
solver uses the coefficient.
Bound evaluations require the exact recorded owner precision, including for
constants and wall traces; rebind the original description after a change.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Number
from typing import Any, Callable, Tuple

from overscreening_breakdown.arithmetic import arithmetic_signature, require_precision


def _finite_scalar(arithmetic, value, name):
    converted = arithmetic.scalar(value)
    if not arithmetic.isfinite(converted):
        raise ValueError("{} must be finite".format(name))
    return converted


def _exactly_one(value):
    # Test the supplied constant before binding: rounding a nonconstant value
    # to 1 in float64 must not silently permit the legacy collocation solver.
    if isinstance(value, str):
        return Decimal(value) == 1
    return bool(value == 1)


@dataclass(frozen=True)
class _BoundCoefficient:
    arithmetic: Any = field(repr=False, compare=False)
    lower_bound: Any
    upper_bound: Any
    wall_value: Any
    interfaces: Tuple[Any, ...]
    is_identity: bool
    _function: Any = field(default=None, repr=False, compare=False)
    precision_signature: tuple = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "precision_signature", arithmetic_signature(self.arithmetic))

    def check_precision(self):
        require_precision(self.arithmetic, self.precision_signature,
                          subject="coefficient", rebuild="bind the coefficient again")

    def __call__(self, x):
        """Evaluate at a finite ``x >= 0``; at zero use the right wall trace.

        The callback's pointwise value at zero is deliberately ignored.  A
        coefficient specified almost everywhere can have an arbitrary value
        there, whereas the weak boundary load needs its declared right trace.
        """
        self.check_precision()
        x = _finite_scalar(self.arithmetic, x, "coefficient coordinate")
        zero = self.arithmetic.scalar(0)
        if x < zero:
            raise ValueError("coefficient coordinate must be nonnegative")
        if x == zero:
            return self.wall_value
        if self._function is None:
            return self.wall_value
        value = self._function(x, self.arithmetic)
        self.check_precision()
        value = _finite_scalar(self.arithmetic, value, "epsilon(x)")
        if not self.lower_bound <= value <= self.upper_bound:
            raise ValueError("epsilon(x) must lie within its declared bounds")
        return value


@dataclass(frozen=True)
class ConstantCoefficient:
    """A positive constant, retained without float conversion until binding."""

    value: Any = 1

    def bind(self, arithmetic):
        signature = arithmetic_signature(arithmetic)
        value = _finite_scalar(arithmetic, self.value, "constant epsilon")
        if value <= arithmetic.scalar(0):
            raise ValueError("constant epsilon must be strictly positive")
        identity = _exactly_one(self.value)
        require_precision(arithmetic, signature, subject="coefficient binding",
                          rebuild="bind the coefficient again")
        return _BoundCoefficient(
            arithmetic=arithmetic,
            lower_bound=value,
            upper_bound=value,
            wall_value=value,
            interfaces=(),
            is_identity=identity,
        )


@dataclass(frozen=True)
class Coefficient:
    """Declared positive piecewise-smooth epsilon with an explicit callback.

    ``function(x, arithmetic)`` receives a scalar in the bound context.  Use
    its operations and exact constants (integers or decimal strings) for full
    mpmath precision.  ``from_function`` explicitly adapts a scalar
    ``epsilon(x)`` callback when no arithmetic argument is needed.

    Numeric validation is deferred to ``bind`` so decimal input is checked at
    the solver's precision.  ``wall_value`` is the right trace at x=0, and
    ``interfaces`` lists every positive interface in strictly increasing
    order.  No regularity or trace limit is inferred from callback samples.
    """

    function: Callable = field(repr=False)
    lower_bound: Any
    upper_bound: Any
    wall_value: Any
    interfaces: Tuple[Any, ...] = ()

    def __init__(
        self, function, *, lower_bound, upper_bound, wall_value, interfaces=()
    ):
        if not callable(function):
            raise TypeError("coefficient function must be callable")
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "upper_bound", upper_bound)
        object.__setattr__(self, "wall_value", wall_value)
        object.__setattr__(self, "interfaces", tuple(interfaces))

    @classmethod
    def from_function(
        cls, function, *, lower_bound, upper_bound, wall_value, interfaces=()
    ):
        """Adapt an explicitly one-argument scalar ``epsilon(x)`` callback."""
        if not callable(function):
            raise TypeError("coefficient function must be callable")

        def contextual_function(x, arithmetic):
            return function(x)

        return cls(
            contextual_function,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            wall_value=wall_value,
            interfaces=interfaces,
        )

    def bind(self, arithmetic):
        signature = arithmetic_signature(arithmetic)
        zero = arithmetic.scalar(0)
        lower = _finite_scalar(arithmetic, self.lower_bound, "lower epsilon bound")
        upper = _finite_scalar(arithmetic, self.upper_bound, "upper epsilon bound")
        wall = _finite_scalar(arithmetic, self.wall_value, "epsilon right wall trace")
        if lower <= zero or upper < lower:
            raise ValueError("epsilon bounds must satisfy 0 < lower_bound <= upper_bound")
        if not lower <= wall <= upper:
            raise ValueError("epsilon right wall trace must lie within its declared bounds")

        interfaces = tuple(
            _finite_scalar(arithmetic, value, "epsilon interface")
            for value in self.interfaces
        )
        previous = zero
        for value in interfaces:
            if value <= previous:
                raise ValueError("epsilon interfaces must be positive and strictly increasing")
            previous = value
        require_precision(arithmetic, signature, subject="coefficient binding",
                          rebuild="bind the coefficient again")
        return _BoundCoefficient(
            arithmetic=arithmetic,
            lower_bound=lower,
            upper_bound=upper,
            wall_value=wall,
            interfaces=interfaces,
            is_identity=False,
            _function=self.function,
        )


def as_coefficient(epsilon):
    """Normalize a description or scalar without guessing callback signatures."""
    if isinstance(epsilon, (ConstantCoefficient, Coefficient)):
        return epsilon
    if callable(epsilon):
        raise TypeError(
            "bare epsilon callbacks require bounds and a right wall trace; use "
            "Coefficient(function, ...) or Coefficient.from_function(epsilon, ...)"
        )
    if not isinstance(epsilon, (Number, str)):
        raise TypeError("epsilon must be a scalar or a coefficient description")
    return ConstantCoefficient(epsilon)
