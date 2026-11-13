"""Physical flux problems with scalar callbacks bound to owned arithmetic.

``FluxProblem`` describes ``(epsilon * (u' - delta**2 * u'''))' = f(x, u)``.
Its source is unscaled: neither division by delta**2 nor homotopy damping is
part of this interface. Discretizations apply their own normalization.

Callbacks receive the bound arithmetic explicitly. A frozen description
captures callback identity, not mutable objects inside a callback's closure;
callers must keep such state unchanged while a solver uses the problem.
Bound evaluations reject owner precision changes before using rounded data
and after callbacks return. Bind the original description at a new precision.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Tuple

from overscreening_breakdown.arithmetic import arithmetic_signature, require_precision
from overscreening_breakdown.coefficients import as_coefficient


_DEFAULT_BOUNDARIES = (
    (0, None, None, 0, None),
    (None, None, None, None, None),
)


def _finite_scalar(arithmetic, value, name, error_type=ValueError):
    result = arithmetic.scalar(value)
    if not arithmetic.isfinite(result):
        raise error_type("{} must be finite".format(name))
    return result


def _copy_boundaries(boundaries):
    copied = tuple(tuple(boundary) for boundary in boundaries)
    if len(copied) != 2 or any(len(boundary) != 5 for boundary in copied):
        raise ValueError("boundary_conditions must contain two rows of five values")
    return copied


@dataclass(frozen=True, init=False)
class FluxProblem:
    """An arithmetic-independent scalar physical problem description.

    ``f(x, u, arithmetic)`` and ``f_u(x, u, arithmetic)`` are the physical
    source and its derivative with respect to u. An optional
    ``initial_guess(x, arithmetic)`` supplies the starting field; the default
    is the wall value times exp(-x). Boundary rows list derivatives 0 through
    4 at the two endpoints, using None for unspecified values.

    Numeric inputs are retained until ``bind`` to preserve decimal strings.
    Only domains [0, R] and [0, infinity) are currently described. A solver
    must separately validate its supported domain and boundary combinations.
    """

    delta: Any
    f: Callable = field(repr=False, compare=False)
    f_u: Callable = field(repr=False, compare=False)
    epsilon: Any
    domain: Tuple[Any, Any]
    boundary_conditions: Tuple[Tuple[Any, ...], ...]
    initial_guess: Any = field(repr=False, compare=False)

    def __init__(
        self, *, delta, f, f_u, epsilon=1, domain=(0, "inf"),
        boundary_conditions=_DEFAULT_BOUNDARIES, initial_guess=None
    ):
        if not callable(f) or not callable(f_u):
            raise TypeError("f and f_u must be callable")
        if initial_guess is not None and not callable(initial_guess):
            raise TypeError("initial_guess must be callable or None")
        domain = tuple(domain)
        if len(domain) != 2:
            raise ValueError("domain must contain two endpoints")
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "f", f)
        object.__setattr__(self, "f_u", f_u)
        object.__setattr__(self, "epsilon", as_coefficient(epsilon))
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "boundary_conditions", _copy_boundaries(boundary_conditions))
        object.__setattr__(self, "initial_guess", initial_guess)

    def bind(self, arithmetic):
        """Convert parameters once in a solver's arithmetic context."""
        signature = arithmetic_signature(arithmetic)
        zero = arithmetic.scalar(0)
        delta = _finite_scalar(arithmetic, self.delta, "delta")
        if delta < zero:
            raise ValueError("delta must be nonnegative")
        left = _finite_scalar(arithmetic, self.domain[0], "left domain endpoint")
        right = arithmetic.scalar(self.domain[1])
        if left != zero or not right > left:
            raise ValueError("domain must be [0, R] with R > 0, or [0, infinity)")
        if not arithmetic.isfinite(right) and right != arithmetic.inf:
            raise ValueError("right domain endpoint must be finite or positive infinity")
        boundaries = tuple(
            tuple(
                None if value is None else _finite_scalar(arithmetic, value, "boundary value")
                for value in row
            )
            for row in self.boundary_conditions
        )
        epsilon = self.epsilon.bind(arithmetic)
        require_precision(arithmetic, signature, subject="physical problem binding",
                          rebuild="bind the original problem again")
        if any(not left < point < right for point in epsilon.interfaces):
            raise ValueError("epsilon interfaces must lie strictly inside the domain")
        return BoundFluxProblem(
            arithmetic=arithmetic,
            delta=delta,
            epsilon=epsilon,
            domain=(left, right),
            boundary_conditions=boundaries,
            _f=self.f,
            _f_u=self.f_u,
            _initial_guess=self.initial_guess,
        )


@dataclass(frozen=True)
class BoundFluxProblem:
    """A scalar physical problem evaluated in one arithmetic context."""

    arithmetic: Any = field(repr=False, compare=False)
    delta: Any
    epsilon: Any
    domain: Tuple[Any, Any]
    boundary_conditions: Tuple[Tuple[Any, ...], ...]
    _f: Callable = field(repr=False, compare=False)
    _f_u: Callable = field(repr=False, compare=False)
    _initial_guess: Any = field(default=None, repr=False, compare=False)
    precision_signature: tuple = field(init=False)

    def __post_init__(self):
        if getattr(self.epsilon, "arithmetic", None) is not self.arithmetic:
            raise ValueError("physical problem and coefficient must share one arithmetic owner")
        self.epsilon.check_precision()
        object.__setattr__(self, "precision_signature", arithmetic_signature(self.arithmetic))

    def check_precision(self):
        require_precision(self.arithmetic, self.precision_signature,
                          subject="physical problem", rebuild="bind the original problem again")
        self.epsilon.check_precision()

    def _coordinate(self, x):
        self.check_precision()
        x = _finite_scalar(self.arithmetic, x, "physical coordinate")
        if not self.domain[0] <= x <= self.domain[1]:
            raise ValueError("physical coordinate lies outside the domain")
        return x

    def _source(self, callback, x, u):
        x = self._coordinate(x)
        u = _finite_scalar(self.arithmetic, u, "field value")
        value = callback(x, u, self.arithmetic)
        self.check_precision()
        return _finite_scalar(
            self.arithmetic, value,
            "physical source value", error_type=OverflowError
        )

    def physical_rhs(self, x, u):
        """Evaluate unscaled f; nonfinite trial outputs raise OverflowError."""
        return self._source(self._f, x, u)

    def physical_rhs_u(self, x, u):
        return self._source(self._f_u, x, u)

    def base_solution(self, x):
        x = self._coordinate(x)
        if self._initial_guess is None:
            wall = self.boundary_conditions[0][0]
            if wall is None:
                wall = self.arithmetic.scalar(0)
            value = wall * self.arithmetic.exp(-x)
        else:
            value = self._initial_guess(x, self.arithmetic)
        self.check_precision()
        return _finite_scalar(self.arithmetic, value, "initial guess value")

    def get_domain(self):
        self.check_precision()
        return self.domain

    def get_boundary_conditions(self):
        self.check_precision()
        return self.boundary_conditions
