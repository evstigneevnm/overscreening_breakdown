"""Immutable manufactured cases, separate from solvers and sweep settings.

The historical classes remain available through ``legacy_problem()``. Their
binary-float constants and continuation rules are deliberately left intact.
Native cases describe the same intended q=0 equations using exact rational
constants and boundary traces evaluated in the caller's arithmetic.
"""

from dataclasses import dataclass, field
from math import comb, factorial
from numbers import Integral
from typing import Callable, Optional, Tuple

from overscreening_breakdown.problems.flux import FluxProblem
from overscreening_breakdown.arithmetic import arithmetic_signature, require_precision


def _coordinate(x, arithmetic):
    x = arithmetic.scalar(x)
    if not arithmetic.isfinite(x) or x < 0:
        raise ValueError("manufactured references require a finite coordinate x >= 0")
    return x


def _product(first, second):
    return tuple(sum(comb(n, k)*first[k]*second[n-k] for k in range(n+1))
                 for n in range(5))


def _reciprocal_cosine(x, arithmetic, offset):
    reciprocal = arithmetic.scalar(1)/(x+offset)
    cosine, sine = arithmetic.cos(reciprocal), arithmetic.sin(reciprocal)
    return (cosine, sine*reciprocal**2, -cosine*reciprocal**4-2*sine*reciprocal**3,
            -sine*reciprocal**6+6*cosine*reciprocal**5+6*sine*reciprocal**4,
            cosine*reciprocal**8+12*sine*reciprocal**7
            -36*cosine*reciprocal**6-24*sine*reciprocal**5)


def profile_derivatives(profile, x, arithmetic):
    """Analytic derivatives 0..4, with native arithmetic throughout."""
    x = _coordinate(x, arithmetic)
    one = arithmetic.scalar(1)
    if profile in ("gaussian", "gaussian_cosine"):
        gaussian = arithmetic.exp(-x*x)
        if gaussian == 0:
            return (arithmetic.scalar(0),)*5
        values = tuple(gaussian*p for p in (
            one, -2*x, 4*x*x-2, 12*x-8*x**3, 12-48*x*x+16*x**4))
        if profile == "gaussian_cosine":
            cosine, sine = arithmetic.cos(x), arithmetic.sin(x)
            return _product(values, (cosine, -sine, -cosine, sine, cosine))
        return values
    if profile in ("exponential", "exponential_cosine", "exponential_reciprocal_cosine"):
        exponential = arithmetic.exp(-x)
        values = tuple((-1)**n*exponential for n in range(5))
        if profile == "exponential":
            return values
        if profile == "exponential_reciprocal_cosine":
            return _product(values, _reciprocal_cosine(x, arithmetic, one))
        cosine, sine = arithmetic.cos(x), arithmetic.sin(x)
        return _product(values, (cosine, -sine, -cosine, sine, cosine))
    if profile in ("rational", "rational_rapid_cosine"):
        reciprocal = one/(one+x)
        values = tuple((-1)**n*factorial(n)*reciprocal**(n+1) for n in range(5))
        if profile == "rational":
            return values
        return _product(values, _reciprocal_cosine(x, arithmetic, one/5))
    if profile == "rational_square_cosine":
        d = one+x*x
        amplitude = (one/d, -2*x/d**2, 8*x*x/d**3-2/d**2,
                     -48*x**3/d**4+24*x/d**3,
                     384*x**4/d**5-288*x*x/d**4+24/d**3)
        return _product(amplitude, _reciprocal_cosine(x, arithmetic, one))
    if profile == "poisson_boltzmann":
        v = arithmetic.tanh(one/4)*arithmetic.exp(-x)
        u = 2*(arithmetic.context.log1p(v)-arithmetic.context.log1p(-v))
        first = -4*v/(one-v*v)
        second = arithmetic.sinh(u)
        third = arithmetic.cosh(u)*first
        fourth = arithmetic.sinh(u)*first*first+arithmetic.cosh(u)*second
        return u, first, second, third, fourth
    raise ValueError("unknown manufactured profile")


@dataclass(frozen=True)
class ManufacturedCase:
    """A reusable equation/reference definition with no numerical owner.

    ``supported_solvers`` lists the current common solver choices. The special
    name ``legacy_collocation`` means only the original general-operator API
    can express the equation. A reference is not a convergence certificate.
    ``w=-u''`` is available even for a second-order or legacy-only equation;
    physical ``flux`` requires a flux-form problem description.
    """

    identifier: str
    name: str
    supported_solvers: Tuple[str, ...]
    _derivatives: Callable = field(repr=False, compare=False)
    _problem_factory: Optional[Callable] = field(repr=False, compare=False)
    _legacy_factory: Optional[Callable] = field(default=None, repr=False, compare=False)
    notes: Tuple[str, ...] = ()

    def make_problem(self):
        if self._problem_factory is None:
            raise ValueError("{} is outside the finite-delta flux formulation; use legacy_problem()"
                             .format(self.identifier))
        return self._problem_factory()

    def legacy_problem(self):
        if self._legacy_factory is None:
            raise ValueError("{} has no historical problem class".format(self.identifier))
        return self._legacy_factory()

    def derivative(self, x, arithmetic, order=0):
        if isinstance(order, bool) or not isinstance(order, Integral) or not 0 <= order <= 4:
            raise ValueError("reference derivative order must be an integer from 0 through 4")
        values = self._derivatives(_coordinate(x, arithmetic), arithmetic)
        return arithmetic.scalar(values[int(order)])

    def u(self, x, arithmetic):
        return self.derivative(x, arithmetic)

    def u_dx(self, x, arithmetic):
        return self.derivative(x, arithmetic, 1)

    def w(self, x, arithmetic):
        return -self.derivative(x, arithmetic, 2)

    def w_dx(self, x, arithmetic):
        return -self.derivative(x, arithmetic, 3)

    def flux(self, x, arithmetic):
        x = _coordinate(x, arithmetic)
        physical = self.make_problem().bind(arithmetic)
        return physical.epsilon(x)*(self.u_dx(x, arithmetic)
                                    +physical.delta**2*self.w_dx(x, arithmetic))


@dataclass(frozen=True)
class NativeFluxDescription:
    """Resolve irrational parameters and wall traces only when binding.

    This uses the same ``bind(arithmetic)`` protocol as FluxProblem without
    rounding sqrt(10), cos(1), or cos(5) in the factory's environment.
    """

    delta: Callable = field(repr=False, compare=False)
    boundaries: Callable = field(repr=False, compare=False)
    f: Callable = field(repr=False, compare=False)
    f_u: Callable = field(repr=False, compare=False)
    initial_guess: Optional[Callable] = field(default=None, repr=False, compare=False)
    epsilon: object = 1
    domain: tuple = (0, "inf")

    def bind(self, arithmetic):
        signature = arithmetic_signature(arithmetic)
        delta = self.delta(arithmetic)
        require_precision(arithmetic, signature, subject="manufactured problem binding",
                          rebuild="bind the original manufactured description again")
        boundaries = self.boundaries(arithmetic)
        require_precision(arithmetic, signature, subject="manufactured problem binding",
                          rebuild="bind the original manufactured description again")
        return FluxProblem(delta=delta, f=self.f, f_u=self.f_u,
                           epsilon=self.epsilon, domain=self.domain,
                           boundary_conditions=boundaries,
                           initial_guess=self.initial_guess).bind(arithmetic)


def validate_number(number, maximum):
    if isinstance(number, bool) or not isinstance(number, Integral) or not 1 <= number <= maximum:
        raise ValueError("case number must be an integer from 1 through {}".format(maximum))
    return int(number)


def native_problem(profile, *, delta_squared, wall_orders, source, source_u,
                   initial_profile="exponential", initial_callback=None, far_zero=False):
    """Construct a bindable description without capturing an arithmetic owner."""
    def delta(arithmetic):
        numerator, denominator = delta_squared
        if numerator == 0:
            return arithmetic.scalar(0)
        return arithmetic.sqrt(arithmetic.scalar(numerator)/denominator)

    def boundaries(arithmetic):
        derivatives = profile_derivatives(profile, arithmetic.scalar(0), arithmetic)
        left = tuple(derivatives[k] if k in wall_orders else None for k in range(5))
        right = (arithmetic.scalar(0), None, None, None, None) if far_zero else (None,)*5
        return left, right

    def initial(x, arithmetic):
        return profile_derivatives(initial_profile, x, arithmetic)[0]

    return NativeFluxDescription(delta, boundaries, source, source_u,
                                 initial if initial_callback is None else initial_callback)
