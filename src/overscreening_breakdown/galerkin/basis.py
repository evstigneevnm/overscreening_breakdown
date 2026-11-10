"""Conforming rational Chebyshev spaces for the half-line mixed problem.

The reference coordinate is ``y=(x-L)/(x+L)``.  ``N`` denotes the polynomial
space size, rather than the number of collocation unknowns::

    V_N = (1-y**2) P_(N-2)    (N-1 coefficients)
    Q_N = (1-y) P_(N-1)       (N coefficients)

The first field is ``u = wall_lift + sum(a_k V_k)`` and the independent second
field is ``w = sum(b_k Q_k)``.  The combined vector consequently has ``2*N-1``
entries, with all ``a_k`` entries first.  This module supplies spaces and
evaluations only; it does not assemble or solve a mixed Galerkin system.
"""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Tuple

from overscreening_breakdown.rational_mapping import RationalHalfLineMap


def _reference_coordinate(mapping, y):
    return mapping.reference_coordinate(y)


@dataclass(frozen=True)
class AffineWallLift:
    """The physical lift ``u_b*L/(x+L) = u_b*(1-y)/2``.

    ``value``, ``dy`` and ``dx`` take a reference coordinate.  ``at`` and
    ``derivative_at`` take a physical coordinate, including positive infinity.
    The value at infinity and its physical derivative are both exactly zero.
    """

    u_b: Any
    mapping: RationalHalfLineMap

    def __post_init__(self):
        self.mapping.check_precision()
        value = self.mapping.arithmetic.scalar(self.u_b)
        if not self.mapping.arithmetic.isfinite(value):
            raise ValueError("wall value must be finite")
        object.__setattr__(self, "u_b", value)

    def value(self, y):
        y = _reference_coordinate(self.mapping, y)
        return self.u_b * ((1-y) / 2)

    def dy(self, y):
        _reference_coordinate(self.mapping, y)
        return -self.u_b / 2

    def dx(self, y):
        y = _reference_coordinate(self.mapping, y)
        return -self.u_b / 2 * self.mapping.inverse_jacobian(y)

    def at(self, x):
        x = self.mapping.physical_coordinate(x)
        arithmetic = self.mapping.arithmetic
        if x == arithmetic.inf:
            return arithmetic.scalar(0)
        if x >= self.mapping.L:
            ratio = self.mapping.L/x
            return self.u_b * (ratio/(1+ratio))
        return self.u_b/(1+x/self.mapping.L)

    def derivative_at(self, x):
        x = self.mapping.physical_coordinate(x)
        arithmetic = self.mapping.arithmetic
        if x == arithmetic.inf:
            return arithmetic.scalar(0)
        value = self.at(x)
        if x >= self.mapping.L:
            return -(value/(1+self.mapping.L/x))/x
        return -(value/(1+x/self.mapping.L))/self.mapping.L


@dataclass(frozen=True)
class BasisTable:
    """Immutable node-major basis and physical-derivative tables.

    Row ``i`` corresponds to ``plan.nodes[i]``.  All entries belong to the
    basis's arithmetic context, and ``check_precision`` rejects reuse after
    that context's precision changes.  ``plan.physical_weights`` supplies the
    matching physical integration weights.
    """

    basis: Any = field(repr=False, compare=False)
    plan: Any = field(repr=False, compare=False)
    v: Tuple[Tuple[Any, ...], ...]
    q: Tuple[Tuple[Any, ...], ...]
    v_dx: Tuple[Tuple[Any, ...], ...]
    q_dx: Tuple[Tuple[Any, ...], ...]

    @property
    def mapping(self):
        return self.basis.mapping

    @property
    def precision_signature(self):
        return self.mapping.precision_signature

    @property
    def nodes(self):
        return self.plan.nodes

    def check_precision(self):
        self.mapping.check_precision()
        self.plan.check_precision()


@dataclass(frozen=True, init=False)
class GalerkinBasis:
    """Half-line conforming spaces expressed in Chebyshev polynomials ``T_k``.

    Every evaluation below takes one *reference* coordinate in ``[-1, 1]``.
    Endpoint traces are evaluated by polynomial recurrences, with no division
    by ``1-y**2``.  Physical derivatives use the rational mapping's chain rule.
    """

    N: int
    mapping: RationalHalfLineMap

    def __init__(self, N, arithmetic, L=1):
        if isinstance(N, bool) or not isinstance(N, Integral) or N < 2:
            raise ValueError("N must be an integer greater than or equal to 2")
        object.__setattr__(self, "N", int(N))
        object.__setattr__(self, "mapping", RationalHalfLineMap(arithmetic, L=L))

    @property
    def arithmetic(self):
        return self.mapping.arithmetic

    @property
    def n_u(self):
        return self.N - 1

    @property
    def n_w(self):
        return self.N

    @property
    def unknown_count(self):
        return 2*self.N - 1

    def check_precision(self):
        self.mapping.check_precision()

    def _polynomials(self, y):
        """Return T_k and T_k' without endpoint-singular derivative formulas."""
        y = _reference_coordinate(self.mapping, y)
        zero = self.arithmetic.scalar(0)
        one = self.arithmetic.scalar(1)
        values, derivatives = [one, y], [zero, one]
        for _ in range(2, self.N):
            values.append(2*y*values[-1] - values[-2])
            derivatives.append(2*values[-2] + 2*y*derivatives[-1]
                               - derivatives[-2])
        return y, values, derivatives

    def v(self, y):
        y, values, _ = self._polynomials(y)
        factor = (1-y)*(1+y)
        return tuple(factor*value for value in values[:self.n_u])

    def q(self, y):
        y, values, _ = self._polynomials(y)
        return tuple((1-y)*value for value in values)

    def v_dy(self, y):
        y, values, derivatives = self._polynomials(y)
        factor = (1-y)*(1+y)
        return tuple(-2*y*value + factor*derivative for value, derivative in
                     zip(values[:self.n_u], derivatives[:self.n_u]))

    def q_dy(self, y):
        y, values, derivatives = self._polynomials(y)
        return tuple(-value + (1-y)*derivative
                     for value, derivative in zip(values, derivatives))

    def v_dx(self, y):
        y = _reference_coordinate(self.mapping, y)
        factor = self.mapping.inverse_jacobian(y)
        return tuple(factor*value for value in self.v_dy(y))

    def q_dx(self, y):
        y = _reference_coordinate(self.mapping, y)
        factor = self.mapping.inverse_jacobian(y)
        return tuple(factor*value for value in self.q_dy(y))

    def lift(self, u_b):
        return AffineWallLift(u_b, self.mapping)

    def split_coefficients(self, coefficients):
        """Validate and split a vector into independent ``u`` and ``w`` parts."""
        self.check_precision()
        values = tuple(self.arithmetic.scalar(value) for value in coefficients)
        if len(values) != self.unknown_count:
            raise ValueError("expected {} mixed coefficients".format(self.unknown_count))
        if any(not self.arithmetic.isfinite(value) for value in values):
            raise ValueError("mixed coefficients must be finite")
        return values[:self.n_u], values[self.n_u:]

    def evaluate_fields(self, coefficients, y, u_b=0):
        """Evaluate independent physical fields ``(u, w)`` at reference ``y``."""
        a, b = self.split_coefficients(coefficients)
        zero = self.arithmetic.scalar(0)
        u = self.lift(u_b).value(y) + sum(
            (coefficient*value for coefficient, value in zip(a, self.v(y))), zero
        )
        w = sum((coefficient*value for coefficient, value in zip(b, self.q(y))), zero)
        return u, w

    def tabulate(self, plan):
        """Tabulate a compatible positive quadrature plan without float casts.

        Context identity is required even for equal mpmath digit counts: a
        table must never combine scalar operations from different owners.
        """
        self.check_precision()
        plan.check_precision()
        if plan.mapping.arithmetic is not self.arithmetic:
            raise ValueError("basis and quadrature must share the same arithmetic owner")
        if plan.mapping.L != self.mapping.L:
            raise ValueError("basis and quadrature must use the same mapping length")
        if plan.precision_signature != self.mapping.precision_signature:
            raise ValueError("basis and quadrature precision signatures must agree")
        return BasisTable(
            basis=self, plan=plan,
            v=tuple(self.v(y) for y in plan.nodes),
            q=tuple(self.q(y) for y in plan.nodes),
            v_dx=tuple(self.v_dx(y) for y in plan.nodes),
            q_dx=tuple(self.q_dx(y) for y in plan.nodes),
        )
