"""Polynomial Petrov spaces for the finite mixed flux problem on [0, R].

N gives N-1 potential coefficients and N+1 independent curvature coefficients.
The two test spaces each have N functions, so both vectors have size 2*N.
All evaluators take the affine reference coordinate y in [-1, 1].
"""

from dataclasses import dataclass
from numbers import Integral

from overscreening_breakdown.affine_mapping import AffineIntervalMap
from overscreening_breakdown.galerkin.basis import BasisTable


@dataclass(frozen=True)
class AffineEndpointLift:
    """Linear lift through the two prescribed endpoint values."""

    u_b: object
    u_R: object
    mapping: object

    def __post_init__(self):
        self.mapping.check_precision()
        a = self.mapping.arithmetic
        for name in ("u_b", "u_R"):
            value = a.scalar(getattr(self, name))
            if not a.isfinite(value):
                raise ValueError("endpoint values must be finite")
            object.__setattr__(self, name, value)

    def value(self, y):
        y = self.mapping.reference_coordinate(y)
        return self.u_b*((1-y)/2)+self.u_R*((1+y)/2)

    def dy(self, y):
        self.mapping.reference_coordinate(y)
        return (self.u_R-self.u_b)/2

    def dx(self, y):
        return self.dy(y)*self.mapping.inverse_jacobian(y)

    def at(self, x):
        return self.value(self.mapping.to_reference(x))

    def derivative_at(self, x):
        return self.dx(self.mapping.to_reference(x))


@dataclass(frozen=True)
class FiniteBasisTable(BasisTable):
    """Native node-major trial and test values and physical derivatives."""

    phi: tuple
    psi: tuple
    phi_dx: tuple
    psi_dx: tuple


@dataclass(frozen=True, init=False)
class FiniteGalerkinBasis:
    """Finite trial v=(1-y²)T_k, w=T_k and one-sided test functions.

    phi=(1+y)T_k vanishes only at the left endpoint; psi=(1-y)T_k
    vanishes only at the right endpoint. Neither trace of w is constrained.
    This Petrov discretization has its own stability requirements; the
    half-line Galerkin stability argument does not apply automatically.
    """

    N: int
    mapping: object
    basis_id = "affine_chebyshev_petrov_mixed"

    def __init__(self, N, arithmetic, domain):
        if isinstance(N, bool) or not isinstance(N, Integral) or N < 2:
            raise ValueError("N must be an integer greater than or equal to 2")
        mapping = AffineIntervalMap(arithmetic, domain=domain)
        if mapping.left != 0:
            raise ValueError("finite mixed spaces currently require domain [0, R]")
        object.__setattr__(self, "N", int(N))
        object.__setattr__(self, "mapping", mapping)

    @property
    def arithmetic(self):
        return self.mapping.arithmetic

    @property
    def domain(self):
        return self.mapping.domain

    @property
    def length(self):
        return self.mapping.length

    @property
    def n_u(self):
        return self.N-1

    @property
    def n_w(self):
        return self.N+1

    @property
    def n_first_tests(self):
        return self.N

    @property
    def n_second_tests(self):
        return self.N

    @property
    def unknown_count(self):
        return 2*self.N

    def check_precision(self):
        self.mapping.check_precision()

    def _polynomials(self, y):
        y = self.mapping.reference_coordinate(y)
        a = self.arithmetic
        values, derivatives = [a.scalar(1), y], [a.scalar(0), a.scalar(1)]
        for _ in range(2, self.N+1):
            values.append(2*y*values[-1]-values[-2])
            derivatives.append(2*values[-2]+2*y*derivatives[-1]-derivatives[-2])
        return y, values, derivatives

    def v(self, y):
        y, values, _ = self._polynomials(y)
        return tuple((1-y)*(1+y)*value for value in values[:self.n_u])

    def q(self, y):
        return tuple(self._polynomials(y)[1])

    def phi(self, y):
        y, values, _ = self._polynomials(y)
        return tuple((1+y)*value for value in values[:self.N])

    def psi(self, y):
        y, values, _ = self._polynomials(y)
        return tuple((1-y)*value for value in values[:self.N])

    def v_dy(self, y):
        y, values, derivatives = self._polynomials(y)
        return tuple(-2*y*value+(1-y)*(1+y)*derivative
                     for value, derivative in zip(values[:self.n_u], derivatives[:self.n_u]))

    def q_dy(self, y):
        return tuple(self._polynomials(y)[2])

    def phi_dy(self, y):
        y, values, derivatives = self._polynomials(y)
        return tuple(value+(1+y)*derivative
                     for value, derivative in zip(values[:self.N], derivatives[:self.N]))

    def psi_dy(self, y):
        y, values, derivatives = self._polynomials(y)
        return tuple(-value+(1-y)*derivative
                     for value, derivative in zip(values[:self.N], derivatives[:self.N]))

    def _dx(self, callback, y):
        factor = self.mapping.inverse_jacobian(y)
        return tuple(factor*value for value in callback(y))

    def v_dx(self, y):
        return self._dx(self.v_dy, y)

    def q_dx(self, y):
        return self._dx(self.q_dy, y)

    def phi_dx(self, y):
        return self._dx(self.phi_dy, y)

    def psi_dx(self, y):
        return self._dx(self.psi_dy, y)

    def lift(self, u_b, u_R):
        return AffineEndpointLift(u_b, u_R, self.mapping)

    def split_coefficients(self, coefficients):
        self.check_precision()
        values = tuple(self.arithmetic.scalar(value) for value in coefficients)
        if len(values) != self.unknown_count or any(not self.arithmetic.isfinite(value) for value in values):
            raise ValueError("finite mixed coefficients must be finite and have 2*N entries")
        return values[:self.n_u], values[self.n_u:]

    def evaluate_fields(self, coefficients, y, u_b=0, u_R=0):
        first, second = self.split_coefficients(coefficients)
        zero = self.arithmetic.scalar(0)
        return (self.lift(u_b, u_R).value(y)+sum(
            (entry*value for entry, value in zip(first, self.v(y))), zero),
            sum((entry*value for entry, value in zip(second, self.q(y))), zero))

    def tabulate(self, plan):
        self.check_precision()
        plan.check_precision()
        if (plan.arithmetic is not self.arithmetic
                or plan.mapping.mapping_type != self.mapping.mapping_type
                or plan.mapping.domain != self.domain):
            raise ValueError("finite basis and plan must share an arithmetic owner and geometry")
        return FiniteBasisTable(
            basis=self, plan=plan, v=tuple(self.v(y) for y in plan.nodes),
            q=tuple(self.q(y) for y in plan.nodes), v_dx=tuple(self.v_dx(y) for y in plan.nodes),
            q_dx=tuple(self.q_dx(y) for y in plan.nodes), phi=tuple(self.phi(y) for y in plan.nodes),
            psi=tuple(self.psi(y) for y in plan.nodes), phi_dx=tuple(self.phi_dx(y) for y in plan.nodes),
            psi_dx=tuple(self.psi_dx(y) for y in plan.nodes),
        )
