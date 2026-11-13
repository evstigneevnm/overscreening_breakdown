"""Genuine epsilon jumps with continuous u,u',u'' and physical flux."""

from collections import OrderedDict
from dataclasses import dataclass, field
from math import factorial
from numbers import Integral
from typing import Callable

from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.manufactured.common import ManufacturedCase
from overscreening_breakdown.problems.flux import FluxProblem
from overscreening_breakdown.arithmetic import arithmetic_signature, require_precision


def transmission_case():
    """Return the existing interface fixture with epsilon=1/2 about x=1.

    u=1/(1+x)+(5/12)*(x-1)_+^3/(1+x)^4, delta=2.
    At x=1, coefficient and derivative references use their right traces.
    Physical flux is continuous and equals 5/4 at the interface.
    """
    def derivatives(x, arithmetic):
        one = arithmetic.scalar(1)
        reciprocal = one/(one+x)
        coefficients = ((one,) if x < 1 else
                        (arithmetic.scalar(17)/12, -arithmetic.scalar(5)/2,
                         arithmetic.scalar(5), -arithmetic.scalar(10)/3))
        return tuple(sum(((-1)**order*factorial(k+order-1)//factorial(k-1)
                          *coefficient*reciprocal**(k+order)
                          for k, coefficient in enumerate(coefficients, 1)), arithmetic.scalar(0))
                     for order in range(5))

    def source(x, u, arithmetic):
        values = derivatives(x, arithmetic)
        epsilon = arithmetic.scalar(1 if x < 1 else 2)
        return (epsilon*(values[2]-4*values[4])+(u-values[0])
                +arithmetic.scalar(1)/8*(u**3-values[0]**3))

    def source_u(x, u, arithmetic):
        return 1+arithmetic.scalar(3)/8*u*u

    def factory():
        epsilon = Coefficient(lambda x, a: a.scalar(1 if x < 1 else 2),
                              lower_bound=1, upper_bound=2, wall_value=1, interfaces=(1,))
        return FluxProblem(delta="2", epsilon=epsilon, f=source, f_u=source_u,
                           boundary_conditions=(("1", None, None, "-6", None), (None,)*5))

    return ManufacturedCase("transmission", "C2 rational transmission solution", ("galerkin",),
                            derivatives, factory,
                            notes=("Right traces at x=1; no pointwise finite-N flux continuity guarantee.",))


def _reciprocal_derivatives(coefficients, coordinate, arithmetic):
    """Differentiate a four-term reciprocal polynomial analytically."""
    reciprocal = arithmetic.scalar(1)/(1+coordinate)
    return tuple(sum(((-1)**order*factorial(k+order-1)//factorial(k-1)
                      *coefficient*reciprocal**(k+order)
                      for k, coefficient in enumerate(coefficients, 1)), arithmetic.scalar(0))
                 for order in range(5))


@dataclass(frozen=True)
class _LayeredReference:
    material: object
    delta: object
    beta: object
    coefficients: tuple
    signature: tuple
    _values: object = field(default_factory=OrderedDict, repr=False, compare=False)

    @property
    def arithmetic(self):
        return self.material.arithmetic

    def check_precision(self):
        require_precision(self.arithmetic, self.signature, subject="reference",
                          rebuild="bind the manufactured case again")
        self.material.check_precision()

    def derivatives(self, x, region=None):
        self.check_precision()
        a = self.arithmetic
        x = a.scalar(x)
        if not a.isfinite(x) or x < 0:
            raise ValueError("manufactured reference coordinates must be finite and nonnegative")
        if region is None:
            region = self.material.region_index(x)
        if isinstance(region, bool) or not isinstance(region, Integral) or not 0 <= region < len(self.coefficients):
            raise ValueError("reference region index is outside the material")
        key = (x, int(region))
        if key not in self._values:
            values = _reciprocal_derivatives(self.coefficients[region], x, a)
            if any(not a.isfinite(value) for value in values):
                raise OverflowError("manufactured reference is outside the arithmetic range")
            self._values[key] = values
            if len(self._values) > 2048:
                self._values.popitem(last=False)
        return self._values[key]

    def interface_derivative(self, index, order=0, side="right"):
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.material.interfaces):
            raise ValueError("interface index is outside the material")
        if isinstance(order, bool) or not isinstance(order, Integral) or not 0 <= order <= 4:
            raise ValueError("reference derivative order must be between zero and four")
        if side not in ("left", "right"):
            raise ValueError("interface side must be 'left' or 'right'")
        region = index if side == "left" else index+1
        return self.derivatives(self.material.interfaces[index], region)[order]

    def flux_trace(self, index, side="right"):
        return self.material.trace(index, side=side)*(
            self.interface_derivative(index, 1, side)
            -self.delta**2*self.interface_derivative(index, 3, side))

    def flux(self, x):
        values = self.derivatives(x)
        return self.material.epsilon(x)*(values[1]-self.delta**2*values[3])

    def flux_dx(self, x):
        values = self.derivatives(x)
        return (self.material.epsilon_dx(x)*(values[1]-self.delta**2*values[3])
                +self.material.epsilon(x)*(values[2]-self.delta**2*values[4]))

    def source(self, x, u):
        values = self.derivatives(x)
        result = (self.material.epsilon_dx(x)*(values[1]-self.delta**2*values[3])
                  +self.material.epsilon(x)*(values[2]-self.delta**2*values[4])
                  +(u-values[0]))
        return result if self.beta == 0 else result+self.beta*(u**3-values[0]**3)

    def source_u(self, x, u):
        self.check_precision()
        return (self.arithmetic.scalar(1) if self.beta == 0 else
                self.arithmetic.scalar(1)+3*self.beta*u*u)


@dataclass(frozen=True)
class LayeredTransmissionCase(ManufacturedCase):
    """ManufacturedCase with exact one-sided transmission diagnostics.

    Interface indices are zero-based. The main field evaluators retain the
    usual right-trace convention. Numerical Galerkin fields do not inherit
    these exact pointwise flux identities at finite resolution.
    """

    material: object = field(default=None, repr=False, compare=False)
    _reference: Callable = field(default=None, repr=False, compare=False)

    def interface_derivative(self, index, arithmetic, order=0, side="right"):
        return self._reference(arithmetic).interface_derivative(index, order, side)

    def flux_trace(self, index, arithmetic, side="right"):
        return self._reference(arithmetic).flux_trace(index, side)

    def flux(self, x, arithmetic):
        return self._reference(arithmetic).flux(arithmetic.scalar(x))

    def flux_dx(self, x, arithmetic):
        """Analytic ordinary derivative of the reference flux inside a layer.

        Values at interfaces use the right trace. The reference flux itself
        is continuous, so no distributional interface source is omitted.
        """
        return self._reference(arithmetic).flux_dx(arithmetic.scalar(x))


@dataclass(frozen=True)
class _LayeredFluxDescription:
    """Freeze analytic reference data when a numerical owner binds the source."""

    _reference: Callable = field(repr=False, compare=False)
    epsilon: object
    delta: object
    domain: tuple = (0, "inf")

    def bind(self, arithmetic):
        reference = self._reference(arithmetic)
        return FluxProblem(
            delta=reference.delta, epsilon=self.epsilon,
            f=lambda x, u, a: reference.source(x, u),
            f_u=lambda x, u, a: reference.source_u(x, u),
            boundary_conditions=((1, None, None, -6, None),
                                 (0, None, None, None, None)),
            initial_guess=lambda x, a: a.exp(-x),
        ).bind(arithmetic)


def layered_transmission_case(material, *, nonlinear=True, delta="2", beta="0.125"):
    """Return a known C2 solution for any supported layered material.

    Starting from U_0=1/(1+x), add a_i*(x-x_i)_+**3/(1+x)**4 at each
    interface. Its first three traces vanish, and a_i is chosen to preserve
    epsilon*(u'-delta**2*u'''). Every interval consequently uses only four
    reciprocal powers. The derivatives and piecewise-smooth epsilon term
    are analytic; the solver itself needs no derivative of epsilon.

    The physical source is J_*' +(u-u_*) + beta*(u**3-u_* **3), or the same
    expression with beta=0 for the linear case. All parameter conversion and
    interface recursion happen in a solver-owned arithmetic context.
    """
    if not isinstance(nonlinear, bool):
        raise TypeError("nonlinear must be a boolean")
    if not callable(getattr(material, "bind", None)) or not hasattr(material, "coefficient"):
        raise TypeError("material must provide a coefficient and bind(arithmetic)")
    references = OrderedDict()

    def reference(arithmetic):
        signature = arithmetic_signature(arithmetic)
        key = (id(arithmetic), signature)
        cached = references.get(key)
        if cached is not None:
            references.move_to_end(key)
            return cached
        bound = material.bind(arithmetic)
        native_delta, native_beta = arithmetic.scalar(delta), arithmetic.scalar(beta)
        if not arithmetic.isfinite(native_delta) or native_delta <= 0:
            raise ValueError("manufactured transmission cases require positive finite delta")
        if not arithmetic.isfinite(native_beta) or native_beta < 0:
            raise ValueError("manufactured beta must be finite and nonnegative")
        if not nonlinear:
            native_beta = arithmetic.scalar(0)
        one, zero = arithmetic.scalar(1), arithmetic.scalar(0)
        coefficients = [(one, zero, zero, zero)]
        delta_squared = native_delta**2
        if not arithmetic.isfinite(delta_squared) or delta_squared <= 0:
            raise ValueError("manufactured delta squared must remain finite and positive")
        for index, point in enumerate(bound.interfaces):
            previous = _reciprocal_derivatives(coefficients[-1], point, arithmetic)
            ratio = bound.trace(index, side="left")/bound.trace(index, side="right")
            amplitude = ((one+point)**4/(6*delta_squared)*(one-ratio)
                         *(previous[1]-delta_squared*previous[3]))
            shift = one+point
            addition = (amplitude, -3*amplitude*shift,
                        3*amplitude*shift**2, -amplitude*shift**3)
            coefficients.append(tuple(value+change for value, change in
                                      zip(coefficients[-1], addition)))
        result = _LayeredReference(bound, native_delta, native_beta,
                                   tuple(coefficients), signature)
        references[key] = result
        # Strong references are bounded; retained callbacks never switch their
        # owner when another solver binds the same case at another precision.
        if len(references) > 8:
            references.popitem(last=False)
        return result

    def derivatives(x, arithmetic):
        return reference(arithmetic).derivatives(x)

    def factory():
        return _LayeredFluxDescription(reference, material.coefficient, delta)

    kind = "nonlinear" if nonlinear else "linear"
    count = getattr(material, "interface_count", "custom")
    material_kind = getattr(material, "kind", "custom")
    return LayeredTransmissionCase(
        "layered-transmission:{}:{}:{}".format(material_kind, count, kind),
        "Known {} transmission solution ({} interfaces)".format(kind, count), ("galerkin",),
        derivatives, factory,
        notes=("Exact reference u,u',u'' and physical flux are continuous at every interface.",
               "Finite-resolution numerical flux traces are not continuity certificates."),
        material=material, _reference=reference,
    )
