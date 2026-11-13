"""Owned-arithmetic snapshots of legacy collocation coefficient vectors."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

from overscreening_breakdown.collocation.adapter import CollocationProblemAdapter
from overscreening_breakdown.arithmetic import arithmetic_signature, require_precision
from overscreening_breakdown.problems.flux import BoundFluxProblem
from overscreening_breakdown.rational_mapping import RationalHalfLineMap
from overscreening_breakdown.galerkin.solution import _sum, _distance_polynomial


@dataclass(frozen=True, init=False, eq=False)
class CollocationSolution:
    """Evaluate a collocation polynomial without the legacy global context.

    ``w`` is derived as -u'', whereas the mixed Galerkin solution stores an
    independent w field. A collocation polynomial may have a nonzero value at
    infinity; that value is retained rather than silently changing coefficients.
    """

    physical: BoundFluxProblem = field(repr=False)
    N: int
    L: Any
    raw_L: Any = field(repr=False)
    method: str
    converged: bool
    nonlinear_converged: bool
    newton_result: Any = field(repr=False)
    initialization_result: Any = field(repr=False)
    scipy_result: Any = field(repr=False)
    _entries: tuple = field(repr=False)
    solver_type = "collocation"
    plan = None
    quadrature_converged = None
    quadrature_history = ()
    projection_info = ()

    def __init__(self, *, physical, coefficients, N, L=1, method="newton",
                 converged=False, nonlinear_converged=None, newton_result=None,
                 initialization_result=None, scipy_result=None):
        if not isinstance(physical, BoundFluxProblem):
            raise TypeError("physical must be a BoundFluxProblem")
        physical.check_precision()
        CollocationProblemAdapter(physical)
        if isinstance(N, bool) or not isinstance(N, Integral) or N < 1:
            raise ValueError("N must be a positive integer")
        arithmetic = physical.arithmetic
        mapping = RationalHalfLineMap(arithmetic, L)
        entries = tuple(arithmetic.scalar(value) for value in coefficients)
        count = int(N)+sum(value is not None for row in physical.boundary_conditions for value in row)
        if len(entries) != count or any(not arithmetic.isfinite(value) for value in entries):
            raise ValueError("collocation coefficients must be finite and match N plus boundary rows")
        for name, value in (
            ("physical", physical), ("N", int(N)), ("L", mapping.L), ("raw_L", L),
            ("mapping", mapping), ("method", method), ("converged", bool(converged)),
            ("nonlinear_converged", bool(converged) if nonlinear_converged is None
             else bool(nonlinear_converged)), ("newton_result", newton_result),
            ("initialization_result", initialization_result),
            ("scipy_result", scipy_result),
            ("_entries", entries), ("_signature", arithmetic_signature(arithmetic)),
            ("_tail", _distance_polynomial(arithmetic, entries)),
        ):
            object.__setattr__(self, name, value)

    @property
    def arithmetic(self):
        return self.physical.arithmetic

    @property
    def domain(self):
        return self.physical.domain

    @property
    def unknown_count(self):
        return len(self._entries)

    @property
    def coefficients(self):
        self.check_precision()
        return self.arithmetic.vector(self._entries)

    @property
    def u_coefficients(self):
        return self.coefficients

    def check_precision(self):
        self.physical.check_precision()
        self.mapping.check_precision()
        require_precision(self.arithmetic, self._signature, subject="solution",
                          rebuild="load a new snapshot")

    def _coordinate(self, x):
        self.check_precision()
        x = self.arithmetic.scalar(x)
        if self.domain[1] == self.arithmetic.inf:
            return self.mapping.physical_coordinate(x)
        if not self.arithmetic.isfinite(x) or not self.domain[0] <= x <= self.domain[1]:
            raise ValueError("physical coordinate lies outside the finite domain")
        return x

    def _derivative_scalar(self, x, order=0):
        x = self._coordinate(x)
        arithmetic = self.arithmetic
        zero, one = arithmetic.scalar(0), arithmetic.scalar(1)
        if self.domain[1] == arithmetic.inf:
            if x == arithmetic.inf:
                return _sum(arithmetic, self._entries) if order == 0 else zero
            ratio = self.L/x if x >= self.L else x/self.L
            r = 2*(ratio/(1+ratio)) if x >= self.L else 2/(1+ratio)
            chain = (r*(r/2))/self.L if order >= 1 else zero
            second = -(chain*r)/self.L if order >= 2 else zero
            third = -(3*second*r)/(2*self.L) if order >= 3 else zero
        else:
            length = self.domain[1]-self.domain[0]
            r = 2*((self.domain[1]-x)/length)
            chain, second, third = 2/length, zero, zero
        y = one-r
        if r*len(self._entries)**2 < arithmetic.scalar("0.25"):
            # Horner evaluation in distance from infinity retains decaying
            # tails after y itself rounds to 1.
            values = []
            coefficients = self._tail
            for derivative in range(4):
                value = zero
                for entry in reversed(coefficients):
                    value = value*r+entry
                values.append(value if derivative % 2 == 0 else -value)
                coefficients = tuple((index+1)*entry for index, entry in enumerate(coefficients[1:]))
        else:
            t, d, dd, ddd = [one], [zero], [zero], [zero]
            if len(self._entries) > 1:
                t.append(y); d.append(one); dd.append(zero); ddd.append(zero)
            for k in range(2, len(self._entries)):
                t.append(2*y*t[-1]-t[-2])
                d.append(2*t[-2]+2*y*d[-1]-d[-2])
                dd.append(4*d[-2]+2*y*dd[-1]-dd[-2])
                ddd.append(6*dd[-2]+2*y*ddd[-1]-ddd[-2])
            values = [_sum(arithmetic, (entry*value for entry, value in zip(self._entries, sequence)))
                      for sequence in (t, d, dd, ddd)]
        result = (values[0] if order == 0 else values[1]*chain if order == 1 else
                  values[2]*chain**2+values[1]*second if order == 2 else
                  values[3]*chain**3+3*values[2]*chain*second+values[1]*third)
        if not arithmetic.isfinite(result):
            raise OverflowError("collocation field evaluation is not finite")
        return result

    def _evaluate(self, x, order=0, sign=1, flux=False):
        self.check_precision()
        inputs = np.asarray(x, dtype=object)
        output = np.empty(inputs.shape, dtype=object if self.arithmetic.use_mpmath else np.float64)
        for index in np.ndindex(inputs.shape):
            coordinate = self._coordinate(inputs[index])
            if flux:
                value = (self.arithmetic.scalar(0) if coordinate == self.arithmetic.inf else
                         self.physical.epsilon(coordinate)*(
                             self._derivative_scalar(coordinate, 1)
                             -self.physical.delta**2*self._derivative_scalar(coordinate, 3)))
            else:
                value = sign*self._derivative_scalar(coordinate, order)
            if not self.arithmetic.isfinite(value):
                raise OverflowError("collocation field evaluation is not finite")
            output[index] = value
        return output[()] if inputs.ndim == 0 else output

    def u(self, x):
        return self._evaluate(x)

    def w(self, x):
        return self._evaluate(x, 2, -1)

    def u_dx(self, x):
        return self._evaluate(x, 1)

    def w_dx(self, x):
        return self._evaluate(x, 3, -1)

    def flux(self, x):
        return self._evaluate(x, flux=True)

    def save(self, path):
        from overscreening_breakdown.solution_io import save_solution
        return save_solution(self, path)
