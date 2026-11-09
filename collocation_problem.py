"""Adapt a scalar physical flux problem to the legacy collocation interface.

Only an explicitly constant epsilon=1 is supported.  The physical source is
unchanged; this adapter supplies the normalization and continuation factor
used by collocation.  A bound physical view owns its arithmetic, while the
existing collocation basis still uses its legacy global mpmath context.
"""

from dataclasses import dataclass

import numpy as np

from arithmetic import MpmathArithmetic, NumpyArithmetic
from physical_problem import BoundFluxProblem, FluxProblem


@dataclass(frozen=True)
class CollocationProblemAdapter:
    """Read-only numerical view of a physical problem for collocation.

    A finite interval requires two boundary conditions when delta=0, or four
    when delta>0.  On the half-line, one or two wall conditions respectively
    are required; an optional zero value at infinity is also accepted.
    Derivative conditions at infinity and nonzero far-field data are not
    supported by this adapter.
    """

    physical: BoundFluxProblem

    def __post_init__(self):
        if not isinstance(self.physical, BoundFluxProblem):
            raise TypeError("physical must be a BoundFluxProblem")
        self.validate_collocation_compatibility()
        self._validate_boundaries()

    @property
    def arithmetic(self):
        return self.physical.arithmetic

    def validate_collocation_compatibility(self):
        if not self.physical.epsilon.is_identity:
            raise ValueError(
                "collocation supports explicitly constant epsilon=1 only; "
                "nontrivial epsilon requires the Galerkin flux solver"
            )

    def _validate_boundaries(self):
        order = 2 if self.physical.delta == 0 else 4
        left, right = self.physical.boundary_conditions
        for boundary in (left, right):
            if any(value is not None for value in boundary[order:]):
                raise ValueError(
                    "a collocation problem of order {} requires boundary "
                    "derivative orders below {}".format(order, order)
                )
        if self.arithmetic.isfinite(self.physical.domain[1]):
            count = sum(value is not None for row in (left, right) for value in row)
            if count != order:
                raise ValueError(
                    "a finite-interval collocation problem requires exactly "
                    "{} boundary conditions".format(order)
                )
        else:
            if any(value is not None for value in right[1:]):
                raise ValueError("derivative boundary conditions at infinity are unsupported")
            if right[0] is not None and right[0] != 0:
                raise ValueError("the collocation adapter requires zero far-field data")
            if sum(value is not None for value in left) != order // 2:
                raise ValueError(
                    "a half-line collocation problem requires exactly {} wall "
                    "boundary conditions".format(order // 2)
                )

    def get_domain(self):
        """Return a fresh list for legacy basis code that copies/mutates it."""
        return list(self.physical.domain)

    def get_boundary_conditions(self):
        return [list(boundary) for boundary in self.physical.boundary_conditions]

    def _map(self, callback, *arguments):
        # Explicit dtype avoids vectorize's first-element inference, float64
        # narrowing of mpf values, and failures on empty argument arrays.
        arrays = np.broadcast_arrays(
            *(np.asarray(argument, dtype=object) for argument in arguments)
        )
        if arrays[0].shape == ():
            return callback(*(argument.item() for argument in arrays))
        dtype = object if self.arithmetic.use_mpmath else np.float64
        result = np.empty(arrays[0].shape, dtype=dtype)
        for index in np.ndindex(result.shape):
            result[index] = callback(*(argument[index] for argument in arrays))
        return result

    def base_solution(self, x):
        return self._map(self.physical.base_solution, x)

    def left_hand_side(self, x=None, u=None, ux=None, q=None):
        scalar = self.arithmetic.scalar
        if self.physical.delta == 0:
            return [None, None, scalar(1), None, scalar(0)]
        return [None, None, scalar(1) / self.physical.delta**2, None, scalar(-1)]

    def _source(self, callback, x, u, q):
        q = self.arithmetic.scalar(q)
        if not self.arithmetic.isfinite(q) or q < 0:
            raise ValueError("the homotopy parameter q must be finite and nonnegative")
        scale = self.left_hand_side()[2]

        def evaluate(coordinate, value):
            source = callback(coordinate, value)
            coordinate = self.arithmetic.scalar(coordinate)
            return scale * self.arithmetic.exp(-q * coordinate) * source

        return self._map(evaluate, x, u)

    def right_hand_side(self, x, u, q=0):
        return self._source(self.physical.physical_rhs, x, u, q)

    def right_hand_side_linearization(self, x, u, q=0):
        return self._source(self.physical.physical_rhs_u, x, u, q)


def prepare_collocation_problem(problem, *, use_mpmath=False, prec=100):
    """Bind a FluxProblem before the collocation basis reads its metadata.

    Existing legacy problem objects pass through unchanged.  Passing an
    already bound physical problem is rejected to avoid reusing parameters
    rounded at a different precision.  Reusing an adapter is supported only
    with its existing backend and precision.
    """
    if isinstance(problem, CollocationProblemAdapter):
        arithmetic = problem.arithmetic
        if (bool(use_mpmath) != arithmetic.use_mpmath
                or (use_mpmath and arithmetic.dps != prec)):
            raise TypeError(
                "the collocation adapter arithmetic does not match the solver; "
                "pass the unbound FluxProblem description"
            )
        return problem
    if isinstance(problem, BoundFluxProblem):
        raise TypeError(
            "pass the unbound FluxProblem description so collocation can bind "
            "it using the solver's arithmetic and precision"
        )
    if isinstance(problem, FluxProblem):
        arithmetic = MpmathArithmetic(prec) if use_mpmath else NumpyArithmetic()
        return CollocationProblemAdapter(problem.bind(arithmetic))
    return problem
