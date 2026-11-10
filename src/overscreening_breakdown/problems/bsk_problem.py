"""Arithmetic-independent BSK parameters for the common physical solver API."""

from dataclasses import dataclass, field
from typing import Any

from overscreening_breakdown.coefficients import as_coefficient
from overscreening_breakdown.problems.bsk import _physical_rhs, _physical_rhs_u
from overscreening_breakdown.problems.flux import FluxProblem, _copy_boundaries


@dataclass(frozen=True, init=False)
class BSKProblem:
    """Describe the existing BSK physical source without numerical state.

    Parameters retain their original values until ``bind(arithmetic)``. Use
    integers, decimal strings, or Decimal inputs for high-precision data.
    ``initial_guess(x, arithmetic)`` follows the same scalar callback contract
    as FluxProblem. Domain and boundary data are explicit; selecting delta=0
    does not silently remove supplied fourth-order boundary conditions.

    The Gaussian retains the existing model normalization, without a 1/sigma
    factor. Arithmetic and precision are selected by the solver, not here.
    """

    sigma: Any
    mu: Any
    gamma: Any
    delta: Any
    epsilon: Any
    domain: tuple
    boundary_conditions: tuple
    initial_guess: Any = field(repr=False, compare=False)

    def __init__(self, *, sigma="10", mu="0", gamma="0.5", delta="50",
                 epsilon=1, domain=(0, "inf"),
                 boundary_conditions=((100, None, None, 0, None), (None,)*5),
                 initial_guess=None):
        if initial_guess is not None and not callable(initial_guess):
            raise TypeError("initial_guess must be callable or None")
        domain = tuple(domain)
        if len(domain) != 2:
            raise ValueError("domain must contain two endpoints")
        boundaries = _copy_boundaries(boundary_conditions)
        coefficient = as_coefficient(epsilon)
        for name, value in (("sigma", sigma), ("mu", mu), ("gamma", gamma),
                            ("delta", delta), ("epsilon", coefficient), ("domain", domain),
                            ("boundary_conditions", boundaries), ("initial_guess", initial_guess)):
            object.__setattr__(self, name, value)

    def bind(self, arithmetic):
        """Create an independent native physical snapshot for one solver."""
        sigma, mu, gamma = (arithmetic.scalar(value)
                            for value in (self.sigma, self.mu, self.gamma))
        if not all(arithmetic.isfinite(value) for value in (sigma, mu, gamma)):
            raise ValueError("sigma, mu and gamma must be finite")
        if sigma <= 0 or gamma < 0:
            raise ValueError("the physical BSK model requires sigma > 0 and gamma >= 0")
        return FluxProblem(
            delta=self.delta, epsilon=self.epsilon, domain=self.domain,
            boundary_conditions=self.boundary_conditions, initial_guess=self.initial_guess,
            f=lambda x, u, a: _physical_rhs(x, u, a, sigma, mu, gamma),
            f_u=lambda x, u, a: _physical_rhs_u(x, u, a, sigma, mu, gamma),
        ).bind(arithmetic)
