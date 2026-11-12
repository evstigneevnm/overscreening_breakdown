"""Physical-coordinate norms for finite and half-line quadrature plans.

These integrate in dx using the supplied plan's physical nodes and weights.
Solution methods additionally check ownership, domain, and interface splits.
"""

from .galerkin.solution import (
    physical_h1_error,
    physical_h1_norm,
    physical_l2_error,
    physical_l2_norm,
)

__all__ = [
    "physical_h1_error",
    "physical_h1_norm",
    "physical_l2_error",
    "physical_l2_norm",
]
