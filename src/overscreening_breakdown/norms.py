"""Physical-coordinate norm helpers with their existing call signatures."""

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
