"""Compatibility alias for the original flat-module import."""

import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module(
    "overscreening_breakdown.problems.flux"
)
