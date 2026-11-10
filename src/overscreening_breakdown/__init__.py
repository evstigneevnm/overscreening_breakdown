"""Half-line collocation and mixed Galerkin solvers.

The exports retain their existing signatures. Imports are lazy so importing a
numerical component does not initialize plotting or either discretization.
"""

from importlib import import_module
from importlib.metadata import version


_EXPORTS = {
    "FluxProblem": (".problems.flux", "FluxProblem"),
    "BoundFluxProblem": (".problems.flux", "BoundFluxProblem"),
    "Coefficient": (".coefficients", "Coefficient"),
    "ConstantCoefficient": (".coefficients", "ConstantCoefficient"),
    "overscreening_breakdown": (".problems.bsk", "overscreening_breakdown"),
    "overscreening_breakdown_segment": (".problems.bsk", "overscreening_breakdown_segment"),
    "solve": (".solvers", "solve"),
    "load_solution": (".solvers", "load_solution"),
    "load_legacy_collocation": (".solvers", "load_legacy_collocation"),
    "project_collocation": (".projection", "project_collocation"),
    "CollocationSolution": (".collocation.solution", "CollocationSolution"),
    "GalerkinSolution": (".galerkin.solution", "GalerkinSolution"),
    "plot_fields": (".plotting", "plot_fields"),
    "plot_coefficients": (".plotting", "plot_coefficients"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name == "__version__":
        return version("overscreening-breakdown")
    if name not in _EXPORTS:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    module_name, attribute = _EXPORTS[name]
    result = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = result
    return result


def __dir__():
    return sorted(set(globals()) | set(__all__) | {"__version__"})
