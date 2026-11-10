"""Public problem, solve, solution, and diagnostic interfaces.

Imports are lazy: importing the package does not initialize plotting or either
discretization. Advanced numerical components use package-qualified modules;
the former flat-module imports are no longer installed.
"""

from importlib import import_module
from importlib.metadata import version


_EXPORTS = {
    "BSKProblem": (".problems.bsk_problem", "BSKProblem"),
    "FluxProblem": (".problems.flux", "FluxProblem"),
    "BoundFluxProblem": (".problems.flux", "BoundFluxProblem"),
    "Coefficient": (".coefficients", "Coefficient"),
    "ConstantCoefficient": (".coefficients", "ConstantCoefficient"),
    "NumpyArithmetic": (".arithmetic", "NumpyArithmetic"),
    "MpmathArithmetic": (".arithmetic", "MpmathArithmetic"),
    "QuadratureFactory": (".quadrature", "QuadratureFactory"),
    "QuadraturePlan": (".quadrature", "QuadraturePlan"),
    "solve": (".solvers", "solve"),
    "save_solution": (".solution_io", "save_solution"),
    "load_solution": (".solvers", "load_solution"),
    "load_legacy_collocation": (".solvers", "load_legacy_collocation"),
    "project_collocation": (".projection", "project_collocation"),
    "CollocationSolution": (".collocation.solution", "CollocationSolution"),
    "GalerkinSolution": (".galerkin.solution", "GalerkinSolution"),
    "plot_fields": (".plotting", "plot_fields"),
    "plot_coefficients": (".plotting", "plot_coefficients"),
    "physical_l2_norm": (".norms", "physical_l2_norm"),
    "physical_h1_norm": (".norms", "physical_h1_norm"),
    "physical_l2_error": (".norms", "physical_l2_error"),
    "physical_h1_error": (".norms", "physical_h1_error"),
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
