# Public API and breaking import migration

The public interface is now the installed `overscreening_breakdown` package.
The flat compatibility modules, lowercase package-level BSK class names, and
three Python example wrappers have been removed intentionally. There is no
compatibility import layer for those names.

## Define a physical problem

Use the immutable `BSKProblem` for the existing physical model, with parameters
provided at construction:

```python
from overscreening_breakdown import BSKProblem, solve

problem = BSKProblem(
    sigma="10", mu="0", gamma="0.5", delta="1", epsilon=1,
    boundary_conditions=(("1", None, None, "0", None), (None,)*5),
)
solution = solve(problem, 8, solver="collocation", tolerance="1e-10")
print(solution.u(1), solution.w(1), solution.flux(1))
```

The constructor accepts `sigma`, `mu`, `gamma`, `delta`, `epsilon`, `domain`,
`boundary_conditions`, and optional `initial_guess(x, arithmetic)`. Numeric
inputs remain unconverted until binding to the solver's arithmetic. Use
integers or decimal strings for high-precision inputs. Create a new problem
when changing parameters; there is no `set_parameters` method on `BSKProblem`.

Boundary rows list derivative orders 0 through 4 at the two domain endpoints;
`None` leaves a condition unspecified. The old `u0` and `u0xxx` arguments become
entries 0 and 3 of the wall row. A half-line fourth-order mixed problem uses
`u(0)=u_b` and `u'''(0)=q_b`. The defaults remain wall value 100, third derivative
zero, and domain `[0,infinity)`. The Gaussian and physical-source normalization
remain the model's existing ones.

There is no separate public segment BSK class. Supply the finite domain and
boundary conditions explicitly; only collocation supports this domain:

```python
finite_problem = BSKProblem(
    delta=0, domain=("0", "1"),
    boundary_conditions=(("1", None, None, None, None),
                         ("0", None, None, None, None)),
)
finite_solution = solve(finite_problem, 8, solver="collocation")
```

Use `FluxProblem(delta=..., f=..., f_u=..., ...)` for a different physical
source. Its callbacks take `(x, u, arithmetic)` and return unscaled physical
values. `Coefficient` describes a nontrivial epsilon with explicit positive
bounds, wall trace and interface positions; `ConstantCoefficient` describes a
constant. Collocation accepts explicitly constant epsilon one. Variable or
non-unit epsilon requires the Galerkin solver and its supported boundary data.

## Solve, inspect, and save

Use the common solver's keyword interface:

```python
solution = solve(
    problem, 8, solver="collocation", L="1", use_mpmath=True,
    precision=50, method="newton", tolerance="1e-35",
)
```

`use_mpmath=False` selects NumPy float64; `True` selects a solver-owned mpmath
context with the requested decimal `precision`. Python versus gmpy mpmath is a
process-level choice made before import. The NumPy collocation methods remain
`newton`, `root`, `root_no_jac`, and `nonlin_solve`; mpmath and Galerkin use
Newton. No automatic method or solver selection depends on epsilon.

Use `verbosity` for live diagnostics from either discretization and arithmetic:

```python
solution = solve(problem, 8, solver="collocation", verbosity=2)
```

| Level | Solver text |
| --- | --- |
| `0` | Quiet; the library default |
| `1` | Solve information, major phases and stages, refinement decisions, final outcome |
| `2` | Level 1 plus accepted Newton/adjoint steps and periodic SciPy evaluations |
| `3` | Level 2 plus rejected trials, backtracking, recovery reasons, Jacobian and linear-solve phases |

`verbosity` must be an integer from 0 through 3; booleans are rejected. The old
`verbose` keyword has been removed: replace `verbose=True` with `verbosity=2`
and `verbose=False` with `verbosity=0`. There is no compatibility alias.
Notebook helpers and example command lines use level 2 by default. Example
`run()` functions retain the quiet library default. Scripts accept
`--verbosity 0` through `--verbosity 3`; their final scientific result reports
remain available even when solver progress is quiet.

An optional observer receives immutable `ProgressEvent` objects independently
of console verbosity:

```python
events = []
solution = solve(problem, 8, verbosity=0, progress_callback=events.append)
for event in events:
    print(event.kind, event.solver, event.method, event.elapsed, event.data)
```

`event.data` is a read-only mapping of native numerical scalars and immutable
metadata, with no coefficient arrays or live arithmetic contexts. No event
history is retained unless the observer collects it. `progress_interval=1.0`
sets the minimum time between frequent evaluation/trial text messages; zero
disables throttling. Phase transitions and final outcomes are always printed
at their selected level, and observer delivery is not throttled. Observer
errors propagate as `ProgressCallbackError`, retaining the original exception
as their cause; they are not treated as recoverable numerical failures.

Residual messages identify their norm and target: collocation uses RMS and
Galerkin uses the Euclidean norm. SciPy HYBR and `fsolve` report actual
function/Jacobian evaluations rather than accepted Newton iterations. Their
stop statuses remain distinct from independently verified convergence.
See the [numerical guide](numerical_methods.md#live-progress-reporting) for
phase timing, precision, and the reporting contract.

Initialization, Newton recovery, and homotopy have separate controls:

| Control | Values and scope |
| --- | --- |
| `initializer` | `None` (default) or `"fsolve"`; NumPy collocation only |
| `initializer_tolerance` | Positive fsolve step tolerance, default `"0.1"` |
| `initializer_max_evaluations` | Hard initializer residual/Jacobian evaluation budget, default `200` |
| `newton_fallback` | `None` (default) or `"adjoint"`; Newton with either solver and arithmetic |
| `adjoint_max_steps` | Maximum accepted recovery steps per fixed homotopy stage, default `20` |
| `use_globalization` | Existing homotopy control; independent of initializer and fallback |

For example, `solve(problem, 8, initializer="fsolve")` requests a bounded SciPy
initialization before the selected NumPy collocation solve. It does not certify
final convergence. A native mpmath or mixed solve can instead enable
`newton_fallback="adjoint"` while retaining its arithmetic. Requesting fsolve
with mpmath or Galerkin raises an error. Adjoint recovery requires
`method="newton"` and is attempted only when an ordinary Newton step cannot be
accepted. See the [merit and constraints](numerical_methods.md#newton-recovery-and-initialization).

Results are `CollocationSolution` or `GalerkinSolution`, exposing `u`, `w`,
`u_dx`, `w_dx`, `flux`, coefficients, actual unknown count, and convergence
metadata. Galerkin requires both `nonlinear_converged` and
`quadrature_converged`; these are distinct from physical approximation error.
See the [numerical guide](numerical_methods.md) for all solver and quadrature
controls, low-level assembly, reference fields, and physical norms.
`initialization_result` records an explicitly requested initializer;
`newton_result` records actual adjoint steps separately from Newton attempts.
For SciPy collocation methods, `scipy_result` retains SciPy's success flag,
status, message and evaluation counts alongside the verified final residual
and its acceptance target. The solver recomputes the complete residual at
`q=0`, including boundary equations; convergence requires finite coefficients
and a finite residual RMS no larger than `tolerance*sqrt(unknown_count)`.
This is the historical collocation Newton residual target. SciPy's step-based
termination flag alone does not determine convergence. `scipy_result` is
`None` for Newton and Galerkin solves. This algebraic check does not establish
spatial accuracy; manufactured field and boundary checks remain separate.
These diagnostics are available during the process and are not serialized in
the saved-solution format.

The package root exports these supported groups:

| Purpose | Names |
| --- | --- |
| Physical descriptions | `BSKProblem`, `FluxProblem`, `BoundFluxProblem`, `Coefficient`, `ConstantCoefficient` |
| Arithmetic and integration | `NumpyArithmetic`, `MpmathArithmetic`, `QuadratureFactory`, `QuadraturePlan` |
| Solving and results | `solve`, `CollocationSolution`, `GalerkinSolution` |
| Data and explicit conversion | `save_solution`, `load_solution`, `load_legacy_collocation`, `project_collocation` |
| Physical diagnostics | `physical_l2_norm`, `physical_h1_norm`, `physical_l2_error`, `physical_h1_error` |
| Rendering | `plot_fields`, `plot_coefficients` |
| Live progress | `ProgressEvent` |

Importing the package is lazy. Low-level numerical components remain available
through canonical submodules for assembly work and explicit legacy experiments.
They do not require flat modules installed beside the package.

## Replace old imports

For user code, prefer the root names above. The table also gives canonical
paths for low-level code that needs the earlier classes or helpers:

| Removed import/module | Replacement |
| --- | --- |
| `from overscreening_breakdown import overscreening_breakdown` | `from overscreening_breakdown import BSKProblem` and constructor parameters |
| `overscreening_breakdown_segment` | `BSKProblem(domain=..., boundary_conditions=...)` |
| `physical_problem` | `overscreening_breakdown.problems.flux` or root `FluxProblem` |
| `arithmetic` | `overscreening_breakdown.arithmetic` or root arithmetic classes |
| `coefficients` | `overscreening_breakdown.coefficients` or root coefficient classes |
| `solvers` | `overscreening_breakdown.solvers` or root `solve` |
| `solution_io` | `overscreening_breakdown.solution_io` or root save/load functions |
| `projection` | `overscreening_breakdown.projection` or root `project_collocation` |
| `solution_plotting` | `overscreening_breakdown.plotting` or root plotting functions |
| `spectral1d` | `overscreening_breakdown.collocation.spectral1d` |
| `collocation_problem` | `overscreening_breakdown.collocation.adapter` |
| `collocation_solution` | `overscreening_breakdown.collocation.solution` |
| `galerkin` | `overscreening_breakdown.galerkin.basis` |
| `galerkin_assembly` | `overscreening_breakdown.galerkin.assembly` |
| `solutions` | `overscreening_breakdown.galerkin.solution`; physical helpers also in `overscreening_breakdown.norms` |
| `nonlinear_solver` | `overscreening_breakdown.nonlinear_solver` |
| `rational_mapping` | `overscreening_breakdown.rational_mapping` |
| `quadrature` | `overscreening_breakdown.quadrature` or root quadrature types |
| `manufactured_problems` | Prefer `overscreening_breakdown.manufactured` catalog; historical definitions in `.manufactured.legacy_problems` |
| `manufactured_solutions_solver` | `overscreening_breakdown.manufactured.legacy_runner` for explicit historical plots |
| `problems_for_updated_testing` | `overscreening_breakdown.manufactured.legacy_additional` |
| `nonlinear_problem_on_segment` | `overscreening_breakdown.problems.segment` |

The removed example wrappers map as follows:

| Removed command | Current command |
| --- | --- |
| `python -m examples.flux_problem_collocation` | `python -m examples.collocation` |
| `python -m examples.flux_problem_galerkin` | `python -m examples.variable_epsilon` |
| `python -m examples.common_solver_example` | `python -m examples.restart_and_projection` |

`examples.galerkin` demonstrates the constant-epsilon mixed solve. The four
notebooks remain under `examples/notebooks/` and use the current public names.

## Saved-data compatibility

The JSON schema, coefficient interpretation, arithmetic precision preservation,
and explicit one-way collocation-to-Galerkin projection are unchanged by this
namespace cleanup. Supply the physical description again when loading; source
callbacks are never serialized or identity-certified. Solver mismatches remain
errors, and no automatic conversion occurs.

Trusted numerical legacy pickle payloads—bare numerical arrays or version-1
coefficient dictionaries—remain readable with canonical
`load_legacy_collocation(path, problem=..., N=..., L=...)`. Pickles of project
Python objects that depend on removed module/class globals are unsupported.
This distinction does not change the numerical coefficient reader. See the
[saved-solution guide](saved_solutions.md) for exact contracts and precision
conversion behavior.
