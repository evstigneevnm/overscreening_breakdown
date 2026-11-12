# User examples

Install the project first, then run these from the repository root:

```bash
python -m pip install -e ".[test,notebook]"
python -m examples.collocation --backend both
python -m examples.galerkin --backend both
python -m examples.variable_epsilon --backend both
python -m examples.restart_and_projection --backend both
python -m examples.bsk --solver both --backend both
```

The command lines display live solver progress at `--verbosity 2`. Use
`--verbosity 0` for quiet solver output, `1` for phase summaries, or `3` for
trial and linear-solve details. Final example data reports still appear at
level 0. The Python `run()` functions default to `verbosity=0` and also accept
`progress_callback` and `progress_interval` when embedding an example.

Each script returns native solution objects from the common `solve` entry point.
Both nonlinear and quadrature convergence flags are reported. Collocation has
no quadrature convergence check, so that flag is `None`. Galerkin stores
independent `u` and `w` fields and requires both checks to pass.

The examples use the reusable manufactured-case catalog. Their physical source,
derivative, exact fields and coefficient descriptions are defined together in
the installed package; examples contain no separate copies of those formulas.
The constant-coefficient case can use either solver. The smooth coefficient
and the `--coefficient jump` case require Galerkin. The jump example uses a
global basis at modest resolution, so quadrature/Newton convergence alone does
not establish a small physical field error. Its NumPy quadrature comparison
uses an explicit absolute target of `1e-9` above the observed assembly rounding
floor, separately from the `1e-11` Newton target.

Pass `--output-dir /tmp/overscreening-example` to retain typed JSON snapshots and
field/coefficient PDFs. Scripts also run by absolute path from another working
directory once the package is installed:

```bash
python /path/to/project/examples/galerkin.py --backend mpmath --precision 50 \
  --output-dir /tmp/overscreening-example
```

`restart_and_projection.py` demonstrates a same-solver file restart, explicit
collocation-to-Galerkin projection, the subsequent nonlinear solve, and saving
both independent mixed fields. Projected fields carry `converged=False` until
the mixed equations and quadrature checks pass. A Galerkin snapshot cannot be
loaded as collocation data.

`bsk.py` constructs an immutable `BSKProblem` directly, with explicit physical
parameters and boundary values, and reuses it across both discretizations and
arithmetic backends. To change a problem, create a new description. A finite
segment uses the same class with an explicit domain and endpoint data. The
Gaussian-forced example keeps a separate `1e-12` quadrature consistency target;
50-digit arithmetic and a tighter Newton target do not themselves establish
50-digit integration accuracy.

Use the module names listed above. The former `flux_problem_collocation`,
`flux_problem_galerkin`, and `common_solver_example` entry points have been
removed, together with the flat compatibility import modules.

## Discontinuous Galerkin notebook

[`execution_example_galerkin_discontinuous.ipynb`](notebooks/execution_example_galerkin_discontinuous.ipynb)
is a guided workflow for layered materials. Its input cell selects a material
with 1, 2, 3, 5, 7, or 10 interfaces, a BSK or manufactured source, and an
arithmetic/integration profile. Every solve declares its resolution explicitly:
`N_selected`, `N_linear`, `N_nonlinear`, and the optional examples' own `N` values.
Each also declares a required `newton_atol`, the Newton absolute residual
tolerance. The adjacent quoted decimal settings make each target editable,
including a separate target for the finer resolution and the preset sweep.
Galerkin uses `N-1` potential-correction coefficients and `N` auxiliary-field
coefficients, giving `2*N-1` global unknowns across all layers. A small material
table can replace the presets with user-specified
positions and epsilon values. Both constant layers and smooth variation within
each layer are supported. An advanced section accepts an arbitrary positive,
bounded coefficient with declared interfaces.

Physical-definition cells visibly construct `BSKProblem` and the two
manufactured `FluxProblem` objects, including `f`, `f_u`, boundary conditions,
and initial guesses. Edit those definitions to change the equations.
`run_defined_problem(problem, ...)` passes the supplied object to the common
solver; its model label controls reporting. Resolution studies reuse the same
problem, and the sweep calls the visible definition functions for each material.

With optional sections switched off, **Run All** performs one selected solve
and two known-answer checks.
It plots the material, potential, auxiliary field, physical flux, convergence,
and manufactured errors. Reports distinguish nonlinear convergence, integration
verification, and spatial resolution. Optional cells enable a two-resolution
comparison, native mpmath arithmetic, and all 18 material/source combinations.
The comparison requires an explicit `fine_N`; the sweep uses an editable
`N_by_interfaces` table. The `standard`, `strict_integration`, and
`high_precision` profiles never change `N` or `newton_atol`. Galerkin Newton
uses a Euclidean discrete-residual target with zero relative tolerance; the
target controls the algebraic solve independently of physical field error.
Reports and convergence plots show the target, and exports retain it.
Automatic starting quadrature uses
`2*N+4` points per panel and retains independent refinement checks. Experiment
plot titles and exported settings record the actual resolution.

Separate cells after each manufactured check show absolute and
percentage-relative physical `L2(dx)` errors for `u`, `w`, and flux. Optional
linear and nonlinear convergence studies use editable `N_values_linear` and
`N_values_nonlinear` lists with explicit Newton targets. Enable their individual
switches to compute the tables; both are initially off. The notebook explains
why interface flux-error peaks can remain large while integrated errors fall.

The optional **adaptive homotopy recovery** section has its own explicit
material, nonlinear problem, initial guess, resolution, and work limits.
Set `run_homotopy_demo=True` to compare a short-budget direct solve, reference
continuation, and a larger-budget direct Newton control. Separate cells show
the reference and accepted/rejected stages, parameter path, residuals, fields,
and known-answer errors. This demonstrates recovery from a short stage budget;
the direct control can also converge, so it does not establish a speed advantage.

By default it saves settings, typed solution JSON, decimal profile CSV, and PDFs
under `notebook-output/galerkin-discontinuous` in the kernel's working directory.
Set `OVERSCREENING_OUTPUT_DIR` before starting the kernel to choose another
location, or set `save_results=False` to disable exports. Rerunning the same
case replaces its exports. Custom executable coefficient functions must be
supplied again when reloading a solution.

The four original notebooks are in [`notebooks/`](notebooks/). Their research
parameters and sweep sizes are retained, and stale outputs are cleared. They
import the installed package without changing directories. Set
`OVERSCREENING_OUTPUT_DIR` to choose where saved files go. Ordinary solves and
typed restarts use the common interface and immutable `BSKProblem` descriptions.
The ordinary execution, finite-segment, and mu-sweep notebooks use only public
`solve` calls. Their historical "adjoint initialization" was SciPy fsolve;
it is now named `initializer="fsolve"` and requested only for NumPy
collocation. The 120-digit experiment explicitly uses `initializer=None`.
`newton_fallback="adjoint"` enables a separate native recovery method in either
solver and arithmetic. A bounded comparison reports actual recovery counts,
which can be zero when Newton succeeds directly. `use_globalization` retains
its independent homotopy meaning.

The three ordinary notebook `run_collocation()` helpers default to
`verbosity=2` and forward `progress_callback` and `progress_interval`. Set
`verbosity=0` on an individual solve or in the helper default to suppress
live solver text during a large sweep. Direct restarts and the bounded
collocation/Galerkin comparison also select live reporting. The former
notebook `status()` printer is removed; the shared reporter prints the final
outcome. Inspect `result.initialization_result`, `result.newton_result`, or
`result.scipy_result` directly for additional full-precision diagnostics.

Unsupported delta-zero plus third-derivative wall-data sweeps live separately
in [`legacy_boundary_experiments.ipynb`](../experiments/notebooks/legacy_boundary_experiments.ipynb)
with standalone canonical legacy setup. The general-operator notebook remains
explicitly legacy. The unused historical SciPy/VODE coefficient flow is retained
as [`experiments/legacy_adjoint_flow.py`](../experiments/legacy_adjoint_flow.py),
outside the native nonlinear solver. Full research sweeps are opt-in; CI checks
the notebook setup and bounded comparison without running the expensive cells.
