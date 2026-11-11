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

Unsupported delta-zero plus third-derivative wall-data sweeps live separately
in [`legacy_boundary_experiments.ipynb`](../experiments/notebooks/legacy_boundary_experiments.ipynb)
with standalone canonical legacy setup. The general-operator notebook remains
explicitly legacy. The unused historical SciPy/VODE coefficient flow is retained
as [`experiments/legacy_adjoint_flow.py`](../experiments/legacy_adjoint_flow.py),
outside the native nonlinear solver. Full research sweeps are opt-in; CI checks
the notebook setup and bounded comparison without running the expensive cells.
