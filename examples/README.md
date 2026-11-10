# User examples

Install the project first, then run these from the repository root:

```bash
python -m pip install -e ".[test,notebook]"
python -m examples.collocation --backend both
python -m examples.galerkin --backend both
python -m examples.variable_epsilon --backend both
python -m examples.restart_and_projection --backend both
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

The earlier `flux_problem_collocation`, `flux_problem_galerkin`, and
`common_solver_example` module names remain compatibility entry points. Their
tuple arities are retained; result entries use the common solution and
quadrature-history interfaces.

The four original notebooks are in [`notebooks/`](notebooks/). Their research
parameters and sweep sizes are retained, and stale outputs are cleared. They
import the installed package without changing directories. Set
`OVERSCREENING_OUTPUT_DIR` to choose where saved files go. Ordinary solves and
typed restarts use the common interface. Explicitly labelled adjoint and
unsupported boundary/operator experiments retain the canonical legacy
collocation path. Those full sweeps are opt-in; CI executes the bounded scripts
and checks notebook structure without running the expensive cells.
