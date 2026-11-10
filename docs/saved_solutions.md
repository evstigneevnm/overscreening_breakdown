# Saved solutions and explicit conversion

`solution.save(path)` or `overscreening_breakdown.save_solution(solution, path)` writes a
data-only JSON snapshot. It records the solver type, basis/version, actual
unknown count, mapping/domain, exact working precision, physical metadata,
coefficient fields, quadrature geometry, final convergence flags, and any
projection provenance. Mpmath values use exact binary mantissas and exponents;
float64 uses hexadecimal values, preserving signed zero. Decimal mapping and
interface inputs are retained for a later precision change.

```python
from overscreening_breakdown import load_solution, solve
from overscreening_breakdown.manufactured import rational_case

problem = rational_case(nonlinear=True).make_problem()
mixed = solve(problem, 4, solver="galerkin", use_mpmath=True,
              precision=50, tolerance="1e-35")
mixed.save("mixed.json")
restored = load_solution("mixed.json", problem=problem, solver="galerkin")
```

Supply the unbound physical problem again: executable callbacks are never
serialized. Loading verifies its domain, delta, boundary data, epsilon
description, and sampled epsilon values at the saved arithmetic precision.
Sampling does not certify arbitrary callback identity, and source `f`/`f_u`
callbacks are supplied by the caller. Loading reconstructs fields without
solving. An expected solver mismatch is rejected in either direction; loading
never projects coefficients automatically.

The default restores the saved NumPy/mpmath arithmetic mode and exact bit
precision, including a noncanonical bit count. The mpmath Python/gmpy
implementation follows the current process; loading cannot switch it.
Explicit `use_mpmath` and `precision` overrides can
round data. A backend or working-bit change clears all convergence flags;
otherwise the saved flags remain provenance rather than a new numerical
certificate. Newton iteration objects and quadrature iteration histories are
not serialized. Loading never changes global mpmath precision.

The existing collocation `save_solution` still writes its compatible pickle
format, now tagged `solver_type="collocation"`. Its reader accepts older
untagged payloads and bare arrays and rejects tagged Galerkin data. Import
trusted legacy coefficients with explicit missing geometry using
`overscreening_breakdown.load_legacy_collocation(path, problem=problem, N=..., L=...)`. The
JSON loader does not read pickle files. An imported vector is not marked as a
converged solution.

The public namespace cleanup does not change these numerical file formats.
Bare numerical arrays and version-1 coefficient dictionaries remain readable
through that canonical legacy importer. Pickles of project Python objects
that refer to removed globals such as `spectral1d.solve_nonlinear_problem` or
`overscreening_breakdown.overscreening_breakdown` are no longer supported.
No import shim or automatic class-name remapping is installed for them.

For the permitted one-way conversion, use `overscreening_breakdown.project_collocation`:

```python
from overscreening_breakdown import project_collocation

collocation = solve(problem, 8, solver="collocation", use_mpmath=True,
                    precision=50, tolerance="1e-35")
restart = project_collocation(collocation, N=4, L="1", trace_tolerance="1e-35")
mixed = solve(
    problem, 4, solver="galerkin", L="1", use_mpmath=True,
    precision=50, tolerance="1e-35", initial_guess=restart,
)
```

This projects the corrected collocation field and its negative second
derivative independently in physical H1. It retains the source's arithmetic
owner and physical problem. The source must satisfy the mixed half-line,
epsilon-one, delta-positive boundary requirements. Tiny wall/far-field defects
are explicitly corrected before integration and recorded in `projection_info`;
defects above the trace tolerance are rejected. A projected restart has all
convergence flags false until the target equations are solved and checked.
There is no reverse conversion. If mapping length changes, projection loads
are rational and its fixed rule requires a separate integration assessment;
`quadrature_order` and `subdivisions` control that rule.
The default trace tolerance is scaled machine roundoff; the explicit threshold
in the example permits the stated normalization at the collocation solve's
tolerance and records the applied corrections.

Run the common-interface [example](../examples/restart_and_projection.py) from the
repository root:

```bash
python -B -m examples.restart_and_projection
```

Add `--output-dir /tmp/common-solver-example` to retain its JSON snapshot and
the field/coefficient PDF figures. `overscreening_breakdown.plot_fields(solution)`
returns a figure with `u`, `w`, and physical flux; `plot_coefficients(solution)`
keeps the two Galerkin coefficient fields in separate panels. They also accept
collocation solution objects, preserve native arithmetic through field
evaluation, and convert to float64 only for Matplotlib rendering. Neither
helper shows or saves the figure automatically.
