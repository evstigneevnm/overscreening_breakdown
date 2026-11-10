# Installation and development

Linux with Python 3.9–3.13 is the validated platform. The distribution name is
`overscreening-breakdown`; the import package is `overscreening_breakdown`.
The existing Apache-2.0 license is included in source and wheel distributions.

## Install the package

Activate the Python or Anaconda environment you want to use, then run from
the checkout root:

```bash
python -m pip install .
```

Runtime dependencies are NumPy, SciPy, mpmath, and Matplotlib. Their conditional
versions match the validated selections in `requirements.txt`. Optional extras
provide the accelerated mpmath backend, notebook support, tests, and tools:

```bash
python -m pip install ".[gmpy,notebook]"
```

The existing full requirements file remains available. It includes the
validated runtime, gmpy2, and notebook dependencies, and does not install the
project itself:

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

No package has been published as part of this reorganization. Install from the
checkout or a built local wheel rather than assuming a public index release.

## Editable development

Use a separate environment and install the validated stack and editable source:

```bash
python -m pip install -r requirements-dev.txt
```

Equivalently, install `-e ".[test,dev,gmpy]"`. The `test` extra includes the
inline plotting stack. Python 3.9 selects pytest 8.4 plus `pytest-subtests`;
newer supported interpreters select pytest 9 with built-in subtest support.
This preserves the existing `unittest.TestCase.subTest` precision and method
loops. The `dev` extra adds build, Ruff, and pre-commit tools.

```bash
MPLBACKEND=Agg python -m pytest
python -m ruff check src tests examples benchmarks
python -m pre_commit validate-config
python -m pre_commit install
```

Tests use importlib mode and default to `tests/`, with strict configuration and
marker validation. Expensive manual sweeps live under `experiments/` rather
than pytest discovery. `slow` tests, when present, are excluded by default;
run them explicitly with `python -m pytest -m slow`. Existing tests retain
their unittest implementation and can also run with
`python -m unittest discover -s tests -t .`.

For optional coverage reporting, run
`python -m pytest --cov=overscreening_breakdown --cov-report=term-missing`.
No coverage percentage threshold is imposed in this structural stage.

Set `MPMATH_NOGMPY=1` before starting Python for the pure Python mpmath backend.
For gmpy, install the `gmpy` extra and leave that variable unset; an empty
value still disables gmpy. Use `CI_ARTIFACT_DIR` to retain plot artifacts.

## Build and verify distributions

```bash
python -m build
python -m pip install "dist/overscreening_breakdown-0.1.0-py3-none-any.whl[test]"
```

The default build creates a source archive, then builds its wheel from that
archive. The archive contains tests, examples and notebooks, benchmarks, experiments, docs,
and requirements files; the wheel contains only the runtime package, legacy
compatibility modules, and distribution metadata/license.

CI installs this wheel before running the ten Linux Python/backend
configurations. It checks that imports come from the installed distribution
rather than `src/`, runs pytest with JUnit output, and retains dependency lists
and plots. One job records coverage. A separate job validates the limited
correctness lint rules and builds the distributions.

## Compatibility during the reorganization

Stage 1 moved and packaged the existing implementation and tests while
preserving behavior. Stage 2 migrates manufactured cases, user examples,
notebooks, and opt-in studies to the canonical package and common solver.
Stage 3 will change the public interface separately. See the
[numerical guide](numerical_methods.md) for catalog classifications and the
explicit historical equation paths.

The first stage retains old import paths such as `spectral1d`, `solvers`, and
`physical_problem` through installed compatibility modules. Legacy imports of
the BSK classes from `overscreening_breakdown` remain available. Current examples
use canonical imports; notebooks live in `examples/notebooks/`. Run bounded
examples with `python -m examples.collocation`, `python -m examples.galerkin`,
`python -m examples.variable_epsilon`, and
`python -m examples.restart_and_projection` after installation. Existing Python
example entry points delegate to these commands. The new
canonical implementation lives under `src/overscreening_breakdown/`; no
`PYTHONPATH=src` setting is needed for installed or editable use.

Benchmark modules are checkout/source-archive tools; see
`python -m benchmarks.run_manufactured --help` and
`python -m benchmarks.run_precision --help`. Heavy sweeps are opt-in and have a
separate manual workflow. Only bounded CLI and numerical checks belong to the
ten-job pull-request matrix. Saved-solution contracts and one-way conversion
are documented in [saved_solutions.md](saved_solutions.md).
