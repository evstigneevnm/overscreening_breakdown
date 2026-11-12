# Overscreening breakdown

Collocation and mixed Galerkin solvers for overscreening and generalized
half-line flux equations,

\[
\left(\epsilon(x)\bigl(u'(x)-\delta^2u'''(x)\bigr)\right)'=f(x,u(x)).
\]

Linux with Python 3.9–3.13 is the validated platform. Windows and other
operating systems remain unsupported because they have not been tested.

## Install and run

Activate your Python or Anaconda environment, then install from this checkout:

```bash
python -m pip install -r requirements.txt
python -m pip install .
python -B -m examples.collocation --backend numpy
python -B -m examples.variable_epsilon --backend mpmath --precision 50
```

`requirements.txt` selects the validated numerical, gmpy, and notebook versions
for the interpreter. Minimal installs and optional extras are described in the
[installation and build guide](INSTALL.md). No public package publication is
assumed. Open the notebooks under [examples/notebooks](examples/notebooks)
using your existing Jupyter interface.

The public interface now uses canonical package imports and an immutable
`BSKProblem`. Old flat imports and lowercase BSK entry points have been removed.
See the [breaking migration guide](docs/public_api.md) for replacements.

A manufactured physical problem can be shared by the two solvers when its
coefficient and boundary data permit both:

```python
from overscreening_breakdown import solve
from overscreening_breakdown.manufactured import rational_case

case = rational_case(coefficient="constant", nonlinear=True)
problem = case.make_problem()
collocation = solve(problem, 8, solver="collocation")
galerkin = solve(problem, 4, solver="galerkin", quadrature_order=8)
print(galerkin.u(1), galerkin.w(1), galerkin.flux(1))
print(galerkin.nonlinear_converged, galerkin.quadrature_converged)
```

Galerkin returns independent fields `u` and `w`; collocation derives `w=-u''`.
`N` has different meanings for the discretizations, so compare actual unknown
counts and attained accuracy. Galerkin's overall convergence requires both
Newton and quadrature checks.

NumPy collocation can refine its starting guess with `initializer="fsolve"`.
Both solvers can enable native adjoint recovery with `newton_fallback="adjoint"`,
including mpmath runs. Initialization, adjoint steps, and homotopy have separate
controls and diagnostics; see the [public API guide](docs/public_api.md).

With `use_globalization=True`, a failed physical Newton solve can recover from
a computed screened linear reference. Both methods follow the same complete
residual blend from `q=1` to the original problem at `q=0`, retaining accepted
states and reducing the continuation step after a failed stage. Physical source
callbacks remain independent of `q`. Configure `ContinuationOptions` and
`ScreenedLinearReference` through `continuation=...`; `globalization_init` has
been removed. See [reference continuation](docs/numerical_methods.md#reference-continuation)
for defaults, separate stage and total budgets, and reference diagnostics.

Set `verbosity=2` on either solver for live phase and iteration diagnostics,
or `0` for quiet library calls. Notebook helpers and example command lines
default to level 2; scripts accept `--verbosity 0` through `--verbosity 3`.
The old `verbose` keyword has been removed. Structured `progress_callback`
events retain native numerical precision independently of console output.

- [Numerical methods, problem definitions, and manufactured catalog](docs/numerical_methods.md)
- [Saved solutions, explicit projection, and restart compatibility](docs/saved_solutions.md)
- [Installation, development, testing, and distribution builds](docs/installation.md)
- [Public API and breaking import migration](docs/public_api.md)

## Layout and migration stages

```text
src/overscreening_breakdown/   Installed numerical package
    problems/                Physical flux, BSK, and segment definitions
    collocation/             Discretization, adapter, and evaluable solution
    galerkin/                Conforming basis, mixed assembly, solution
    manufactured/            Reusable native cases and explicit legacy definitions
tests/                       Unit, integration, manufactured, plotting, legacy
examples/                    Bounded runnable examples and notebooks/
benchmarks/                  Opt-in studies with JSON/CSV results
experiments/                 Explicit legacy plots and manual explorations
docs/                        Numerical, persistence, and installation guides
```

The three migration stages are complete: package/test organization,
manufactured cases/examples/benchmarks, and the public interface cleanup.
Use `BSKProblem`, `FluxProblem`, `solve`, and the typed solution APIs from
`overscreening_breakdown`. Canonical low-level legacy definitions and numerical
saved-data readers remain available for the experiments that need them;
the old import aliases and Python example wrapper modules are removed.
The catalog documents native-arithmetic corrections and the common homotopy;
unsupported equations are not silently reinterpreted as flux problems.

## Tests and opt-in studies

```bash
python -m pip install -r requirements-dev.txt
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B -m pytest -q
```

CI builds and installs a wheel, then tests Python 3.9–3.13 with both pure Python
and gmpy mpmath backends (ten Linux jobs). It covers NumPy collocation methods,
mpmath Newton at 15, 30, 50, 100, and 120 digits, Galerkin assembly/refinement,
saved solutions, explicit projection, bounded examples, and PDF/inline PNG
rendering. The tests distinguish spatial accuracy from arithmetic precision.
Set `MPMATH_NOGMPY=1` before starting Python for the pure backend; leave it
unset for gmpy. Plots, JUnit reports, logs, and installed dependencies are kept
as CI artifacts. One job records coverage.

Heavy studies stay outside pull-request CI. Start with a single tiny benchmark:

```bash
python -B -m benchmarks.run_manufactured --case rational:constant \
  --solver galerkin --N 4 --quadrature-order 8 --output-dir /tmp/benchmark-one
```

Resolution, quadrature, and arithmetic studies have separate commands in the
[numerical guide](docs/numerical_methods.md). A manual `workflow_dispatch`
benchmark workflow retains JSON/CSV results. No wall-clock thresholds are part
of the numerical assertions. `experiments.manufactured_solutions` delegates to
the bounded runner; its historical expensive plotting sweep requires
`--legacy-full-sweep` explicitly.

## Description


#### Problem formulation

Overscreening breakdown problem describes the electric double layer (EDL) near electrodes with heterogeneous surface structure. We investigate the interplay between short-range ion correlations, which provides charge density oscillations and overscreening regime, with the impact of electrode surface morphology.

We solve the Modified Poisson-Fermi equation in dimensionless form:
$$(1-\delta_c^2 \nabla^2) \nabla^2 u = \frac{\sinh(u)}{1 +2\gamma \sinh^2\left(u/2\right)} - \frac{0.5 \mu g(x)e^{u}}{1 +2\gamma \sinh^2\left(u/2\right)}$$
with boundary condition: $$u(x = 0) = V, \quad u'''(x = 0) = 0$$ where $u$ is electrostatic potential, $\delta_c$ is correlation length describing electrostatic correlations, $\gamma$ is the compacity parameter, $\mu$ is the magnitude of ion separation caused by electrode surface geometry and ion size asymmetry, $g(x) = \frac{1}{\sqrt{2\pi}} \exp\{-\frac {x^2}{2\sigma^2}\}$ has the form of the Gauss distribution function describing the area of electrode surface structure impact with roughness deviation $\sigma$, and $V$ is the potential on the electrode.

The main parameters, treated as the input of physical problem, are:

- $\delta_c$ to control the scale of electrostatic correlations
- $\sigma$ to control the scale, where surface roughness acts on the EDL
- $\mu$ to control the magnitude of electrode surface roughness contribution
- $V$ to control the initial structure of EDL
- $\gamma$ to control the EDL density

As the output you obtain the solution on electrostatic potential $u$ and then using classical electrostatic definition you can also calculate: charge density of EDL, ion concentrations, cumulative charge or differential capacitance.

For more details we refer to our article: <https://arxiv.org/abs/2401.13458>

```bibtex
@article{Nesterova_2025, title={Mechanism of overscreening breakdown by molecular-scale electrode surface morphology in asymmetric ionic liquids}, volume={677}, ISSN={0021-9797}, url={http://dx.doi.org/10.1016/j.jcis.2024.08.040}, DOI={10.1016/j.jcis.2024.08.040}, journal={Journal of Colloid and Interface Science}, publisher={Elsevier BV}, author={Nesterova, Irina and Evstigneev, Nikolay M. and Ryabkov, Oleg I. and Gerke, Kirill M. and Khlyupin, Aleksey}, year={2025}, month=Jan, pages={396–405} }
```

#### Numerical method

The problem is solved on the $R^+$ domain by using an analytical mapping to the domain [0, pi].
Newton's method with spectral collocation computes a solution from an initial
guess. Reference continuation provides recovery for difficult cases; successful
completion requires a verified residual for the original physical problem.
For the example-driven manual, see:

```
examples/notebooks/execution_example.ipynb
```

For the tests on the stability of parameters, see:

```
examples/notebooks/test_different_mu.ipynb
```

## Contacts

The numerical method is implemented by Evstigneev Nikolay and Ryabkov Oleg.
Physical formulation by: Alexey Khlyupin, Irina Nesterova, Kirill Gerke.
If you have any questions, please contact <khlyupin@phystech.edu> or <irina.nesterova@phystech.edu>

## Funding

Partially, Evstigneev Nikolay and Ryabkov Oleg appreciate the support of the work by the Russian Science Foundation (RSF), project number 23-21-00095.
