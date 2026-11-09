# Overview of overscreening_breakdown

#### This is a repository for the pseudo-spectral collocation numerical method for the overscreening breakdown problem. For the discription, see bellow

## Requirements and Installation

> **Platform support:** Linux only. Windows and other operating systems have not
> been tested and are currently unsupported.

Use Python 3.9 through 3.13, the versions covered by CI. After cloning the repository,
activate your chosen Python environment and install the validated dependencies:

```bash
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) automatically selects the validated package
versions for your Python interpreter. It includes the numerical libraries, `gmpy2`
acceleration, and the notebook kernel/inline plotting stack. Open the notebooks
using your existing Jupyter interface, such as Anaconda's JupyterLab.

## Testing

The lightweight GitHub Actions suite runs on Ubuntu 24.04 with Python 3.9, 3.10,
3.11, 3.12, and 3.13, using both the pure Python and gmpy mpmath backends, for ten
jobs in total.
It covers NumPy float64 with `newton`,
`root`, `root_no_jac`, and `nonlin_solve`, and mpmath Newton with 15, 30, 50,
100, and 120 decimal digits. Unsupported mpmath/SciPy combinations must raise
`ValueError`.

Run the same suite locally from the repository root:

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B -m unittest -v \
  test_solution_persistence test_scipy_solver test_l2_error \
  test_newton_globalization \
  test_arithmetic test_coefficients test_flux_problem test_physical_rhs \
  test_manufactured_linear_light test_manufactured_nonlinear_light \
  test_plotting_light
```

Set `MPMATH_NOGMPY=1` before starting Python to test the pure Python backend.
For the gmpy backend, install `gmpy2` and leave `MPMATH_NOGMPY` unset (an empty
value still disables gmpy). CI installs the same
[requirements.txt](requirements.txt) used for local environments.

The collocation Newton method uses the shared driver in `nonlinear_solver.py`.
It retains damped steps and homotopy continuation, with convergence verified at
the physical parameter `q=0`. `solver.get_newton_result()` exposes the accepted
step history, continuation stages, stopping targets, and convergence/failure
reason. The existing coefficient return value and `is_converged()` remain
available. The legacy `tolerance` keeps its existing RMS residual scaling;
the shared driver's direct interface has separate `atol` and `rtol` controls.

The light manufactured tests reuse the existing analytic problems with small,
fixed resolutions. They check residuals, boundary conditions, analytic error,
refinement, and agreement between methods/backends. The plotting smoke tests
render PDFs and an IPython inline PNG. Spatial approximation error is checked
separately from arithmetic precision; increasing decimal digits alone does not
remove discretization error.

Plot files use temporary directories by default. Set `CI_ARTIFACT_DIR=ci-artifacts`
to keep them. GitHub Actions retains the plots, test log, and installed dependency
list for seven days. The original full manufactured sweeps and notebooks remain
manual tests and are excluded from CI through an explicit module list.

## Physical problem interface

`overscreening_breakdown` exposes its existing `delta` parameter and an immutable
`epsilon` description. `physical_rhs(x, u)` and `physical_rhs_u(x, u)` evaluate
the unscaled source and its analytic derivative, without homotopy damping.
The existing `right_hand_side` methods retain their collocation normalization
and homotopy factor. The Gaussian normalization used by the model is preserved.

Bind a snapshot of a problem to arithmetic owned by its eventual solver:

```python
from arithmetic import MpmathArithmetic
from coefficients import Coefficient
from overscreening_breakdown import overscreening_breakdown

epsilon = Coefficient(
    lambda x, a: a.scalar(1) if x < a.scalar("1") else a.scalar(2),
    lower_bound=1, upper_bound=2, wall_value=1, interfaces=("1",),
)
problem = overscreening_breakdown(epsilon=epsilon)
problem.set_parameters(delta="2", gamma="0.5", u0="1")
physical = problem.bind(MpmathArithmetic(dps=120))
value = physical.physical_rhs("0.5", "0.1")
wall_epsilon = physical.epsilon.wall_value
```

`NumpyArithmetic()` supplies float64 operations. Each `MpmathArithmetic` owns an
independent context and leaves global mpmath precision unchanged. Use decimal
strings or integers for high-precision parameters; an existing binary float
retains its original rounding. Binding copies parameter and boundary values;
later changes to the source problem do not change an existing snapshot.

`Coefficient(function, ...)` uses a scalar `function(x, arithmetic)` callback.
`Coefficient.from_function(epsilon_x, ...)` adapts an explicit one-argument
`epsilon(x)` function. Supply positive finite lower/upper bounds, the right
wall trace, and every interior interface. These are caller declarations;
evaluations check the bounds, while regularity is not inferred by sampling.
At zero, evaluation returns the declared right trace. `ConstantCoefficient`
describes a constant. Omitting `epsilon` in `set_parameters` preserves it.

For a different physical source, `physical_problem.FluxProblem` accepts `delta`,
`epsilon`, `f(x, u, arithmetic)`, `f_u(x, u, arithmetic)`, and optional domain,
boundary data and `initial_guess(x, arithmetic)`, then exposes the same bound
interface. Inputs and outputs are real scalars. Nonfinite source outputs raise
`OverflowError`, allowing a nonlinear solver to backtrack a trial state.
User callbacks must use compatible arithmetic and keep any captured state
unchanged. The legacy `initial_guess_solution(x)` callback follows that same
responsibility when adapted to a bound BSK problem.

This stage provides problem descriptions and evaluations; Galerkin assembly is
still pending. Existing collocation solvers accept only explicitly constant
`epsilon=1`. Their legacy mpmath path still uses global precision and mutates
legacy problem arithmetic; precision isolation applies to the new bound views.
The physical interface allows `delta=0`; a fourth-order mixed solver must
separately reject that second-order case.

#### For manufactured glabal tests run the following from your local folder

```
python3 test_manufactures_solutions.py
```

resutls will be stored in "./figures" folder.

Other tests are:

- test_basis.py
- test_matrices.py
- test_solve_nonlinear_problem.py
- test_different_mu.ipynb

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
Newton's method with spectral discretization over the collocation method is used in order to find the solution from some initial guess. The method includes homotopy for difficult cases that allow one to converge to the solution.
For the example-driven manual, see:

```
execution_example.ipynb
```

For the tests on the stability of parameters, see:

```
test_different_mu.ipynb
```

## Contacts

The numerical method is implemented by Evstigneev Nikolay and Ryabkov Oleg.
Physical formulation by: Alexey Khlyupin, Irina Nesterova, Kirill Gerke.
If you have any questions, please contact <khlyupin@phystech.edu> or <irina.nesterova@phystech.edu>

## Funding

Partially, Evstigneev Nikolay and Ryabkov Oleg appreciate the support of the work by the Russian Science Foundation (RSF), project number 23-21-00095.
