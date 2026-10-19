# Overview of overscreening_breakdown

#### This is a repository for the pseudo-spectral collocation numerical method for the overscreening breakdown problem. For the discription, see bellow.

## Requirements and Installation
* #### python 3.8 or newer with modules:
- numpy
- matplotlib
- scipy
- pickle
* #### clone repository locally into your local folder

## Testing:

The lightweight GitHub Actions suite runs on Python 3.12 and 3.13 with both
the pure Python and gmpy mpmath backends. It covers NumPy float64 with `newton`,
`root`, `root_no_jac`, and `nonlin_solve`, and mpmath Newton with 15, 30, 50,
100, and 120 decimal digits. Unsupported mpmath/SciPy combinations must raise
`ValueError`.

Run the same suite locally from the repository root:

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B -m unittest -v \
  test_solution_persistence test_scipy_solver test_l2_error \
  test_manufactured_linear_light test_manufactured_nonlinear_light \
  test_plotting_light
```

Set `MPMATH_NOGMPY=1` before starting Python to test the pure Python backend.
For the gmpy backend, install `gmpy2` and leave `MPMATH_NOGMPY` unset (an empty
value still disables gmpy). Dependency versions used by CI are recorded in
[the workflow](.github/workflows/ci.yml) until project requirements are finalized.

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

#### For manufactured glabal tests run the following from your local folder:
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

For more details we refer to our article: https://arxiv.org/abs/2401.13458

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
If you have any questions, please contact khlyupin@phystech.edu or irina.nesterova@phystech.edu

## Funding
Partially, Evstigneev Nikolay and Ryabkov Oleg appreciate the support of the work by the Russian Science Foundation (RSF), project number 23-21-00095.
