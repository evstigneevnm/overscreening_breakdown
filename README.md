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
  test_collocation_adapter test_flux_collocation \
  test_rational_mapping test_galerkin_quadrature test_galerkin_basis \
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

`solve_nonlinear_problem` accepts an unbound `FluxProblem` directly when epsilon
is explicitly constant 1. It binds the description using the solver's backend
and precision, then adapts the physical callbacks to collocation's existing
normalization and homotopy. Both the constructor and `set_problem()` perform
this preparation. Each solver receives its own view; the description is not
modified. Pass the original description rather than an already bound snapshot,
which could contain values rounded at another precision.

For a finite interval, supply two boundary conditions for `delta=0` or four for
`delta>0`, with derivative orders below the differential order. On the half-line,
supply one or two wall conditions respectively, with an optional zero value at
infinity. Derivative conditions at infinity and nonzero far-field values are
unsupported by the adapter. For `delta=0`, replace the default boundary tuple,
which includes a third derivative, with second-order boundary data.

The legacy collocation basis still uses global mpmath precision. Independent
bound callback contexts do not make the entire collocation solver independent
of that global setting. Legacy problem objects retain their existing path.
Galerkin assembly is pending; the future fourth-order mixed solver will require
`delta>0`.

### FluxProblem collocation example

Run the complete [example](examples/flux_problem_collocation.py) from the
repository root:

```bash
python -B -m examples.flux_problem_collocation
```

It constructs a nonlinear manufactured problem with the exact solution
`u(x) = 1/(1+x)`, `delta=2`, `epsilon=1`, `u(0)=1`, and `u'''(0)=-6`.
The scalar source callbacks use the supplied arithmetic and contain no
collocation normalization. The same immutable problem is solved with NumPy
`root` and 50-digit mpmath Newton; both runs report convergence and sampled
field errors. CI exercises these example solves as well.

The solver call uses the existing interface:

```python
from examples.flux_problem_collocation import make_problem
from spectral1d import solve_nonlinear_problem

problem = make_problem()
solver = solve_nonlinear_problem(
    N=8, L=1, problem=problem, use_method="newton",
    use_mpmath=True, prec=50, tolerance="1e-35",
)
coefficients = solver.solve_problem()
```

### Galerkin bases and quadrature

`galerkin.GalerkinBasis` supplies conforming half-line spaces beside the
collocation code. With `y=(x-L)/(x+L)`, it uses
`V_N=(1-y**2) P_(N-2)` and `Q_N=(1-y) P_(N-1)`, expressed in Chebyshev
polynomials. Here `N>=2` is the maximum polynomial degree. The mixed vector
contains `N-1` coefficients for `u-wall_lift`, followed by `N` independent
coefficients for `w`: **`2*N-1` unknowns**. This differs from collocation's
meaning of `N`; equal `N` does not mean equal unknown counts.

Basis and lift methods `v`, `q`, `v_dx`, `q_dx`, `value` and `dx` take a
reference coordinate `y` in `[-1,1]`; `_dx`/`dx` returns a physical derivative.
The wall lift is `u_b*L/(x+L)`. Its `at(x)` and `derivative_at(x)` methods take
physical coordinates and retain the decaying tail when a large finite `x`
would round to `y=1`. Both fields vanish at infinity; only the first field's
homogeneous basis vanishes at the wall. The mixed second field is evaluated
independently, rather than defined by differentiating the first field.

`quadrature.QuadratureFactory` creates positive composite Gauss–Legendre
plans, split at every supplied physical interface. The nodes are strictly
inside each mapped panel, so neither interfaces nor infinity are sampled.
`plan.weights` integrate in `dy`; `plan.physical_weights` include `dx/dy` and
integrate in the physical measure `dx`. Use the same plan for every term of
the future mixed system.

```python
from arithmetic import MpmathArithmetic
from galerkin import GalerkinBasis
from quadrature import QuadratureFactory

arithmetic = MpmathArithmetic(dps=50)
basis = GalerkinBasis(N=4, arithmetic=arithmetic, L="1")
factory = QuadratureFactory(arithmetic, L="1", interfaces=("1",))
plan = factory.plan(order=basis.N + 1)
table = basis.tabulate(plan)
lift = basis.lift("1")
integral = plan.integrate(lambda x: 1/(1+x)**2)  # physical integral: 1
check = plan.check_refinement(
    lambda x: 1/(1+x)**2, atol="1e-40", rtol="1e-40",
)
```

`order` is the number of Gauss points per mapped panel; `subdivisions` splits
each interface interval uniformly in `y`. At least `N+1` Gauss points per
panel integrate these spaces' physical mass and derivative Gram forms exactly
in exact arithmetic. Variable coefficients and nonlinear loads generally need
additional quadrature refinement. `check_refinement` compares a fixed
integrand on the coarse and refined plans against `atol+rtol*abs(refined)`;
the difference is a diagnostic, not a rigorous error bound or proof of
nonlinear convergence. Interfaces must be supplied explicitly.

Each factory caches a bounded number of plans and Gauss rules for its arithmetic
owner and exact bit precision. Changing that context's precision invalidates old
maps, bases, lifts and plans. The factory rebuilds from its original `L` and
interface inputs on its next request; rebuild the basis as well. Decimal strings
preserve raw geometry when raising precision; already rounded scalars cannot
recover discarded digits. Tables require the same arithmetic owner and mapping
length as their plan. The new map uses `y=-1` at the wall and `y=1` at infinity,
independently of the legacy collocation trigonometric coordinates.

This stage provides quadrature and conforming spaces. Mixed residual/Jacobian
assembly and a Galerkin solve remain the next stage.

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
