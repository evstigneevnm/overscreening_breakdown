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
  test_collocation_adapter test_flux_collocation test_collocation_mapping \
  test_rational_mapping test_galerkin_quadrature test_galerkin_basis \
  test_galerkin_assembly test_galerkin_manufactured_light \
  test_galerkin_solution test_solution_format test_galerkin_projection \
  test_solver_dispatch test_galerkin_plotting \
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

The mixed Galerkin tests additionally cover physical weak-form assembly,
Jacobian directional checks, scalar source homotopy, quadrature refinement, and
small manufactured problems with constant, smooth-variable, and jump
coefficients. Tiny exactly represented fields exercise the selected mpmath
precisions without a spatial error floor.
Solution tests cover physical field norms, solver-tagged persistence, explicit
collocation-to-Galerkin projection, the common entry point, and Galerkin PDF/PNG
rendering. The original collocation plots remain in the regression suite.

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
The mixed Galerkin assembly described below uses solver-owned arithmetic and
requires `delta>0`.

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
the mixed system.

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
length as their plan.

`rational_mapping.RationalHalfLineMap` provides the shared half-line geometry.
Its reference coordinate has `y=-1` at the wall and `y=1` at infinity.
Collocation keeps its angular coordinate `t`, with `y=cos(t)`, `t=pi` at the
wall and `t=0` at infinity. The map's `to_angle(x)` and `from_angle(t)` evaluate
the half-angle formulas directly to retain small angles and large finite
physical coordinates. Collocation's mapping wrappers accept scalars and arrays
of any shape and preserve the input. They refresh their cached map when `L`,
the arithmetic backend, or the exact mpmath bit precision changes. The existing
collocation basis, derivative formulas, node ordering, and saved coefficient
interpretation are retained. Its arithmetic still follows global mpmath
precision, as described above.

### Mixed Galerkin assembly

`galerkin_assembly.GalerkinAssembly(problem, basis, plan)` binds an unbound
`FluxProblem` or `overscreening_breakdown` description to the basis's arithmetic.
It captures the physical parameters, boundary data, and coefficient description
once. An already bound problem is rejected because its parameters may have
been rounded at another precision. Basis and plan must share the same arithmetic
owner, mapping length, and exact precision. The plan must have at least `N+1`
points per panel and split every declared epsilon interface; extra splits are
allowed.

Write `epsilon_b=epsilon(0+)`, `e=epsilon/epsilon_b`, and
`f_q(x,u)=exp(-q*x)*f(x,u)`. With all pairings integrated in physical `dx`, the
assembled residual represents

\[
\begin{aligned}
  (u',\phi')-(w,\phi)&=0,\\
  \delta^2(ew',\psi')+((e-1)u',\psi')+(w,\psi)
    +\epsilon_b^{-1}(f_q(\cdot,u),\psi)
    -\delta^2q_b\psi(0)&=0.
\end{aligned}
\]

Only the second equation is normalized by `epsilon_b`. The physical source is
not divided by `delta**2`. The homotopy parameter `q` damps `f` and `f_u`
identically and leaves the coefficient, lift, and natural wall load unchanged;
`q=0` is the physical problem. No derivative of epsilon is needed.

This assembly supports `[0,infinity)`, positive `delta`, and the wall data
`u(0)=u_b`, `u'''(0)=q_b`, with no other wall conditions. An optional explicit
zero value at infinity is accepted. The lift enforces the first condition;
the second contributes the natural load `delta**2*q_b*psi(0)`. The independent
field `w` has no imposed wall value or derivative row. Nontrivial constant,
smooth-variable, and declared piecewise-smooth epsilon descriptions are accepted.

Constant differential blocks, lift contributions, and the wall load are cached.
`linear_matrix`, `lift_vector`, `boundary_vector`, and `affine_vector` return
native copies. `linear_matrix` excludes the physical source; use
`jacobian(coefficients, q)` for the complete derivative, including `f_u`.
`residual(coefficients, q)` and `jacobian` use the coefficient order documented
above and the Newton convention `DR*step=-R`.

For example, the same physical problem used by the collocation example can
supply a small mixed system:

```python
from arithmetic import MpmathArithmetic
from examples.flux_problem_collocation import make_problem
from galerkin import GalerkinBasis
from galerkin_assembly import GalerkinAssembly
from nonlinear_solver import solve_newton
from quadrature import QuadratureFactory

arithmetic = MpmathArithmetic(dps=50)
basis = GalerkinBasis(N=4, arithmetic=arithmetic, L="1")
plan = QuadratureFactory(arithmetic, L="1").plan(order=8)
assembly = GalerkinAssembly(make_problem(), basis, plan)
result = solve_newton(assembly.nonlinear_system(), atol="1e-35")
u, w = basis.evaluate_fields(result.state, y=0, u_b="1")
```

`nonlinear_system(initial_guess=None)` supplies callbacks to the shared Newton
driver. Its default mixed vector is zero, meaning `u=wall_lift` and `w=0`.
An explicit guess must contain all `2*N-1` mixed coefficients. This low-level
interface does not project the problem's scalar initial guess or construct
`w` by differentiating it. Nonlinear trial overflow is reported to the shared
driver for damping; invalid problem descriptions and unexpected callback errors
propagate.

`with_plan(new_plan)` and `refined(order_factor=2, subdivisions_factor=1)`
create a new assembly while retaining the same bound physical snapshot and
basis. The previous coefficient vector can initialize its Newton solve.
`check_refinement(coefficients, atol=..., rtol=..., q=0)` compares the residual
and analytic Jacobian on the current and refined plans at the same state and
homotopy parameter. Its two thresholds are `atol+rtol*norm(refined_value)`;
both comparisons must pass. Residuals use an **algebraic Euclidean norm** and
Jacobians an **algebraic Frobenius norm**. These diagnostics are separate from
physical field errors and nonlinear convergence, and are not rigorous
quadrature error bounds. The assembly does not run an outer refinement loop.

Run the complete [mixed example](examples/flux_problem_galerkin.py), including
explicit quadrature refinement, from the repository root:

```bash
python -B -m examples.flux_problem_galerkin
```

It reuses a physical problem with `epsilon(x)=1+1/(1+x)` for NumPy and
50-digit mpmath solves, reporting Newton convergence, quadrature convergence,
and sampled errors in both independent fields separately.

### Common solver entry point and physical fields

`solvers.solve(problem, N, solver=...)` accepts the same unbound physical
description for either solver when epsilon is explicitly constant one.
Nontrivial epsilon selects the mixed Galerkin method explicitly. For example:

```python
from examples.flux_problem_collocation import make_problem
from solvers import solve

problem = make_problem()
collocation = solve(
    problem, 8, solver="collocation", L="1", use_mpmath=True,
    precision=50, method="newton", tolerance="1e-35",
)
mixed = solve(
    problem, 4, solver="galerkin", L="1", use_mpmath=True,
    precision=50, tolerance="1e-35", quadrature_order=8,
)
u_at_one, w_at_one = mixed.u("1"), mixed.w("1")
```

Collocation retains `newton`, `root`, `root_no_jac`, and `nonlin_solve` with
NumPy, and Newton with mpmath. Galerkin uses the shared Newton driver with
damping and continuation. A finite interval or `delta=0` can still use the
supported collocation boundary families; the mixed Galerkin method supports
only the fourth-order half-line problem described above. Existing manufactured
classes need an explicit `FluxProblem` adapter before using the common entry
point; their plots and notebooks are not automatically converted to Galerkin.

Both solution types expose `u(x)`, `w(x)`, `u_dx(x)`, `w_dx(x)`, and `flux(x)`
in physical coordinates. Scalars return native arithmetic scalars; arrays
preserve shape and return float64 or object arrays. Galerkin stores independent
coefficients for both fields and evaluates
`flux=epsilon*(u_dx+delta**2*w_dx)`. Collocation retains its single coefficient
vector and derives `w=-u''`. Coefficient properties return copies. Saved or
evaluated solution objects own their arithmetic and reject a subsequent change
to that context's exact bit precision. The common collocation wrapper restores
global mpmath precision after calling the legacy implementation; direct legacy
solver instances retain their existing global-context limitation.

For Galerkin, `converged` requires both `nonlinear_converged` and
`quadrature_converged`. The latter compares the residual and Jacobian at the
same fixed state on a finer rule. `quadrature_atol` and `quadrature_rtol` default
to the requested Newton tolerance. `max_refinements=3` permits three additional
solves on successively finer plans; reaching that limit can leave Newton
converged while the overall result remains unconverged. `newton_result` and
`quadrature_history` expose the separate outcomes. The final Newton convergence
check is at `q=0`. These algebraic diagnostics remain distinct from physical
field errors.

`initial_guess` can be a compatible solution object, a new-format saved path,
raw coefficients in the selected basis, or a scalar callback
`initial_guess(x, arithmetic)`. A solution restart must match the solver,
resolution, mapping, and physical metadata. A Galerkin callback is projected
into the first field, with its independent second field initialized to zero.
A supplied problem callback is used in the same way. A solver change requires
the explicit projection described below.

### Physical norms

Mixed solutions provide `l2_norm(field="u", plan=...)`,
`h1_norm(field="u", plan=...)`, `l2_error(reference, field="u", plan=...)`,
and `h1_error(reference, reference_dx, field="u", plan=...)`. Select `"u"` or
`"w"`; `"flux"` is also available for L2 diagnostics. All integrate in physical
`dx`, and H1 includes both the value and its physical derivative. For example:

```python
error = mixed.l2_error(lambda x: 1/(1+x), field="u", plan=mixed.plan)
w_norm = mixed.h1_norm(field="w", plan=mixed.plan)
```

The scalar helpers `solutions.physical_l2_norm`, `physical_h1_norm`,
`physical_l2_error`, and `physical_h1_error` also accept field callbacks from
either solution type. The supplied positive quadrature plan must use compatible
arithmetic; mixed field diagnostics additionally check mapping length and
epsilon interface splits. Quadrature error remains part of these numerical
norms. The legacy `spectral1d.L2_error` retains its mapped-coordinate meaning
and existing return shape.
The caller must supply integrable fields: a finite quadrature value does not
establish integrability on the half-line. A collocation polynomial with a
nonzero value at infinity is outside physical L2/H1 until explicitly corrected.

### Saved solutions and explicit conversion

`solution.save(path)` or `solution_io.save_solution(solution, path)` writes a
data-only JSON snapshot. It records the solver type, basis/version, actual
unknown count, mapping/domain, exact working precision, physical metadata,
coefficient fields, quadrature geometry, final convergence flags, and any
projection provenance. Mpmath values use exact binary mantissas and exponents;
float64 uses hexadecimal values, preserving signed zero. Decimal mapping and
interface inputs are retained for a later precision change.

```python
from solvers import load_solution

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

The default restores the saved backend and exact bit precision, including a
noncanonical bit count. Explicit `use_mpmath` and `precision` overrides can
round data. A backend or working-bit change clears all convergence flags;
otherwise the saved flags remain provenance rather than a new numerical
certificate. Newton iteration objects and quadrature iteration histories are
not serialized. Loading never changes global mpmath precision.

The existing collocation `save_solution` still writes its compatible pickle
format, now tagged `solver_type="collocation"`. Its reader accepts older
untagged payloads and bare arrays and rejects tagged Galerkin data. Import
trusted legacy coefficients with explicit missing geometry using
`solvers.load_legacy_collocation(path, problem=problem, N=..., L=...)`. The
JSON loader does not read pickle files. An imported vector is not marked as a
converged solution.

For the permitted one-way conversion, use `solvers.project_collocation`:

```python
from solvers import project_collocation

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

Run the common-interface [example](examples/common_solver_example.py) from the
repository root:

```bash
python -B -m examples.common_solver_example
```

Add `--output-dir /tmp/common-solver-example` to retain its JSON snapshot and
the field/coefficient PDF figures. `solution_plotting.plot_fields(solution)`
returns a figure with `u`, `w`, and physical flux; `plot_coefficients(solution)`
keeps the two Galerkin coefficient fields in separate panels. They also accept
collocation solution objects, preserve native arithmetic through field
evaluation, and convert to float64 only for Matplotlib rendering. Neither
helper shows or saves the figure automatically.

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
