# Numerical methods and manufactured studies

## Physical problem and arithmetic

`BSKProblem` is an immutable physical description with explicit sigma, mu,
gamma, delta, epsilon, domain, boundary conditions, and optional scalar initial
guess. Bind it to solver-owned arithmetic to evaluate `physical_rhs(x, u)`
and `physical_rhs_u(x, u)`, the unscaled source and its analytic derivative
without homotopy damping. The model's Gaussian normalization is preserved.

Bind a snapshot of a problem to arithmetic owned by its eventual solver:

```python
from overscreening_breakdown import BSKProblem, Coefficient, MpmathArithmetic

epsilon = Coefficient(
    lambda x, a: a.scalar(1) if x < a.scalar("1") else a.scalar(2),
    lower_bound=1, upper_bound=2, wall_value=1, interfaces=("1",),
)
problem = BSKProblem(
    sigma="10", mu="0", gamma="0.5", delta="2", epsilon=epsilon,
    boundary_conditions=(("1", None, None, "0", None), (None,)*5),
)
physical = problem.bind(MpmathArithmetic(dps=120))
value = physical.physical_rhs("0.5", "0.1")
wall_epsilon = physical.epsilon.wall_value
```

`NumpyArithmetic()` supplies float64 operations. Each `MpmathArithmetic` owns an
independent context and leaves global mpmath precision unchanged. Use decimal
strings or integers for high-precision parameters; an existing binary float
retains its original rounding. Construct a new immutable description when
changing physical parameters. Binding copies parameter and boundary values
into a numerical snapshot without changing another owner's context.

`Coefficient(function, ...)` uses a scalar `function(x, arithmetic)` callback.
`Coefficient.from_function(epsilon_x, ...)` adapts an explicit one-argument
`epsilon(x)` function. Supply positive finite lower/upper bounds, the right
wall trace, and every interior interface. These are caller declarations;
evaluations check the bounds, while regularity is not inferred by sampling.
At zero, evaluation returns the declared right trace. `ConstantCoefficient`
describes a constant.

For a different physical source, `overscreening_breakdown.FluxProblem` accepts `delta`,
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

### Runnable examples

The bounded examples use the common `solve` entry point and catalog definitions:

```bash
python -B -m examples.collocation --backend numpy
python -B -m examples.galerkin --backend mpmath --precision 50
python -B -m examples.variable_epsilon --backend both
python -B -m examples.restart_and_projection
```

Pass `--output-dir /tmp/overscreening-example` to retain the applicable JSON
snapshots and figures. Inspect `--help` for options. The old three Python
example wrapper modules have been removed. Notebooks live
under `examples/notebooks/` and use the installed canonical package.

### Galerkin bases and quadrature

`overscreening_breakdown.galerkin.basis.GalerkinBasis` supplies conforming half-line spaces beside the
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

`overscreening_breakdown.quadrature.QuadratureFactory` creates positive composite Gauss–Legendre
plans, split at every supplied physical interface. The nodes are strictly
inside each mapped panel, so neither interfaces nor infinity are sampled.
`plan.weights` integrate in `dy`; `plan.physical_weights` include `dx/dy` and
integrate in the physical measure `dx`. Use the same plan for every term of
the mixed system.

```python
from overscreening_breakdown.arithmetic import MpmathArithmetic
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.quadrature import QuadratureFactory

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

`overscreening_breakdown.rational_mapping.RationalHalfLineMap` provides the shared half-line geometry.
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

`overscreening_breakdown.galerkin.assembly.GalerkinAssembly(problem, basis, plan)` binds an unbound
`FluxProblem` or `BSKProblem` description to the basis's arithmetic.
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
from overscreening_breakdown.arithmetic import MpmathArithmetic
from overscreening_breakdown.manufactured import rational_case
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.nonlinear_solver import solve_newton
from overscreening_breakdown.quadrature import QuadratureFactory

arithmetic = MpmathArithmetic(dps=50)
basis = GalerkinBasis(N=4, arithmetic=arithmetic, L="1")
plan = QuadratureFactory(arithmetic, L="1").plan(order=8)
assembly = GalerkinAssembly(rational_case(nonlinear=True).make_problem(), basis, plan)
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

Run the [variable-epsilon example](../examples/variable_epsilon.py), including
common-entry quadrature refinement, from the repository root:

```bash
python -B -m examples.variable_epsilon
```

It reuses a physical problem with `epsilon(x)=1+1/(1+x)` for NumPy and
50-digit mpmath solves, reporting Newton convergence, quadrature convergence,
and sampled errors in both independent fields separately.

### Common solver entry point and physical fields

`overscreening_breakdown.solve(problem, N, solver=...)` accepts the same unbound physical
description for either solver when epsilon is explicitly constant one.
Nontrivial epsilon selects the mixed Galerkin method explicitly. For example:

```python
from overscreening_breakdown import solve
from overscreening_breakdown.manufactured import rational_case

problem = rational_case(nonlinear=True).make_problem()
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
only the fourth-order half-line problem described above. The manufactured catalog supplies native descriptions for representable
legacy equations; the solver support table below identifies the permitted paths.

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

For NumPy collocation methods `root`, `root_no_jac`, and `nonlin_solve`, the
returned coefficients are checked against a freshly assembled residual at
`q=0`. If `M=unknown_count`, convergence requires finite coefficients and all
finite residual entries, including every tau boundary row, with

\[
\frac{\|R(c,0)\|_2}{\sqrt M}
\le \texttt{tolerance}\sqrt M.
\]

The right side preserves the historical collocation Newton residual target.
The HYBR tolerance supplied to SciPy is a step criterion; its success flag and
cached residual do not replace this independent check. A stopped SciPy solve
can still satisfy the residual target, while a reported SciPy success with a
large or nonfinite residual remains unconverged.

`scipy_result` records `method`, the certified `converged` flag and `reason`,
`scipy_success`, `status`, `message`, `nfev`, `njev`, `residual_norm` (RMS),
`residual_max`, `target_norm`, and `homotopy=0`. Evaluation counts are SciPy's
reported counts and exclude the additional residual verification. The record
is absent (`None`) for Newton and Galerkin solves. A new solve clears previous
SciPy diagnostics before calling user callbacks. These checks certify the
assembled algebraic equations at the requested tolerance; spatial error and
manufactured field, PDE and boundary accuracy assertions remain independent.

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

The scalar helpers `overscreening_breakdown.norms.physical_l2_norm`, `physical_h1_norm`,
`physical_l2_error`, and `physical_h1_error` also accept field callbacks from
either solution type. The supplied positive quadrature plan must use compatible
arithmetic; mixed field diagnostics additionally check mapping length and
epsilon interface splits. Quadrature error remains part of these numerical
norms. The legacy `spectral1d.L2_error` retains its mapped-coordinate meaning
and existing return shape.
The caller must supply integrable fields: a finite quadrature value does not
establish integrability on the half-line. A collocation polynomial with a
nonzero value at infinity is outside physical L2/H1 until explicitly corrected.

### Live progress reporting

The common `solve` entry point accepts `verbosity=0`,
`progress_callback=None`, and `progress_interval=1.0`. The same reporting
controls apply to collocation, mixed Galerkin, NumPy, mpmath, initialization,
Newton, adjoint recovery, homotopy and quadrature refinement. The old
`verbose` argument is removed without an alias.

Level 0 emits no solver progress text. Level 1 reports the solve description,
major phases and stages, refinement decisions, and final outcome. Level 2
adds accepted Newton/adjoint steps and periodic SciPy evaluation progress.
Level 3 adds trial rejection, backtracking, recovery reasons, Jacobian
construction, and linear-solve phases. Notebook helpers and example command
lines select level 2; ordinary library calls and example `run()` functions
default to level 0.

Starts are emitted before expensive work, including quadrature construction,
matrix assembly, initialization, Jacobian construction and linear solving.
Completions carry elapsed times. A blocking factorization can report its start
and completion; intermediate progress requires support from that operation.
Frequent evaluation/trial text is limited by `progress_interval` seconds,
while phase changes and final outcomes are unthrottled. Output is flushed
during execution so notebooks can display it before the solve returns.

Events identify the solver and method and retain their numerical scalars in
the active arithmetic. Collocation residuals are labelled RMS; mixed residuals
are Euclidean. Both display their actual acceptance target and homotopy
parameter where relevant. A successful continuation stage is distinct from
final convergence at `q=0`. Galerkin reports nonlinear and quadrature
convergence separately, including refinement differences and targets. SciPy's
stop status and independently verified residual outcome remain separate.

HYBR and `fsolve` do not provide accepted-iteration callbacks. Their progress
therefore reports evaluated residuals and function/Jacobian call counts;
these events do not claim accepted Newton steps. Reporting reuses computed
values and does not introduce additional residual, Jacobian or quadrature
evaluations, change tolerances, or alter numerical acceptance decisions.
Live evaluation counters count actual wrapped calls; the returned
`scipy_result.nfev` and `njev` are SciPy-reported counts and can differ because
SciPy versions account for preflight calls differently.
Text uses a compact number of significant digits; mpmath values are formatted
through their arithmetic owner without an intermediate `float` conversion.

An observer passed as `progress_callback(event)` receives immutable
`ProgressEvent` values even at `verbosity=0`. Fields are `kind`, `level`,
`solver`, `method`, `elapsed`, and a read-only `data` mapping. The mapping
contains native numerical scalars and immutable metadata, never mutable
coefficient arrays, matrices, problem objects or arithmetic contexts. An
observer can collect events for a residual plot or write its own log; the
solver does not retain every trial event by default. Existing numerical
histories retain their established meanings.

The shared Newton driver's accepted-state callback keeps its separate
contract. Progress observers do not control iteration acceptance. Observer
exceptions propagate as `ProgressCallbackError` with the original exception
as their cause, so an observer overflow cannot be mistaken for numerical
overflow in the solver. Quiet reporting does not suppress user-callback
output, ordinary warnings or exceptions, and does not replace process-wide
`sys.stdout` or configure global logging handlers.

Benchmark command lines accept `--verbosity 0` through `--verbosity 3` and
default to 0. Enabled solver text streams during each run and is also retained
in its `solver_log` result field.

### Newton recovery and initialization

The default nonlinear iteration remains damped Newton. Optional controls act
at different points of the solve:

```python
# SciPy initialization before a NumPy collocation solve.
initialized = solve(problem, 8, solver="collocation", initializer="fsolve",
                    initializer_tolerance="0.1", initializer_max_evaluations=200)

# Native recovery during Newton, usable with either discretization/arithmetic.
recovered = solve(problem, 4, solver="galerkin", use_mpmath=True,
                  precision=50, tolerance="1e-35", newton_fallback="adjoint",
                  adjoint_max_steps=20, use_globalization=True)
```

`initializer="fsolve"` is NumPy collocation only. It uses the analytic
Jacobian and a hard combined residual/Jacobian evaluation budget. Its
`initializer_tolerance` is a SciPy step criterion, not the requested final
residual tolerance. The initializer retains the best finite improving
candidate it evaluates; if none improves the starting state, that state is
retained. SciPy's success/status and candidate acceptance are reported
separately in `initialization_result`. Callback programming errors propagate.
The selected final solver still determines convergence, including the
verified physical equation at homotopy `q=0`. The former
`use_adjoint_opimization` flag was this fsolve initialization and has been
removed; it was not an adjoint algorithm.

`newton_fallback="adjoint"` requires `method="newton"`. After a singular
Newton solve or an exhausted Newton line search, it attempts residual-merit
descent at the current fixed homotopy parameter. If `R` is the assembled
residual and `J=DR`, the merit and its gradient are

\[
\Phi(z)=\tfrac12 R(z)^T W R(z),\qquad
\nabla\Phi(z)=J(z)^T W R(z).
\]

`W` is the identity for the mixed algebraic Euclidean norm, and the identity
divided by the collocation residual dimension for its RMS norm. The gradient
is formed using the active NumPy or owned mpmath arithmetic. Galerkin uses
the negative gradient. Collocation projects that direction onto the nullspace
of its normalized linear boundary rows, preserving those boundary values
during recovery. Galerkin's homogeneous first-field basis already preserves
the essential wall and far-field traces; its second-field natural wall load
remains in the residual.

The admissible direction is scaled by a positive factor when its largest
component exceeds one. Backtracking halves the step and requires a finite
trial, strict residual-norm decrease, and Armijo sufficient decrease with
coefficient `1e-4`. Nonfinite trial states can be rejected and retried;
unrelated callback errors propagate. A zero admissible gradient or exhausted
recovery budget can still leave the solve unconverged. This residual-merit
adjoint is a discrete Jacobian-transpose operation, not a continuous adjoint
PDE or an assertion of physical-error reduction.

`adjoint_max_steps` bounds accepted recovery steps separately for each fixed
homotopy stage. `use_globalization` continues to control homotopy independently;
an accepted recovery step does not change `q`. No fallback step alone establishes
convergence: the ordinary residual acceptance test, final `q=0` check, and
Galerkin quadrature check remain necessary.

`newton_result.iterations` counts attempted Newton solves. Accepted recovery
steps and their evaluation costs are reported by `adjoint_steps` and
`adjoint_evaluations`, with `recovery_reasons` on the result and stages.
Iteration `history` records carry `kind="newton"` or `"adjoint"`, with merit,
gradient norm, recovery reason, and evaluation count where applicable. A
reported gradient norm is the full Euclidean gradient norm before collocation
boundary projection. Galerkin `quadrature_history` entries retain the Newton
result for each attempted plan, so earlier recovery remains inspectable.
These objects and initializer diagnostics are runtime records; the saved
solution JSON schema remains unchanged.

The original VODE experiment has moved to
[`experiments/legacy_adjoint_flow.py`](../experiments/legacy_adjoint_flow.py).
It computes its historical matrix inverse only when explicitly called and
retains the old preconditioned flow for research. It is separate from the
native residual-merit fallback above. Ordinary notebook comparisons now use
the public initializer/fallback controls, while unsupported boundary sweeps
have a standalone research notebook under `experiments/notebooks/`.

## Manufactured catalog

Use reusable case definitions from the installed package:

```python
from overscreening_breakdown import solve
from overscreening_breakdown.manufactured import linear_case, rational_case

case = linear_case(2)
solution = solve(case.make_problem(), 24, solver="collocation", L="2")
x = solution.arithmetic.scalar("0.5")
error = abs(solution.u(x)-case.u(x, solution.arithmetic))
```

Every case exposes `identifier`, `name`, `supported_solvers`, `make_problem()`,
and reference methods `u`, `u_dx`, `w`, `w_dx`, and `flux`, each taking
`(x, arithmetic)`. `derivative(x, arithmetic, order)` supplies derivatives
through order four. Factories retain no arithmetic owner. Some descriptions
compute irrational delta values and wall traces only in `bind(arithmetic)`;
they implement the same binding protocol as `FluxProblem` and are accepted by
`solve` and the JSON loader. No solver-specific scaling belongs in their `f`.

| Factory | Common collocation | Mixed Galerkin | Restriction |
| --- | --- | --- | --- |
| `linear_case(1)`, `(2)`, `(7)` | Yes | Yes | Fourth order, supported wall value/third derivative |
| `linear_case(3)` | Yes | No | Second order, `delta=0` |
| `linear_case(4)`, `(5)`, `(6)` | Yes | No | Wall first/second derivative conditions outside the mixed formulation |
| `linear_case(8)` | Legacy API only | No | Pure fourth derivative cannot be represented by finite-delta flux form |
| `nonlinear_case(1)` | Yes | No | Second order, `delta=0` |
| `nonlinear_case(2)`, `(3)`, `(4)`, `(6)`, `(7)` | Yes | Yes | Fourth order with supported boundary data |
| `nonlinear_case(5)` | Yes | No | Wall first derivative condition |
| `rational_case(coefficient="constant")` | Yes | Yes | Explicit epsilon one |
| `rational_case(coefficient="scaled"/"smooth"/"exponential")` | No | Yes | Nontrivial epsilon |
| `transmission_case()` | No | Yes | Declared epsilon jump at `x=1` |
| `polynomial_segment_case()` | Yes | No | Finite interval, second order |

`rational_case(..., nonlinear=True)` adds a nonlinear manufactured source with
an analytic derivative. The default exact fields are `u=1/(1+x)` and
`w=-2/(1+x)**3`, with `L=1`, `delta=2`. At `N=4` both fields lie in the mixed
spaces, so this fixture can expose arithmetic errors below float64 accuracy.
The smooth exponential coefficient requires adequate quadrature even when the
fields are exactly representable. The transmission fixture enforces exact
physical flux continuity in the manufactured reference; a global finite-N
approximation should be assessed with weak residuals and flux L2 error rather
than exact pointwise transmission at the jump.

The native catalog retains the physical equations at `q=0` and evaluates
constants using the supplied arithmetic. This corrects historical rounded
decimal/trigonometric boundary literals and binary-float fractions. The common
homotopy damps the entire physical source. In historical nonlinear cases 2 and
3, the old implementation damped only the reaction contribution; the final
`q=0` equation agrees, while the continuation path follows the existing common
solver convention. A native source is not a wrapper around a mutable legacy
problem object.

`case.legacy_problem()` explicitly returns a fresh historical class where
available, including its original constants and normalization. Linear case 8
requires this path. Definitions in `manufactured.legacy_additional` involving
epsilon multiplying selected derivatives are also retained as legacy equations:
they are not the divergence-form variable-epsilon problem. The old flat import
aliases are removed. Canonical low-level modules remain available for these
explicitly selected equations, and numerical coefficient files retain their
dedicated legacy reader. Old project-object pickle names are unsupported.

## Opt-in benchmark studies

Run these modules from the installed repository or a source archive. They are
not shipped in the runtime wheel and do not run heavy sweeps in pull-request
CI. The default runs one small rational case. Each invocation varies one study
axis; case selectors may be repeated explicitly.

```bash
# Spatial resolution, with quadrature convergence checked separately.
python -B -m benchmarks.run_manufactured --case transmission --solver galerkin \
  --study resolution --N 8 16 32 --diagnostic-order 128 \
  --output-dir /tmp/transmission-resolution

# Fixed N and precision; prevent adaptive changes to the requested rule.
python -B -m benchmarks.run_manufactured --case rational:exponential \
  --solver galerkin --study quadrature --N 8 --quadrature-order 9 18 36 \
  --max-refinements 0 --output-dir /tmp/quadrature-study

# Fixed case, N, and quadrature; native mpmath arithmetic at increasing digits.
python -B -m benchmarks.run_precision --case rational:constant --solver galerkin \
  --N 4 --quadrature-order 8 --precisions 30 50 100 120 \
  --output-dir /tmp/precision-study

# Preserve a general operator that the flux formulation cannot express.
python -B -m benchmarks.run_manufactured --case linear:8 \
  --solver legacy_collocation --N 24 --output-dir /tmp/legacy-fourth-order
```

Selectors are `linear:1` through `linear:8`, `nonlinear:1` through
`nonlinear:7`, `rational:<coefficient>`, `rational-nonlinear:<coefficient>`, and
`transmission`. Solver selection is explicit. Inspect `--help` for mapping,
Newton/globalization, precision, quadrature, and independent diagnostic options.
Legacy linear problems use a direct solve; legacy nonlinear problems use the
selected existing nonlinear method. `--method` applies to nonlinear solves.

`run_precision` holds both N and the quadrature rule fixed and disables adaptive
quadrature solves. Its default tolerance is `auto`, using
`10**(-max(8, precision-12))`; set an explicit tolerance when studying a different
stopping criterion. A tight tolerance does not remove spatial or quadrature
error. A quadrature study can deliberately return an unconverged overall flag:
the fixed coarse rule's independent refinement check may fail.

`--backend python` or `--backend gmpy` verifies the actual mpmath backend loaded
by that process. To compare backends, start separate processes with
`MPMATH_NOGMPY=1` or with that variable unset. A runner cannot switch the mpmath
backend after import. `--arithmetic numpy` uses float64 regardless of a requested
mpmath digit setting; its actual precision is recorded as 53 bits.

Every attempted solve produces a JSON/CSV row, including nonconvergence and
callback exceptions. A failed solve or diagnostic returns a nonzero process
status. The output records versions, platform, revision and working-tree state,
actual backend/bits, requested controls, actual unknown count, native coefficient
decimals, Newton histories and final homotopy, quadrature histories, convergence
flags, and separate solve/diagnostic wall times. High-precision values are
decimal strings without a float64 intermediate; timing values remain measured
wall-clock quantities. The solve timing includes assembly and linear solves;
it is not a claimed breakdown of those internal costs. No warm-cache claim or
performance threshold is inferred from it.
SciPy collocation rows also retain a `scipy` record containing the stop status,
message, evaluation counts, verified final residual and acceptance target.
That record distinguishes SciPy termination from residual-certified convergence
in both the common and legacy nonlinear benchmark paths; other methods record
`scipy=null`.

Physical errors use an independent fixed Gauss rule whose order is reported;
increase `--diagnostic-order` to assess integration error. Galerkin L2/H1
estimates cover the half-line and assume integrable reference fields. Raw
collocation can retain a tiny nonzero value at infinity, so its errors are
explicitly labeled `*_finite_interval` and integrated over `[0,8]` by default;
`--diagnostic-endpoint` changes the endpoint. Sampled errors and wall/far traces
are also recorded. A finite quadrature result is not proof of integrability.
Compare the solvers using epsilon one, matched accuracy, and actual unknown
counts, rather than equal N alone.

The manual [benchmark workflow](../.github/workflows/benchmarks.yml) uses
`workflow_dispatch`, starts with one small case, and uploads numerical records.
Its failure status retains the output for inspection. The experiment command
`python -m experiments.manufactured_solutions` delegates to the bounded runner.
The historical heavy plotting sweep remains explicitly available with
`python -m experiments.manufactured_solutions --legacy-full-sweep`, writing to
`./figures`; its original legacy runtime modules remain unchanged.
