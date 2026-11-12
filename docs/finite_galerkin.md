# Finite-interval mixed Galerkin

Use the common `solve` interface for

\[
 [\epsilon(x)(u'-\delta^2u''')]'=f(x,u),\qquad 0<x<R,
\]

with `delta>0` and boundary data
\[
 u(0)=u_{\mathrm L},\quad u'''(0)=q_{\mathrm L},\quad
 u(R)=u_{\mathrm R},\quad u'(R)=s_{\mathrm R}.
\]

The finite domain comes from the physical problem. `L` is used only by the
half-line mapping; it does not set the interval endpoint. This implementation
initially supports domains starting at zero and the four conditions above.
Finite collocation retains its existing boundary families and requires
explicitly constant epsilon one.

```python
from overscreening_breakdown import BSKProblem, LayeredMaterial, solve

material = LayeredMaterial(((0, "1"), ("1", "2"), ("2.5", "0.8")))
problem = BSKProblem(
    delta="2", epsilon=material.coefficient, domain=(0, "4"),
    boundary_conditions=(("1", None, None, "0", None),
                         ("0", "0", None, None, None)),
)
solution = solve(problem, 12, solver="galerkin", use_mpmath=True,
                 precision=50, tolerance="1e-30", verbosity=1)
print(solution.converged, solution.u("1"), solution.w("1"))
```

Use `FluxProblem` with unscaled `f(x,u,arithmetic)` and `f_u` for a custom
source. No derivative of epsilon is required by assembly. Declare all epsilon
interfaces strictly inside the finite interval; quadrature splits at each.
Both NumPy and solver-owned mpmath arithmetic use the same formulation.

## Spaces and boundary loads

With `y=2*x/R-1`, the independent fields are
\[
 u=\ell+(1-y^2)\mathbb P_{N-2},\qquad w\in\mathbb P_N,
 \qquad \ell=u_{\mathrm L}+(u_{\mathrm R}-u_{\mathrm L})x/R.
\]

There are `N-1` homogeneous potential coefficients and `N+1` auxiliary-field
coefficients, giving `2*N` unknowns for `N>=2`. The tests are
\[
 \phi\in(1+y)\mathbb P_{N-1},\qquad
 \psi\in(1-y)\mathbb P_{N-1}.
\]

Writing `e=epsilon/epsilon(0+)`, the two residual blocks express
\[
 \int_0^R u'\phi'\,dx-\int_0^R w\phi\,dx=s_{\mathrm R}\phi(R),
\]
\[
 \int_0^R\left[(e-1)u'\psi'+\delta^2e w'\psi'
                  +w\psi+\frac{f(x,u)}{\epsilon(0+)}\psi\right]dx
 =\delta^2q_{\mathrm L}\psi(0).
\]

Endpoint values of u are essential conditions in the lift. The right slope
and left third derivative enter as weak boundary loads. Their pointwise
derivatives need not equal the prescribed values exactly at finite N.
No value is prescribed for w at R; it is generally nonzero.

The trial and test spaces differ, so this is a finite Petrov formulation.
The existing half-line stability argument is not a proof for these spaces.

## Convergence, diagnostics, and saved data

Newton, backtracking, adjoint recovery, adaptive reference continuation, and
verbosity/callback options are shared with the other discretizations. The
screened reference keeps the finite domain, spaces, delta, and all boundary
data, with constant epsilon equal to its wall trace. A quadrature change
rebuilds both physical and reference operators. Success requires verified
physical residual convergence at q=0 and a quadrature refinement check.

`u`, `w`, `u_dx`, `w_dx`, and `flux` evaluate independent mixed fields in
physical coordinates. Physical L2/H1 diagnostics accept a compatible finite
quadrature plan. Tightening Newton tolerance or arithmetic precision does not
replace increasing N. Increasing R changes the physical boundary-value
problem and should be studied separately.

Snapshots use a distinct finite Petrov basis identifier, affine mapping,
interval endpoints, two-field layout, precision, and quadrature metadata.
Loading requires the original physical description; executable callbacks are
not stored. Wrong-domain, wrong-basis, and wrong-solver restarts are rejected.
Existing half-line and finite-collocation formats retain their interpretation.

`project_collocation(source, N=...)` explicitly projects a compatible finite
collocation solution into both finite Galerkin fields in physical H1. The
projection is an initial state, with convergence flags false. Reverse
conversion is unsupported.

For known-answer checks, `manufactured.finite_segment_case()` supplies a
quartic reference exactly representable at N=4. Select `coefficient="smooth"`
or supply a `LayeredMaterial` for variable/transmission cases. The latter
preserve u, u', u'', and physical flux across every interface; they require
spatial refinement to resolve the derivative jumps.
