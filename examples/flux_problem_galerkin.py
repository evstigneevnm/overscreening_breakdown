"""Solve a variable-epsilon FluxProblem with the mixed Galerkin assembly.

From the repository root, run::

    python -B -m examples.flux_problem_galerkin

The exact independent fields are u=1/(1+x) and w=-2/(1+x)**3, with
epsilon=1+1/(1+x), delta=2, u(0)=1 and u'''(0)=-6.  Both fields belong to
the N=4 spaces.  The same physical description is reused for NumPy and
50-digit mpmath arithmetic, including every quadrature operation.

Newton convergence and quadrature consistency are checked separately.  A
failed quadrature comparison creates a new discrete system, initialized from
the last accepted coefficients.  This example uses the low-level assembly;
common solver dispatch and mixed solution persistence are separate stages.
"""

from numbers import Integral

from arithmetic import MpmathArithmetic, NumpyArithmetic
from coefficients import Coefficient
from galerkin import GalerkinBasis
from galerkin_assembly import GalerkinAssembly
from nonlinear_solver import solve_newton
from physical_problem import FluxProblem
from quadrature import QuadratureFactory


def make_problem():
    """Create an arithmetic-independent nonlinear physical problem."""
    def epsilon(x, arithmetic):
        one = arithmetic.scalar(1)
        return one+one/(one+x)

    def source(x, u, arithmetic):
        one = arithmetic.scalar(1)
        s = one/(one+x)
        return (2*s**3+3*s**4-96*s**5-120*s**6
                +(u-s)+arithmetic.scalar("0.125")*(u**3-s**3))

    def source_u(x, u, arithmetic):
        return arithmetic.scalar(1)+arithmetic.scalar("0.375")*u*u

    return FluxProblem(
        delta="2",
        epsilon=Coefficient(epsilon, lower_bound=1, upper_bound=2, wall_value=2),
        f=source,
        f_u=source_u,
        boundary_conditions=(("1", None, None, "-6", None), (None,)*5),
    )


def solve_example(problem, *, use_mpmath=False, precision=50, max_refinements=3):
    """Return assembly, Newton result, quadrature check and sampled field error.

    ``max_refinements`` counts additional quadrature plans, so zero still
    performs one solve and its refinement comparison.  A comparison always
    uses the same fixed coefficient vector on both plans; only a failed
    comparison triggers a new nonlinear solve on the finer plan.
    """
    if (isinstance(max_refinements, bool)
            or not isinstance(max_refinements, Integral) or max_refinements < 0):
        raise ValueError("max_refinements must be a nonnegative integer")
    arithmetic = MpmathArithmetic(precision) if use_mpmath else NumpyArithmetic()
    tolerance = arithmetic.scalar(
        "1e-{}".format(max(10, precision-12)) if use_mpmath else "1e-11")
    basis = GalerkinBasis(4, arithmetic, L=1)
    interfaces = problem.bind(arithmetic).epsilon.interfaces
    plan = QuadratureFactory(arithmetic, L=1, interfaces=interfaces).plan(8)
    assembly = GalerkinAssembly(problem, basis, plan)
    initial = None  # u is the wall lift, while the independent w field is zero.
    for refinement in range(int(max_refinements)+1):
        result = solve_newton(
            assembly.nonlinear_system(initial), atol=tolerance, rtol=0,
            use_globalization=True,
        )
        if not result.converged:
            raise RuntimeError("The Galerkin Newton solve did not converge: {}".format(result.reason))
        check = assembly.check_refinement(result.state, atol=tolerance, rtol=0)
        if check.converged:
            errors = []
            for coordinate in ("0", "0.125", "1", "4", "10"):
                x = arithmetic.scalar(coordinate)
                exact = arithmetic.scalar(1)/(1+x)
                u, w = basis.evaluate_fields(
                    result.state, basis.mapping.to_reference(x), u_b=assembly.lift.u_b)
                errors.extend((abs(u-exact), abs(w+2*exact**3)))
            return assembly, result, check, max(errors)
        if refinement == max_refinements:
            raise RuntimeError("Galerkin quadrature did not converge within the refinement limit")
        assembly = check.refined_assembly
        initial = result.state.copy()
    raise AssertionError("unreachable Galerkin refinement state")


def main():
    problem = make_problem()
    for use_mpmath in (False, True):
        _, result, check, error = solve_example(problem, use_mpmath=use_mpmath)
        backend = "mpmath, 50 digits" if use_mpmath else "NumPy, float64"
        print("{}: converged={}, q={}, quadrature converged={}, max sampled field error={}".format(
            backend, result.converged, result.homotopy, check.converged, error))


if __name__ == "__main__":
    main()
