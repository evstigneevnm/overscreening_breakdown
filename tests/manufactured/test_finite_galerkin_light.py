"""Finite physical accuracy, native precision, and transmission refinement."""

import mpmath as mp
import numpy as np
import pytest

from overscreening_breakdown import LayeredMaterial, MpmathArithmetic, solve
from overscreening_breakdown.manufactured import finite_segment_case


CONFIGURATIONS = [(False, 15)]+[(True, digits) for digits in (15, 30, 50, 100, 120)]


@pytest.mark.manufactured
@pytest.mark.parametrize("use_mpmath,digits", CONFIGURATIONS)
@pytest.mark.parametrize("coefficient", ["constant", "smooth"])
@pytest.mark.parametrize("nonzero_boundary", [False, True])
def test_exact_quartic_fields_retain_requested_precision(use_mpmath, digits, coefficient, nonzero_boundary):
    bits = mp.mp.prec
    case = finite_segment_case(coefficient, R="2.125", nonlinear=True,
                               nonzero_boundary=nonzero_boundary)
    tolerance = "1e-{}".format(max(9, digits-8)) if use_mpmath else "1e-11"
    solution = solve(case.make_problem(), 4, solver="galerkin", use_mpmath=use_mpmath,
                     precision=digits, tolerance=tolerance, quadrature_order=12)
    a = solution.arithmetic
    bound = 1000*a.scalar(tolerance)
    assert solution.converged
    assert solution.unknown_count == 8
    assert solution.newton_result.homotopy == 0
    assert solution.quadrature_converged
    assert mp.mp.prec == bits
    for numerator in (0, 1, 3, 7, 8):
        x = a.scalar("2.125")*numerator/8
        for field in ("u", "w", "u_dx", "w_dx", "flux"):
            actual = getattr(solution, field)(x)
            assert isinstance(actual, type(a.scalar(0)))
            assert abs(actual-getattr(case, field)(x, a)) < bound
    assert abs(solution.w(a.scalar("2.125"))) > a.scalar("0.5")
    assert abs(solution.u(0)-1) < bound
    target = solution.physical.boundary_conditions[1][0]
    assert abs(solution.u(a.scalar("2.125"))-target) < bound
    assert solution.l2_error(lambda x: case.u(x, a), plan=solution.plan.refined()) < bound


@pytest.mark.manufactured
@pytest.mark.parametrize("coefficient", ["constant", "smooth"])
def test_linear_problem_and_collocation_use_same_physical_definition(coefficient):
    case = finite_segment_case(coefficient, nonzero_boundary=True)
    problem = case.make_problem()
    mixed = solve(problem, 4, solver="galerkin", tolerance="1e-11")
    assert mixed.converged
    if coefficient == "constant":
        for method in ("newton", "root", "root_no_jac", "nonlin_solve"):
            collocation = solve(problem, 5, method=method, tolerance="1e-10")
            assert collocation.converged
            for x in (0, "0.375", "1.5", 2):
                assert abs(collocation.u(x)-mixed.u(x)) < 1e-8
                assert abs(collocation.w(x)-mixed.w(x)) < 1e-8
    else:
        with pytest.raises(ValueError, match="epsilon"):
            solve(problem, 5, solver="collocation")


@pytest.mark.manufactured
@pytest.mark.parametrize("use_mpmath", [False, True])
def test_genuine_jump_fields_and_flux_improve_under_spatial_refinement(use_mpmath):
    material = LayeredMaterial(((0, 1), ("0.7", 2)))
    case = finite_segment_case(material, nonlinear=True)
    errors = []
    for resolution in (6, 12):
        solution = solve(case.make_problem(), resolution, solver="galerkin",
                         use_mpmath=use_mpmath, precision=35,
                         tolerance="1e-22" if use_mpmath else "1e-10",
                         quadrature_order=2*resolution+4)
        assert solution.converged
        a = solution.arithmetic
        plan = solution.plan.refined()
        errors.append([solution.l2_error(lambda x, f=field: getattr(case, f)(x, a),
                                         field, plan=plan) for field in ("u", "w", "flux")])
    assert all(fine < coarse for coarse, fine in zip(*errors))


@pytest.mark.parametrize("graded", [False, True])
def test_manufactured_source_and_transmission_are_independently_consistent(graded):
    a = MpmathArithmetic(65)
    material = LayeredMaterial(((0, 1), ("0.7", 2), ("1.4", "1.5")),
                               kind="graded" if graded else "constant")
    case = finite_segment_case(material, nonlinear=True, nonzero_boundary=True)
    physical = case.make_problem().bind(a)
    for text in ("0.125", "0.9", "1.75"):
        x = a.scalar(text)
        for order in range(1, 5):
            expected = a.context.diff(lambda t: case.u(t, a), x, order)
            assert abs(case.derivative(x, a, order)-expected) < a.scalar("1e-58")
        flux_dx = a.context.diff(lambda t: case.flux(t, a), x)
        assert abs(physical.physical_rhs(x, case.u(x, a))-flux_dx) < a.scalar("1e-58")
        trial = a.scalar("0.37")
        jacobian = a.context.diff(lambda u: physical.physical_rhs(x, u), trial)
        assert abs(physical.physical_rhs_u(x, trial)-jacobian) < a.scalar("1e-58")
    h = a.scalar("1e-25")
    for point in physical.epsilon.interfaces:
        for order in (0, 1, 2):
            assert abs(case.derivative(point-h, a, order)-case.derivative(point+h, a, order)) < 1000*h
        assert abs(case.flux(point-h, a)-case.flux(point+h, a)) < 1000*h
        assert abs(case.derivative(point-h, a, 3)-case.derivative(point+h, a, 3)) > a.scalar("1e-3")
    assert all(np.isfinite(float(value)) for row in physical.boundary_conditions for value in row
               if value is not None)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_zero_cubic_reaction_does_not_evaluate_overflowing_unused_powers(nonlinear):
    from overscreening_breakdown import NumpyArithmetic
    physical = finite_segment_case(nonlinear=nonlinear, beta="0").make_problem().bind(NumpyArithmetic())
    with np.errstate(over="raise", invalid="raise"):
        assert np.isfinite(physical.physical_rhs(np.float64("0.3"), np.float64("1e200")))
        assert physical.physical_rhs_u(np.float64("0.3"), np.float64("1e200")) == 1
