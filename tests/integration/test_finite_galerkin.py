"""Public finite dispatch, actual continuation recovery, and explicit restarts."""

from dataclasses import replace

import pytest

from overscreening_breakdown import (
    BSKProblem, FiniteGalerkinSolution, LayeredMaterial, project_collocation, solve,
)
from overscreening_breakdown.manufactured import finite_segment_case, rational_case


@pytest.mark.parametrize("use_mpmath,digits", [(False, 15)]+[(True, n) for n in (15, 30, 50, 100, 120)])
def test_actual_reference_recovery_with_difficult_initial_field(use_mpmath, digits):
    case = finite_segment_case(nonlinear=True, beta="2")
    options = dict(N=4, solver="galerkin", use_mpmath=use_mpmath, precision=digits,
                   tolerance="1e-{}".format(max(9, digits-8)) if use_mpmath else "1e-10",
                   quadrature_order=12, max_iterations=8,
                   initial_guess=lambda x, a: a.scalar(30))
    direct = solve(case.make_problem(), use_globalization=False, **options)
    recovered = solve(case.make_problem(), use_globalization=True, **options)
    assert not direct.converged
    assert recovered.converged
    result = recovered.newton_result
    assert result.reference_result.converged
    assert result.reference_result.residual_norm <= result.reference_result.target_norm
    assert any(stage.homotopy > 0 for stage in result.stages)
    assert result.homotopy == result.stages[-1].homotopy == 0
    assert result.stages[-1].converged
    a = recovered.arithmetic
    assert abs(recovered.u("0.7")-case.u(a.scalar("0.7"), a)) < 1000*a.scalar(options["tolerance"])


def test_underintegrated_rule_refines_separately_from_newton():
    case = finite_segment_case("smooth", nonlinear=True)
    events = []
    result = solve(case.make_problem(), 4, solver="galerkin", tolerance="1e-10",
                   quadrature_order=5, progress_callback=events.append)
    assert result.converged
    assert len(result.quadrature_history) > 1
    assert result.quadrature_history[0].nonlinear_converged
    assert not result.quadrature_history[0].quadrature_converged
    assert result.quadrature_history[-1].quadrature_converged
    assert result.plan.order > 5
    assert result.newton_result.homotopy == 0
    assert events


@pytest.mark.parametrize("use_mpmath", [False, True])
def test_explicit_finite_projection_restarts_but_is_not_itself_a_solve(use_mpmath):
    case = finite_segment_case(nonlinear=True, nonzero_boundary=True)
    problem = case.make_problem()
    options = dict(use_mpmath=use_mpmath, precision=50,
                   tolerance="1e-35" if use_mpmath else "1e-10")
    source = solve(problem, 5, solver="collocation", **options)
    assert source.converged
    with pytest.raises(ValueError, match="projection"):
        solve(problem, 4, solver="galerkin", initial_guess=source, **options)
    projected = project_collocation(source, N=4)
    assert isinstance(projected, FiniteGalerkinSolution)
    assert not any((projected.converged, projected.nonlinear_converged, projected.quadrature_converged))
    assert projected.unknown_count == 8
    assert projected.arithmetic is source.arithmetic
    a = source.arithmetic
    bound = a.scalar("1e-30" if use_mpmath else "1e-8")
    assert abs(projected.u("0.6")-source.u("0.6")) < bound
    assert abs(projected.w("0.6")-source.w("0.6")) < bound
    resumed = solve(problem, 4, solver="galerkin", initial_guess=projected, **options)
    assert resumed.converged
    with pytest.raises(ValueError, match="half-line"):
        project_collocation(source, N=4, L=2)
    with pytest.raises(TypeError, match="CollocationSolution"):
        project_collocation(resumed)


def test_restart_rejects_wrong_domain_even_when_vector_lengths_match():
    finite = finite_segment_case().make_problem()
    source = solve(finite, 4, solver="galerkin")
    assert isinstance(source, FiniteGalerkinSolution)
    with pytest.raises(ValueError, match="geometry"):
        solve(finite_segment_case(R=3).make_problem(), 4, solver="galerkin", initial_guess=source)
    halfline = rational_case().make_problem()
    with pytest.raises(ValueError, match="geometry"):
        solve(halfline, 4, solver="galerkin", initial_guess=source)
    resumed = solve(finite, 4, solver="galerkin", initial_guess=source, L=999)
    assert resumed.converged


def test_bsk_and_multiple_interfaces_use_existing_physical_problem():
    material = LayeredMaterial(((0, 1), ("0.5", 2), ("1.25", "0.8"), ("1.7", "1.3")))
    problem = BSKProblem(delta="2", epsilon=material.coefficient, domain=(0, 2),
                         boundary_conditions=((1, None, None, 0, None), (0, 0, None, None, None)))
    result = solve(problem, 8, solver="galerkin", tolerance="1e-10")
    assert result.converged
    assert result.domain == (0, 2)
    assert result.physical.epsilon.interfaces == (0.5, 1.25, 1.7)
    assert result.u(0) == 1
    assert result.u(2) == 0
    assert abs(result.w(2)) > 1e-5
    for invalid in (replace(problem, delta=0),
                    replace(problem, boundary_conditions=((1, None, None, 0, None),
                                                           (0, None, 0, None, None)))):
        with pytest.raises(ValueError):
            solve(invalid, 8, solver="galerkin")


@pytest.mark.parametrize("count", [1, 2, 3, 5, 7, 10])
def test_each_requested_interface_count_is_split_on_the_finite_domain(count):
    # Exact decimal interface inputs exercise each layer count without a
    # spatial-convergence sweep. The linear solve has a fixed small budget.
    layers = ((0, 1),)+tuple((str(index), 1+index % 2) for index in range(1, count+1))
    material = LayeredMaterial(layers)
    case = finite_segment_case(material, R=str(count+1))
    result = solve(case.make_problem(), 4, solver="galerkin", quadrature_order=12)
    assert result.converged
    assert len(result.physical.epsilon.interfaces) == count
    boundaries = {endpoint for panel in result.plan.panels for endpoint in panel}
    assert all(result.mapping.to_reference(point) in boundaries
               for point in result.physical.epsilon.interfaces)
