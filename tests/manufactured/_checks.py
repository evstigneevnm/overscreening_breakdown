"""Independent collocation diagnostics for common-interface verification tests."""

import mpmath as mp
import numpy as np

from overscreening_breakdown.collocation.spectral1d import collocation_discretization


def native_values(function, points, arithmetic):
    inputs = np.asarray(points, dtype=object)
    output = np.empty(inputs.shape, dtype=object if arithmetic.use_mpmath else float)
    for index in np.ndindex(inputs.shape):
        output[index] = function(arithmetic.scalar(inputs[index]), arithmetic)
    return output[()] if inputs.ndim == 0 else output


def collocation_diagnostics(solution, case):
    """Check the physical PDE independently; retain the legacy mapped L2 metric.

    The discretization is used only to evaluate derivatives and diagnostics,
    never to solve or form a Newton residual. Physical half-line L2 integrability
    cannot be inferred from a raw collocation polynomial's finite quadrature.
    """
    arithmetic, physical = solution.arithmetic, solution.physical
    previous_bits = mp.mp.prec
    try:
        evaluator = collocation_discretization(
            solution.N, domain=list(physical.domain), L=solution.L,
            use_mpmath=arithmetic.use_mpmath, prec=arithmetic.dps,
        )
        evaluator.set_boundary_conditions(physical.boundary_conditions)
        coefficients = (mp.matrix(tuple(solution.coefficients)) if arithmetic.use_mpmath
                        else solution.coefficients)
        y = evaluator.all_discrete_points_in_basis()
        x = evaluator.all_discrete_points_in_domain()
        values = solution.u(x)
        lhs = evaluator.solution_with_derivative_in_basis(coefficients, y, 2)
        if physical.delta:
            lhs = lhs-physical.delta**2*evaluator.solution_with_derivative_in_basis(
                coefficients, y, 4)
        rhs = np.asarray([physical.physical_rhs(point, value)
                          for point, value in zip(x, values)], dtype=object)
        defect = max(abs(value) for value in lhs-rhs)
        relative = defect/max(arithmetic.scalar(1), max(abs(value) for value in rhs))
        normalized = defect/max(arithmetic.scalar(1), physical.delta**2)
        mapped_error = evaluator.L2_error(
            coefficients, lambda points: native_values(case.u, points, arithmetic))
        return arithmetic.scalar(relative), arithmetic.scalar(normalized), mapped_error
    finally:
        mp.mp.prec = previous_bits


def boundary_errors(solution):
    arithmetic = solution.arithmetic
    evaluations = (solution.u, solution.u_dx,
                   lambda x: -solution.w(x), lambda x: -solution.w_dx(x))
    errors = []
    for coordinate, conditions in zip(solution.domain, solution.physical.boundary_conditions):
        for order, target in enumerate(conditions):
            if target is not None:
                errors.append(abs(evaluations[order](coordinate)-target))
    return errors or [arithmetic.scalar(0)]
