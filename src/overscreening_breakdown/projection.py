"""Explicit one-way physical H1 projection from collocation to mixed fields.

A projection supplies an admissible Galerkin restart, not a converged solution
of its discrete equations. It retains the source's physical problem and owned
arithmetic. Raising precision cannot restore digits discarded from saved data;
load a new source snapshot before selecting a different arithmetic context.
"""

import math

import numpy as np

from overscreening_breakdown.collocation.solution import CollocationSolution
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.quadrature import QuadratureFactory
from overscreening_breakdown.galerkin.solution import GalerkinSolution, _distance_polynomial
from overscreening_breakdown.galerkin.finite_basis import FiniteGalerkinBasis
from overscreening_breakdown.galerkin.finite_solution import FiniteGalerkinSolution


def _finite(arithmetic, value, name):
    value = arithmetic.scalar(value)
    if not arithmetic.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


def _sum(arithmetic, values):
    values = tuple(_finite(arithmetic, value, "projection term") for value in values)
    return _finite(arithmetic, arithmetic.context.fsum(values) if arithmetic.use_mpmath
                   else math.fsum(values), "projection sum")


def _polynomial_derivatives(arithmetic, coefficients, coordinate):
    """Chebyshev recurrence and its first three analytic derivatives."""
    zero, one = arithmetic.scalar(0), arithmetic.scalar(1)
    rows = [(one, zero, zero, zero)]
    if len(coefficients) > 1:
        rows.append((coordinate, one, zero, zero))
    for _ in range(2, len(coefficients)):
        previous, before = rows[-1], rows[-2]
        rows.append(tuple(
            2*coordinate*previous[order]-before[order]
            +(2*order*previous[order-1] if order else zero)
            for order in range(4)))
    return tuple(_sum(arithmetic, (coefficient*row[order]
                                  for coefficient, row in zip(coefficients, rows)))
                 for order in range(4))


class _CorrectedProfile:
    """Source with its two endpoint corrections factored algebraically.

    Subtracting a tiny constant numerically would still lose the decaying tail.
    Instead use u=ub*(1-y)/2+(1-y**2)*r(y). For each k>=2,
    (T_k(y)-linear_endpoint_interpolant(T_k))/(1-y**2) is a polynomial.
    Its coefficient at T_0 is -k (even k); other same-parity T_j coefficients
    are -2*(k-j), for j<=k-2. The k=0,1 terms are wholly endpoint data.
    """

    def __init__(self, source, entries):
        self.arithmetic = source.arithmetic
        self.mapping = source.mapping
        self.wall = source.physical.boundary_conditions[0][0]
        self.length = source.L
        degree = len(entries)-1
        self.coefficients = tuple(
            _sum(self.arithmetic, (-(k-j)*(1 if j == 0 else 2)*entries[k]
                                   for k in range(j+2, degree+1, 2)))
            for j in range(max(0, degree-1)))
        self.tail = _distance_polynomial(self.arithmetic, self.coefficients)

    def fields(self, x):
        arithmetic = self.arithmetic
        x = self.mapping.physical_coordinate(x)
        zero = arithmetic.scalar(0)
        if x == arithmetic.inf:
            return zero, zero, zero, zero
        if x >= self.length:
            ratio = self.length/x
            distance = 2*(ratio/(1+ratio))
        else:
            distance = arithmetic.scalar(2)/(1+x/self.length)
        y = arithmetic.scalar(1)-distance
        if distance*len(self.coefficients)**2 < arithmetic.scalar("0.25"):
            coefficients, values = self.tail, []
            for order in range(4):
                value = zero
                for entry in reversed(coefficients):
                    value = value*distance+entry
                values.append(value if order % 2 == 0 else -value)
                coefficients = tuple((index+1)*entry for index, entry in
                                     enumerate(coefficients[1:]))
            r, r_y, r_yy, r_yyy = values
        else:
            r, r_y, r_yy, r_yyy = _polynomial_derivatives(
                arithmetic, self.coefficients, y)
        g = self.wall/2+(2-distance)*r
        g_y = r+(2-distance)*r_y
        g_yy = 2*r_y+(2-distance)*r_yy
        g_yyy = 3*r_yy+(2-distance)*r_yyy
        u = distance*g
        u_y = -g+distance*g_y
        u_yy = -2*g_y+distance*g_yy
        u_yyy = -3*g_yy+distance*g_yyy
        chain = (distance*(distance/2))/self.length
        chain_y, chain_yy = -distance/self.length, arithmetic.scalar(1)/self.length
        u_x = chain*u_y
        u_xx = chain*(chain_y*u_y+chain*u_yy)
        u_xxx = chain*((chain_y*chain_y+chain*chain_yy)*u_y
                       +3*chain*chain_y*u_yy+chain*chain*u_yyy)
        return tuple(_finite(arithmetic, value, "corrected source field")
                     for value in (u, -u_xx, u_x, -u_xxx))


def _project_finite(source, *, N, quadrature_order, subdivisions, trace_tolerance):
    """Independent H1 projections with only essential endpoint corrections."""
    arithmetic, physical = source.arithmetic, source.physical
    left, right = physical.boundary_conditions
    if (physical.delta <= 0 or not physical.epsilon.is_identity
            or left[0] is None or left[3] is None
            or any(left[k] is not None for k in (1, 2, 4))
            or right[0] is None or right[1] is None
            or any(right[k] is not None for k in (2, 3, 4))):
        raise ValueError("finite projection requires delta>0, epsilon=1, left u,u''' and right u,u'")
    start, end = physical.domain
    length = end-start
    defects = (source.u(start)-left[0], source.u(end)-right[0])
    if trace_tolerance is None:
        eps = arithmetic.context.eps if arithmetic.use_mpmath else arithmetic.scalar(np.finfo(float).eps)
        scale = max(arithmetic.scalar(1), abs(left[0]), abs(right[0]),
                    _sum(arithmetic, (abs(value) for value in source.coefficients)))
        trace_tolerance = 100*eps*scale
    trace_tolerance = _finite(arithmetic, trace_tolerance, "trace tolerance")
    if trace_tolerance < 0 or any(abs(value) > trace_tolerance for value in defects):
        raise ValueError("source endpoint defects exceed the trace tolerance")
    basis = FiniteGalerkinBasis(len(source.coefficients)+1 if N is None else N,
                                arithmetic, domain=physical.domain)
    if quadrature_order is None:
        quadrature_order = max(basis.N+1, (len(source.coefficients)+basis.N+4)//2)
    plan = QuadratureFactory(arithmetic, domain=physical.domain).plan(
        quadrature_order, subdivisions=subdivisions)
    if plan.order < basis.N+1:
        raise ValueError("projection requires quadrature order >= N+1 per panel")
    table = basis.tabulate(plan)
    lift = basis.lift(left[0], right[0])
    values = []
    for x, y in zip(plan.physical_nodes, plan.nodes):
        t = (x-start)/length
        correction = defects[0]*(1-t)+defects[1]*t
        correction_dx = (defects[1]-defects[0])/length
        values.append((source.u(x)-correction-lift.value(y),
                       source.u_dx(x)-correction_dx-lift.dx(y),
                       source.w(x), source.w_dx(x)))

    def project(rows, derivatives, value_index, derivative_index):
        size = len(rows[0])
        gram = arithmetic.matrix(tuple(tuple(_sum(arithmetic, (
            weight*(row[i]*row[j]+derivative[i]*derivative[j])
            for weight, row, derivative in zip(plan.physical_weights, rows, derivatives)))
            for j in range(size)) for i in range(size)))
        load = arithmetic.vector(tuple(_sum(arithmetic, (
            weight*(value[value_index]*row[i]+value[derivative_index]*derivative[i])
            for weight, value, row, derivative in zip(plan.physical_weights, values, rows, derivatives)))
            for i in range(size)))
        return tuple(_finite(arithmetic, value, "projected coefficient")
                     for value in arithmetic.solve(gram, load))

    coefficients = project(table.v, table.v_dx, 0, 1)+project(table.q, table.q_dx, 2, 3)
    info = (("source_solver", source.solver_type), ("source_N", source.N),
            ("source_domain", source.domain), ("norm", "physical_h1"),
            ("left_adjustment", -defects[0]), ("right_adjustment", -defects[1]),
            ("trace_tolerance", trace_tolerance), ("quadrature_order", plan.order),
            ("subdivisions", plan.subdivisions))
    return FiniteGalerkinSolution(basis, physical, coefficients, plan=plan,
                                  method="projection", projection_info=info)


def project_collocation(source, *, N=None, L=None, arithmetic=None,
                        quadrature_order=None, subdivisions=1,
                        trace_tolerance=None):
    """Project collocation u and -u'' independently in physical H1(dx).

    Accept a ``CollocationSolution`` for epsilon identically one, delta>0,
    and Galerkin-compatible boundary data, on a half-line or finite interval.
    Finite intervals require left u,u''' and right u,u'; their geometry comes
    from the source domain, so an explicit L is rejected. No reverse conversion or
    automatic solver conversion is provided. The target retains the same
    bound physical problem and arithmetic owner; a foreign owner is rejected.

    Tiny source wall/far-field defects are corrected explicitly before any
    integral. The default threshold is 100*eps*max(1,sum(abs(coefficients)),
    abs(u_b)). Larger defects require an explicit ``trace_tolerance`` or are
    rejected. Corrections and provenance are recorded in ``projection_info``.

    If L is unchanged, default quadrature integrates the polynomial H1 loads
    and Grams exactly in exact arithmetic. Changing L produces rational loads;
    the conservative fixed default rule does not certify their integration
    error. ``quadrature_order`` and ``subdivisions`` select the projection rule.
    Every returned convergence flag is false: solve and check quadrature on
    the target discrete equations before treating this restart as a solution.
    """
    if not isinstance(source, CollocationSolution):
        raise TypeError("projection requires a CollocationSolution; no reverse conversion is supported")
    source.check_precision()
    if arithmetic is not None and arithmetic is not source.arithmetic:
        raise ValueError("projection must retain the source arithmetic owner; load a new snapshot first")
    arithmetic = source.arithmetic
    physical = source.physical
    if physical.domain != (arithmetic.scalar(0), arithmetic.inf):
        if L is not None:
            raise ValueError("L is a half-line mapping parameter; finite projection uses the source domain")
        return _project_finite(source, N=N, quadrature_order=quadrature_order,
                               subdivisions=subdivisions, trace_tolerance=trace_tolerance)
    if physical.delta <= 0 or not physical.epsilon.is_identity:
        raise ValueError("projection requires delta>0 and epsilon identically one")
    left, right = physical.boundary_conditions
    if (left[0] is None or left[3] is None
            or any(left[index] is not None for index in (1, 2, 4))
            or right[0] not in (None, 0) or any(value is not None for value in right[1:])):
        raise ValueError("projection requires Galerkin-compatible wall and far-field data")
    entries = tuple(arithmetic.scalar(value) for value in source.coefficients)
    wall = left[0]
    wall_defect = _sum(arithmetic, ((-1)**index*entry
                                   for index, entry in enumerate(entries)))-wall
    far_defect = _sum(arithmetic, entries)
    if trace_tolerance is None:
        eps = (arithmetic.scalar(arithmetic.context.eps) if arithmetic.use_mpmath
               else arithmetic.scalar(np.finfo(np.float64).eps))
        scale = max(arithmetic.scalar(1), abs(wall),
                    _sum(arithmetic, (abs(value) for value in entries)))
        trace_tolerance = 100*eps*scale
    trace_tolerance = _finite(arithmetic, trace_tolerance, "trace tolerance")
    if trace_tolerance < 0:
        raise ValueError("trace tolerance must be nonnegative")
    if abs(wall_defect) > trace_tolerance or abs(far_defect) > trace_tolerance:
        raise ValueError("source endpoint defects exceed the trace tolerance; correct the source traces explicitly")
    degree = len(entries)-1
    basis = GalerkinBasis(len(entries)+1 if N is None else N, arithmetic,
                          L=source.raw_L if L is None else L)
    same_mapping = basis.mapping.L == source.L
    if quadrature_order is None:
        quadrature_order = (max(basis.N+1, (degree+basis.N+4)//2) if same_mapping
                            else 2*max(basis.N, degree+2)+4)
    plan = QuadratureFactory(arithmetic, L=basis.mapping.raw_L).plan(
        quadrature_order, subdivisions=subdivisions)
    if plan.order < basis.N+1:
        raise ValueError("projection requires quadrature order >= N+1 per panel")
    table = basis.tabulate(plan)
    corrected = _CorrectedProfile(source, entries)
    source_values = tuple(corrected.fields(x) for x in plan.physical_nodes)
    lift = basis.lift(wall)
    homogeneous_values = tuple(
        (values[0]-lift.value(y), values[2]-lift.dx(y), values[1], values[3])
        for y, values in zip(plan.nodes, source_values))

    def project(rows, derivatives, value_index, derivative_index):
        size = len(rows[0])
        gram = arithmetic.matrix(tuple(tuple(_sum(arithmetic, (
            weight*(row[i]*row[j]+derivative[i]*derivative[j])
            for weight, row, derivative in zip(plan.physical_weights, rows, derivatives)))
            for j in range(size)) for i in range(size)))
        load = arithmetic.vector(tuple(_sum(arithmetic, (
            weight*(values[value_index]*row[i]+values[derivative_index]*derivative[i])
            for weight, values, row, derivative in
            zip(plan.physical_weights, homogeneous_values, rows, derivatives)))
            for i in range(size)))
        coefficients = arithmetic.solve(gram, load)
        return tuple(_finite(arithmetic, value, "projected coefficient") for value in coefficients)

    a = project(table.v, table.v_dx, 0, 1)
    b = project(table.q, table.q_dx, 2, 3)
    info = (
        ("source_solver", source.solver_type), ("source_N", source.N),
        ("source_L", source.L), ("source_bits", arithmetic.context.prec if arithmetic.use_mpmath else 53),
        ("norm", "physical_h1"), ("wall_adjustment", -wall_defect),
        ("far_adjustment", -far_defect), ("trace_tolerance", trace_tolerance),
        ("same_mapping", bool(same_mapping)), ("quadrature_order", plan.order),
        ("subdivisions", plan.subdivisions),
    )
    return GalerkinSolution(basis, physical, a+b, plan=plan,
                            method="projection", projection_info=info)
