"""The eight historical linear equations, normalized explicitly to flux form."""

from overscreening_breakdown.manufactured import legacy_problems
from overscreening_breakdown.manufactured.common import (
    ManufacturedCase, native_problem, profile_derivatives, validate_number,
)


# (profile, original coefficient of u'', original coefficient of u'''', wall derivatives)
_CATALOG = (
    ("gaussian", (1, 1), -10, (0, 3)),
    ("exponential", (-2500, 1), 1, (0, 3)),
    ("gaussian", (-1, 1), 0, (0,)),
    ("exponential_cosine", (-1, 1), 1, (0, 2)),
    ("rational", (1, 1), -1, (0, 1)),
    ("rational_rapid_cosine", (-1, 1), 1, (0, 1)),
    ("rational_square_cosine", (-1, 1), 1, (0, 3)),
    ("gaussian", (0, 1), 1, (0, 3)),
)


def linear_case(number):
    """Return an immutable case; case 8 retains its pure-fourth-order operator.

    For a2*u''+a4*u''''=g, the physical equation divides by a2 and
    uses delta²=-a4/a2. This is impossible when a2=0 (case 8).
    """
    number = validate_number(number, 8)
    profile, a2_fraction, a4, wall_orders = _CATALOG[number-1]
    legacy_factory = getattr(legacy_problems, "problem_{}".format(number))

    def derivatives(x, arithmetic):
        return profile_derivatives(profile, x, arithmetic)

    def source(x, u, arithmetic):
        values = derivatives(x, arithmetic)
        a2 = arithmetic.scalar(a2_fraction[0])/a2_fraction[1]
        return values[2]+(arithmetic.scalar(a4)/a2)*values[4]

    def source_u(x, u, arithmetic):
        return arithmetic.scalar(0)

    if number == 8:
        factory, supported = None, ("legacy_collocation",)
    else:
        delta_squared = (-a4*a2_fraction[1], a2_fraction[0])
        def factory():
            return native_problem(profile, delta_squared=delta_squared,
                                  wall_orders=wall_orders, source=source, source_u=source_u,
                                  far_zero=number in (1, 2, 6, 7))
        supported = (("collocation", "galerkin") if wall_orders == (0, 3)
                     else ("collocation",))
    return ManufacturedCase("linear:{}".format(number), legacy_factory().get_name(), supported,
                            derivatives, factory, legacy_factory,
                            ("Native exact rational constants and wall traces; historical literals remain in legacy_problem().",))
