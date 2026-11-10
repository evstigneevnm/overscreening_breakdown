"""Historical nonlinear physical equations and reusable rational fixtures."""

from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.manufactured import legacy_problems
from overscreening_breakdown.manufactured.common import (
    ManufacturedCase, native_problem, profile_derivatives, validate_number,
)
from overscreening_breakdown.problems.flux import FluxProblem


_PROFILES = ("poisson_boltzmann", "exponential", "gaussian", "rational",
             "rational_rapid_cosine", "gaussian_cosine", "exponential_reciprocal_cosine")


def _reaction(x, u, arithmetic, number):
    if number == 1:
        return arithmetic.sinh(u), arithmetic.cosh(u)
    half = arithmetic.scalar(1)/2
    hyperbolic = arithmetic.sinh(u*half)
    denominator = 1+hyperbolic**2
    numerator = arithmetic.sinh(u)
    numerator_u = arithmetic.cosh(u)
    if number == 2:
        gaussian = arithmetic.exp(-x*x*half)/arithmetic.sqrt(2*arithmetic.pi)
        gaussian_exponential = half*gaussian*arithmetic.exp(u)
        numerator -= gaussian_exponential
        numerator_u -= gaussian_exponential
    denominator_u = hyperbolic*arithmetic.cosh(u*half)
    return (numerator/denominator,
            (numerator_u*denominator-numerator*denominator_u)/denominator**2)


def nonlinear_case(number):
    """Return the q=0 physical equation, leaving legacy continuation untouched.

    Most old equations have -u''+u'''' on the left and therefore require a
    minus sign on their physical source. Case 2 instead divides by -1/2500.
    """
    number = validate_number(number, 7)
    profile = _PROFILES[number-1]
    legacy_factory = getattr(legacy_problems, "nonlinear_problem_{}".format(number))
    delta_squared = (0, 1) if number == 1 else (2500, 1) if number == 2 else (1, 1)
    wall_orders = (0,) if number == 1 else (0, 1) if number == 5 else (0, 3)
    normalization = 1 if number == 1 else -2500 if number == 2 else -1

    def derivatives(x, arithmetic):
        return profile_derivatives(profile, x, arithmetic)

    def source(x, u, arithmetic):
        reaction, _ = _reaction(x, u, arithmetic, number)
        if number == 1:
            return reaction
        values = derivatives(x, arithmetic)
        exact_reaction = _reaction(x, values[0], arithmetic, number)[0]
        return (values[2]-arithmetic.scalar(delta_squared[0])/delta_squared[1]*values[4]
                +normalization*(reaction-exact_reaction))

    def source_u(x, u, arithmetic):
        return normalization*_reaction(x, u, arithmetic, number)[1]

    def factory():
        initial = "gaussian" if number in (3, 6) else "exponential"
        callback = ((lambda x, a: a.exp(-x)*a.cos(a.scalar(5))) if number == 5 else
                    (lambda x, a: a.cos(a.scalar(1))/(1+x)) if number == 7 else None)
        return native_problem(profile, delta_squared=delta_squared, wall_orders=wall_orders,
                              source=source, source_u=source_u, initial_profile=initial,
                              initial_callback=callback,
                              far_zero=number == 5)

    supported = ("collocation",) if number in (1, 5) else ("collocation", "galerkin")
    return ManufacturedCase("nonlinear:{}".format(number), legacy_factory().get_name(), supported,
                            derivatives, factory, legacy_factory,
                            ("Physical q=0 equation; legacy_problem() retains historical continuation and binary literals.",))


def rational_case(coefficient="constant", nonlinear=False, *, beta="0.125", initial=None):
    """u=1/(1+x), delta=2, positive reaction, and four epsilon variants."""
    if coefficient not in ("constant", "scaled", "smooth", "exponential"):
        raise ValueError("coefficient must be constant, scaled, smooth, or exponential")
    if not isinstance(nonlinear, bool):
        raise TypeError("nonlinear must be a bool")
    if initial not in (None, "exp", "square"):
        raise ValueError("initial must be None, exp, or square")
    constant = 3 if coefficient == "scaled" else 1

    def epsilon_pair(x, arithmetic):
        reciprocal = arithmetic.scalar(1)/(1+x)
        if coefficient == "smooth":
            return 1+reciprocal, -reciprocal**2
        if coefficient == "exponential":
            exponential = arithmetic.exp(-x)
            return 1+exponential, -exponential
        return arithmetic.scalar(constant), arithmetic.scalar(0)

    def derivatives(x, arithmetic):
        return profile_derivatives("rational", x, arithmetic)

    def source(x, u, arithmetic):
        values = derivatives(x, arithmetic)
        epsilon, epsilon_dx = epsilon_pair(x, arithmetic)
        result = epsilon_dx*(values[1]-4*values[3])+epsilon*(values[2]-4*values[4])+(u-values[0])
        if nonlinear:
            result += arithmetic.scalar(beta)*(u**3-values[0]**3)
        return result

    def source_u(x, u, arithmetic):
        return arithmetic.scalar(1)+(3*arithmetic.scalar(beta)*u*u if nonlinear else 0)

    def factory():
        epsilon = (Coefficient(lambda x, a: epsilon_pair(x, a)[0], lower_bound=1,
                               upper_bound=2, wall_value=2)
                   if coefficient in ("smooth", "exponential") else constant)
        guess = (None if initial is None else
                 (lambda x, a: a.exp(-x)) if initial == "exp" else
                 (lambda x, a: (a.scalar(1)/(1+x))**2))
        return FluxProblem(delta="2", epsilon=epsilon, f=source, f_u=source_u,
                           boundary_conditions=(("1", None, None, "-6", None), (None,)*5),
                           initial_guess=guess)

    supported = ("collocation", "galerkin") if coefficient == "constant" else ("galerkin",)
    return ManufacturedCase("rational:{}:{}".format(coefficient, "nonlinear" if nonlinear else "linear"),
                            "Rational exact mixed fields ({})".format(coefficient), supported,
                            derivatives, factory)


def polynomial_segment_case():
    """Finite [0,2] cubic case with both endpoint values and slopes fixed."""
    def derivatives(x, arithmetic):
        s = arithmetic.scalar(1)-x/2
        return s**3, -arithmetic.scalar(3)/2*s*s, arithmetic.scalar(3)/2*s, -arithmetic.scalar(3)/4, arithmetic.scalar(0)

    def source(x, u, arithmetic):
        exact = derivatives(x, arithmetic)
        return exact[2]+(u-exact[0])+arithmetic.scalar("0.1")*(u**3-exact[0]**3)

    def factory():
        return FluxProblem(
            delta="2", domain=(0, 2), f=source,
            f_u=lambda x, u, a: 1+a.scalar("0.3")*u*u,
            boundary_conditions=((1, "-1.5", None, None, None),
                                 (0, 0, None, None, None)),
            initial_guess=lambda x, a: a.scalar("0.8")*derivatives(x, a)[0],
        )

    return ManufacturedCase("polynomial_segment", "Cubic on [0,2] with value/slope data",
                            ("collocation",), derivatives, factory)
