"""A genuine epsilon jump with continuous u,u',u'' and physical flux."""

from math import factorial

from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.manufactured.common import ManufacturedCase
from overscreening_breakdown.problems.flux import FluxProblem


def transmission_case():
    """Return the existing interface fixture with epsilon=1/2 about x=1.

    u=1/(1+x)+(5/12)*(x-1)_+^3/(1+x)^4, delta=2.
    At x=1, coefficient and derivative references use their right traces.
    Physical flux is continuous and equals 5/4 at the interface.
    """
    def derivatives(x, arithmetic):
        one = arithmetic.scalar(1)
        reciprocal = one/(one+x)
        coefficients = ((one,) if x < 1 else
                        (arithmetic.scalar(17)/12, -arithmetic.scalar(5)/2,
                         arithmetic.scalar(5), -arithmetic.scalar(10)/3))
        return tuple(sum(((-1)**order*factorial(k+order-1)//factorial(k-1)
                          *coefficient*reciprocal**(k+order)
                          for k, coefficient in enumerate(coefficients, 1)), arithmetic.scalar(0))
                     for order in range(5))

    def source(x, u, arithmetic):
        values = derivatives(x, arithmetic)
        epsilon = arithmetic.scalar(1 if x < 1 else 2)
        return (epsilon*(values[2]-4*values[4])+(u-values[0])
                +arithmetic.scalar(1)/8*(u**3-values[0]**3))

    def source_u(x, u, arithmetic):
        return 1+arithmetic.scalar(3)/8*u*u

    def factory():
        epsilon = Coefficient(lambda x, a: a.scalar(1 if x < 1 else 2),
                              lower_bound=1, upper_bound=2, wall_value=1, interfaces=(1,))
        return FluxProblem(delta="2", epsilon=epsilon, f=source, f_u=source_u,
                           boundary_conditions=(("1", None, None, "-6", None), (None,)*5))

    return ManufacturedCase("transmission", "C2 rational transmission solution", ("galerkin",),
                            derivatives, factory,
                            notes=("Right traces at x=1; no pointwise finite-N flux continuity guarantee.",))
