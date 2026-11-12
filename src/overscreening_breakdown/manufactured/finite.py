"""Native finite-interval references with genuine transmission conditions."""

from collections import OrderedDict
from math import comb, factorial

from overscreening_breakdown.coefficients import Coefficient
from overscreening_breakdown.layered import LayeredMaterial
from overscreening_breakdown.manufactured.common import ManufacturedCase, NativeFluxDescription
from overscreening_breakdown.rational_mapping import arithmetic_signature


def finite_segment_case(coefficient="constant", *, R="2", delta="2", nonlinear=False,
                        beta="0.125", nonzero_boundary=False):
    """Return a quartic reference, optionally corrected at material interfaces.

    ``coefficient`` is 'constant', 'smooth' (epsilon=1+x/R), or a LayeredMaterial.
    With constant/smooth epsilon, both fields are exactly represented at N=4.
    Layered cases add cubic-onset polynomials preserving u,u',u'' and flux.
    Corrections vanish with their first derivative at R. R and N remain
    independent physical-domain and approximation parameters.
    """
    layered = isinstance(coefficient, LayeredMaterial)
    if not layered and coefficient not in ("constant", "smooth"):
        raise ValueError("coefficient must be constant, smooth, or a LayeredMaterial")
    if not isinstance(nonlinear, bool) or not isinstance(nonzero_boundary, bool):
        raise TypeError("nonlinear and nonzero_boundary must be booleans")
    references = OrderedDict()

    def reference(arithmetic):
        key = (id(arithmetic), arithmetic_signature(arithmetic))
        if key in references:
            references.move_to_end(key)
            return references[key]
        a = arithmetic
        end, length_scale = a.scalar(R), a.scalar(delta)
        reaction = a.scalar(beta) if nonlinear else a.scalar(0)
        if (not a.isfinite(end) or end <= 0 or not a.isfinite(length_scale)
                or length_scale <= 0 or not a.isfinite(reaction) or reaction < 0):
            raise ValueError("finite references require R>0, delta>0 and beta>=0")
        zero, one = a.scalar(0), a.scalar(1)
        base = ((one, a.scalar(2), -one, one/2, one/3, zero) if nonzero_boundary else
                (one, -4*one/3, zero, zero, one/3, zero))
        material = coefficient.bind(a) if layered else None
        points = material.interfaces if layered else ()
        if any(point >= end for point in points):
            raise ValueError("all manufactured interfaces must lie strictly inside [0,R]")
        polynomials = [base]

        def polynomial_values(entries, x):
            t = x/end
            return tuple(sum((entries[k]*(factorial(k)//factorial(k-order))*t**(k-order)
                              for k in range(order, len(entries))), zero)/end**order
                         for order in range(5))

        for index, point in enumerate(points):
            t = point/end
            values = polynomial_values(polynomials[-1], point)
            ratio = material.trace(index, side="left")/material.trace(index, side="right")
            amplitude = ((one-ratio)*(values[1]-length_scale**2*values[3])*end**3
                         /(6*length_scale**2*(one-t)**2))
            # Expand (t-t_i)^3 (1-t)^2 with integer binomial factors.
            addition = [zero]*6
            for i in range(4):
                for j in range(3):
                    addition[i+j] += amplitude*comb(3, i)*(-t)**(3-i)*comb(2, j)*(-1)**j
            polynomials.append(tuple(value+change for value, change in
                                     zip(polynomials[-1], addition)))

        def derivatives(x, region=None):
            x = a.scalar(x)
            if not a.isfinite(x) or not zero <= x <= end:
                raise ValueError("finite reference coordinate lies outside [0,R]")
            if region is None:
                region = sum(x >= point for point in points)
            return polynomial_values(polynomials[region], x)

        def epsilon_pair(x):
            if layered:
                return material.epsilon(x), material.epsilon_dx(x)
            return (one+x/end, one/end) if coefficient == "smooth" else (one, zero)

        def source(x, u, owner):
            exact = derivatives(x)
            epsilon, epsilon_dx = epsilon_pair(x)
            value = (epsilon_dx*(exact[1]-length_scale**2*exact[3])
                     +epsilon*(exact[2]-length_scale**2*exact[4])+(u-exact[0]))
            return value if reaction == 0 else value+reaction*(u**3-exact[0]**3)

        def source_u(x, u, owner):
            return one+3*reaction*u*u

        boundaries = (((one, None, None, 3*one/end**3, None),
                       (17*one/6, 17*one/(6*end), None, None, None)) if nonzero_boundary else
                      ((one, None, None, zero, None), (zero, zero, None, None, None)))
        result = (derivatives, source, source_u, boundaries)
        references[key] = result
        if len(references) > 8:
            references.popitem(last=False)
        return result

    if layered:
        epsilon = coefficient.coefficient
    elif coefficient == "smooth":
        epsilon = Coefficient(lambda x, a: 1+x/a.scalar(R),
                              lower_bound=1, upper_bound=2, wall_value=1)
    else:
        epsilon = 1

    def factory():
        return NativeFluxDescription(
            delta=lambda a: a.scalar(delta), boundaries=lambda a: reference(a)[3],
            f=lambda x, u, a: reference(a)[1](x, u, a),
            f_u=lambda x, u, a: reference(a)[2](x, u, a),
            epsilon=epsilon, domain=(0, R),
        )

    kind = "layered" if layered else coefficient
    return ManufacturedCase(
        "finite-segment:{}:{}:{}".format(kind, "nonlinear" if nonlinear else "linear",
                                         "nonzero" if nonzero_boundary else "zero"),
        "Finite quartic transmission reference ({})".format(kind),
        ("collocation", "galerkin") if kind == "constant" else ("galerkin",),
        lambda x, a: reference(a)[0](x), factory,
        notes=("Independent w=-u'' is generally nonzero at R.",
               "Layered corrections preserve the physical flux and both right essential traces."),
    )
