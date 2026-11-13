import numpy as np
from types import SimpleNamespace

from overscreening_breakdown.arithmetic import NumpyArithmetic
from overscreening_breakdown.coefficients import ConstantCoefficient, as_coefficient
from overscreening_breakdown.problems.flux import FluxProblem


_UNSET = object()


# Original implementations, retained for reference. The growing exponentials
# and squared denominator can overflow even when the result is representable.
# def _gaussian(x, sigma, arithmetic):
#     # Preserve the model's existing normalization (there is no 1/sigma factor).
#     half = arithmetic.scalar("0.5")
#     return arithmetic.exp(-half*(x/sigma)**2)/arithmetic.sqrt(2*arithmetic.pi)
#
# def _physical_rhs(x, u, arithmetic, sigma, mu, gamma):
#     half = arithmetic.scalar("0.5")
#     numerator = arithmetic.sinh(u) - half*mu*_gaussian(x, sigma, arithmetic)*arithmetic.exp(u)
#     denominator = 1 + 2*gamma*arithmetic.sinh(half*u)**2
#     return numerator/denominator
#
# def _physical_rhs_u(x, u, arithmetic, sigma, mu, gamma):
#     # Algebraically the existing analytic derivative, arranged to avoid
#     # cancellation of cosh(u) terms when gamma=1 and u is large.
#     cosh = arithmetic.cosh(u)
#     denominator = 1 - gamma + gamma*cosh
#     numerator = (2*(gamma + (1-gamma)*cosh)
#                  + mu*_gaussian(x, sigma, arithmetic)*((gamma-1)*arithmetic.exp(u)-gamma))
#     return numerator/(2*denominator**2)


def _gaussian_exponent(x, sigma, arithmetic):
    ratio = x/sigma
    # Multiply by 1/2 first: ratio**2 can overflow while ratio**2/2 is finite.
    return (arithmetic.scalar("0.5")*ratio)*ratio


def _gaussian(x, sigma, arithmetic):
    """Existing Gaussian normalization, without a 1/sigma factor."""
    with np.errstate(over="ignore", under="ignore"):
        return arithmetic.exp(-_gaussian_exponent(x, sigma, arithmetic))/arithmetic.sqrt(
            2*arithmetic.pi)


def _signed_logsum(terms, arithmetic):
    """Return sign and log magnitude of a short signed exponential sum.

    Cancel the largest terms first so a small surviving term is not lost
    before two equal and opposite leading terms cancel.
    """
    terms = sorted(((sign, value) for sign, value in terms if sign and value != -np.inf),
                   key=lambda term: term[1], reverse=True)
    sign, value = 0, None
    for other_sign, other in terms:
        if not sign:
            sign, value = other_sign, other
        elif value == other and sign != other_sign:
            sign, value = 0, None
        else:
            if value < other:
                sign, other_sign = other_sign, sign
                value, other = other, value
            difference = other-value
            if sign == other_sign:
                value += arithmetic.log1p(arithmetic.exp(difference))
            else:
                value += arithmetic.log(-arithmetic.expm1(difference))
    return sign, value


def _logterm(coefficient, exponent, arithmetic):
    if coefficient == 0:
        return 0, None
    return (1 if coefficient > 0 else -1,
            arithmetic.log(abs(coefficient))+exponent)


def _exp_logsum(terms, shift, arithmetic):
    sign, value = _signed_logsum(terms, arithmetic)
    return arithmetic.scalar(0) if not sign else sign*arithmetic.exp(value+shift)


def _bsk_log_value(u, b, mu, exponent, normalizer, gamma, arithmetic, derivative):
    """Float64 fallback when individual scaled factors leave its range."""
    a = arithmetic
    size = abs(u)
    zero = a.scalar(0)
    log_two = a.log(a.scalar(2))
    if gamma == 0:
        # The unsaturated source may genuinely overflow. Combining 1-b first
        # also retains the decaying solution when that coefficient is zero.
        terms = [_logterm(1-b, u, a), (1 if derivative else -1, -u)]
        return _exp_logsum(terms, -log_two, a)

    k = -a.expm1(-size)
    log_gamma = a.log(gamma)
    if k == 0:
        return 1-b/2 if derivative else -b/2
    log_k = a.log(k)
    _, log_d = _signed_logsum(
        [(1, log_gamma+2*log_k), (1, log_two-size)], a)
    if not derivative:
        if b == 0 and mu != 0:
            # A Gaussian that underflows before division can still matter.
            log_b = a.log(abs(mu))-exponent-a.log(normalizer)
            terms = [(1, u-size), (-1, -u-size),
                     (-1 if mu > 0 else 1, log_b+u-size)]
        else:
            terms = [_logterm(1-b, u-size, a), (-1, -u-size)]
        return _exp_logsum(terms, -log_d, a)

    if gamma > 2 and size < 1:
        t = a.exp(-size)
        p = k*(1+t)
        base = (2-b)-p if u >= 0 else (2-b)-(1-b)*p
        correction = b-k if u >= 0 else -(k+b*t)
        terms = [_logterm(base, zero, a),
                 _logterm(correction, log_gamma+log_k, a)]
    elif size < 1:
        t = a.exp(-size)
        p = k*(1+t)
        base = (2-b)-p if u >= 0 else (2-b)-(1-b)*p
        terms = []
        if gamma != 1:
            sign_base, log_base = _logterm(base, a.log(abs(1-gamma)), a)
            terms.append(((1 if gamma < 1 else -1)*sign_base, log_base))
        terms.append(_logterm(2-b, log_gamma-size, a))
    else:
        # Expanded factored derivative: at gamma=1 the first two terms
        # vanish identically, retaining its much smaller exponential tail.
        log_terms = [(1-b, zero if u >= 0 else -2*size),
                     (a.scalar(1), -2*size if u >= 0 else zero)]
        terms = []
        if gamma != 1:
            sign_g = 1 if gamma < 1 else -1
            log_g = a.log(abs(1-gamma))
            for coefficient, power in log_terms:
                term_sign, term_log = _logterm(coefficient, log_g+power, a)
                terms.append((sign_g*term_sign, term_log))
        terms.append(_logterm(2-b, log_gamma-size, a))
    return _exp_logsum(terms, log_two-size-2*log_d, a)


def _bsk_scalar(x, u, arithmetic, sigma, mu, gamma, derivative):
    """Evaluate f or f_u using t=exp(-abs(u)) and d=gamma*(1-t)^2+2t.

    For positive u, f=((1-b)-t²)/d; for negative u, f=((1-b)*t²-1)/d.
    The derivative is 2t*C/d², with C=(1-gamma)*A+gamma*t*(2-b),
    A=1+t²-b for positive u and A=1+(1-b)*t² for negative u.
    Equivalent near-zero forms avoid subtracting rounded constant terms.
    """
    a = arithmetic
    x, u = a.scalar(x), a.scalar(u)
    is_mp = a.use_mpmath
    # Suppress only intermediates covered by the scaled/logarithmic fallback.
    # A genuinely nonfinite final value is rejected by BoundFluxProblem.
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        normalizer = a.sqrt(2*a.pi)
        exponent = _gaussian_exponent(x, sigma, a) if mu else a.scalar(0)
        gaussian = a.exp(-exponent) if mu else a.scalar(0)
        b = (mu/normalizer)*gaussian
        if not is_mp and mu and (gaussian < np.finfo(np.float64).tiny
                                 or b == 0 or not np.isfinite(b)):
            # Multiplication can rescue even a *subnormal* Gaussian, after
            # its relative precision has already been lost. Rebuild it first.
            b = (1 if mu > 0 else -1)*a.exp(a.log(abs(mu))-exponent-a.log(normalizer))
        size = abs(u)
        if gamma == 0:
            if is_mp or size < 300:
                if size < 1:
                    # These combinations also retain f_u when b=2 and the
                    # two constant terms cancel at an arbitrarily small u.
                    if derivative:
                        value = (1-b/2)*a.cosh(u)-(b/2)*a.sinh(u)
                    else:
                        value = (1-b/2)*a.sinh(u)-(b/2)*a.cosh(u)
                else:
                    # Combine the growing coefficients before exponentiating:
                    # b=1 leaves a decaying field, even in arbitrary precision.
                    value = ((1-b)/2)*a.exp(u)+(1 if derivative else -1)*a.exp(-u)/2
                if is_mp or np.isfinite(value):
                    return value
            return _bsk_log_value(u, b, mu, exponent, normalizer, gamma, a, derivative)

        if not is_mp and size > 700:
            return _bsk_log_value(u, b, mu, exponent, normalizer, gamma, a, derivative)
        t = a.exp(-size)
        k = -a.expm1(-size)
        d = (gamma*k)*k+2*t
        p = k*(1+t)
        if not derivative:
            if size < 1:
                numerator = p-b if u >= 0 else -p-b*t*t
            else:
                numerator = (1-b)-t*t if u >= 0 else (1-b)*t*t-1
            value = numerator/d
            resolved = numerator != 0 and not (size > 350 and b == 1)
        else:
            if gamma > 2 and size < 1:
                base = (2-b)-p if u >= 0 else (2-b)-(1-b)*p
                correction = b-k if u >= 0 else -(k+b*t)
                factor = base+(gamma*k)*correction
            else:
                if size < 1:
                    base = (2-b)-p if u >= 0 else (2-b)-(1-b)*p
                else:
                    base = (1-b)+t*t if u >= 0 else 1+(1-b)*t*t
                factor = (1-gamma)*base+(gamma*t)*(2-b)
            # Since d >= 2t, the first ratio is bounded by one. Never form d².
            ratio = t/d
            value = (2*ratio)*(factor/d)
            # A tiny ratio can be amplified by the other factor. Reconstruct
            # from logs before it loses digits or underflows to a false zero.
            resolved = (factor != 0 and value != 0
                        and (is_mp or ratio >= np.finfo(np.float64).tiny))
        if is_mp or (np.isfinite(value) and resolved):
            return value
        return _bsk_log_value(u, b, mu, exponent, normalizer, gamma, a, derivative)


def _bsk_values(x, u, arithmetic, sigma, mu, gamma, derivative):
    if np.ndim(x) or np.ndim(u):
        # The public physical interface is scalar; legacy classes also accept
        # arrays and broadcast coordinates against fields in either backend.
        evaluate = np.vectorize(
            lambda xi, ui: _bsk_scalar(xi, ui, arithmetic, sigma, mu, gamma, derivative),
            otypes=[object if arithmetic.use_mpmath else np.float64],
        )
        return evaluate(x, u)
    return _bsk_scalar(x, u, arithmetic, sigma, mu, gamma, derivative)


def _physical_rhs(x, u, arithmetic, sigma, mu, gamma):
    return _bsk_values(x, u, arithmetic, sigma, mu, gamma, False)


def _physical_rhs_u(x, u, arithmetic, sigma, mu, gamma):
    return _bsk_values(x, u, arithmetic, sigma, mu, gamma, True)

class overscreening_breakdown(object):
    def __init__(self, *, epsilon=1):
        self.__domain = [0, np.inf]
        self.__sigma = 10.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__delta = 50.0
        self.__epsilon = as_coefficient(epsilon)
        self.__u0 = 100.0
        self.__u0xxx = 0
        self.__use_mpmath = np
        self.__use_mp_math = False        
        self.__set_functions()
        self.__delta_inverse_check = 0.0
        self.__boundaries = [[self.__u0,None,None,self.__u0xxx,None],[None,None,None,None,None]]
        self.__initial_guess_solution = None

#       define axillary function to be used in function bellow for numpy and mpmath, if one needs high precision arithemtic        


#       these functions and constants that should be defined by a user 
    def __set_functions(self):
        self.__exp = np.vectorize(self.__use_mpmath.exp)
        self.__sqrt = np.vectorize(self.__use_mpmath.sqrt)
        self.__sinh = np.vectorize(self.__use_mpmath.sinh)
        self.__cosh = np.vectorize(self.__use_mpmath.cosh)
        self.__pi = self.__use_mpmath.pi
        self.__arithmetic = SimpleNamespace(
            scalar=self.__use_mpmath.mpf if self.__use_mp_math else np.float64,
            use_mpmath=self.__use_mp_math,
            exp=self.__use_mpmath.exp, expm1=self.__use_mpmath.expm1,
            log=self.__use_mpmath.log, log1p=self.__use_mpmath.log1p,
            sinh=self.__use_mpmath.sinh, cosh=self.__use_mpmath.cosh,
            sqrt=self.__use_mpmath.sqrt, pi=self.__pi,
        )
        return self


    def set_use_mp_math(self, mp_ref):
        self.__use_mp_math = True
        self.__use_mpmath = mp_ref
        self.__set_functions()

    def set_parameters(self, sigma = 10.0, mu = 1.0, gamma = 1.0, delta = 10.0, u0 = 1.0, u0xxx = 0.0, u0x = None, u0xx = None, u0xxxx = None, initial_guess_solution = None, *, epsilon=_UNSET):
        coefficient = self.__epsilon if epsilon is _UNSET else as_coefficient(epsilon)
        self.__sigma = sigma
        self.__mu = mu
        self.__gamma = gamma
        self.__delta = delta
        self.__epsilon = coefficient
        self.__u0 = u0
        self.__u0xxx = u0xxx
        self.__boundaries = [ [u0, u0x, u0xx, u0xxx, u0xxxx], [None,None,None,None,None] ]
        if type(initial_guess_solution) != type(None):
            self.__initial_guess_solution = initial_guess_solution

    def get_domain(self):
        return self.__domain

    def get_boundary_conditions(self):
        return self.__boundaries

    @property
    def delta(self):
        """The existing physical parameter, before collocation normalization."""
        return self.__delta

    @property
    def epsilon(self):
        """Immutable coefficient description; bind it for scalar evaluation."""
        return self.__epsilon

    def validate_collocation_compatibility(self):
        """The existing collocation operator represents epsilon identically 1."""
        if (not isinstance(self.__epsilon, ConstantCoefficient)
                or not self.__epsilon.bind(NumpyArithmetic()).is_identity):
            raise ValueError("nontrivial epsilon requires the Galerkin flux solver; "
                             "the current collocation solver supports epsilon=1 only")

    def __physical_parameters(self):
        return tuple(self.__arithmetic.scalar(value)
                     for value in (self.__sigma, self.__mu, self.__gamma))

    def physical_rhs(self, x, u):
        """Unscaled physical f(x,u), with the legacy scalar/array arithmetic."""
        return _physical_rhs(x, u, self.__arithmetic, *self.__physical_parameters())

    def physical_rhs_u(self, x, u):
        """Analytic partial derivative of the unscaled physical f with respect to u."""
        return _physical_rhs_u(x, u, self.__arithmetic, *self.__physical_parameters())

    def bind(self, arithmetic):
        """Snapshot the physical problem in the supplied solver-owned arithmetic.

        This leaves the source problem and global mpmath precision untouched.
        A custom legacy initial_guess_solution(x) is retained as supplied; its
        closure must use arithmetic compatible with x and remain unchanged.
        FluxProblem also supports explicit initial_guess(x, arithmetic).
        """
        sigma, mu, gamma = (arithmetic.scalar(value)
                            for value in (self.__sigma, self.__mu, self.__gamma))
        if not all(arithmetic.isfinite(value) for value in (sigma, mu, gamma)):
            raise ValueError("sigma, mu and gamma must be finite")
        if sigma <= 0 or gamma < 0:
            raise ValueError("the physical BSK model requires sigma > 0 and gamma >= 0")
        initial_guess = self.__initial_guess_solution
        return FluxProblem(
            delta=self.__delta,
            epsilon=self.__epsilon,
            f=lambda x, u, a: _physical_rhs(x, u, a, sigma, mu, gamma),
            f_u=lambda x, u, a: _physical_rhs_u(x, u, a, sigma, mu, gamma),
            domain=self.get_domain(),
            boundary_conditions=self.get_boundary_conditions(),
            initial_guess=(None if initial_guess is None
                           else lambda x, a: initial_guess(x)),
        ).bind(arithmetic)

    # x is in the problem domain,
    # q is a homotopy parameter i.e. q \in[0,1]
    def base_solution(self, x):
        if self.__initial_guess_solution is not None:
            return self.__initial_guess_solution(x)
        return self.__arithmetic.scalar(self.__u0)*self.__exp(-x)

    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self, x = None, u = None, ux = None, q = None):
        self.validate_collocation_compatibility()
        delta = self.__arithmetic.scalar(self.__delta)
        res = [None, None, 1, None, -delta**2]
        if delta>self.__delta_inverse_check:
            res = [None, None, 1/(delta**2), None, -1]
        return res


    def right_hand_side(self, x, u, q):
        self.validate_collocation_compatibility()
        res = self.physical_rhs(x, u)*self.__exp(-q*x)
        delta = self.__arithmetic.scalar(self.__delta)
        if delta>self.__delta_inverse_check:
            res = res/(delta**2)
        return res

    def right_hand_side_linearization(self, x, u, q):
        self.validate_collocation_compatibility()
        res = self.physical_rhs_u(x, u)*self.__exp(-q*x)
        delta = self.__arithmetic.scalar(self.__delta)
        if delta>self.__delta_inverse_check:
            res = res/(delta**2)
        return res



class overscreening_breakdown_segment(overscreening_breakdown):
    def __init__(self, *, epsilon=1):
        super().__init__(epsilon=epsilon)
        self._overscreening_breakdown__domain = [0, 50]
        self.__boundaries = [[self._overscreening_breakdown__u0,None,None,self._overscreening_breakdown__u0xxx,None],[0,0,None,None,None]]
    
    def get_domain(self):
        return self._overscreening_breakdown__domain

    def get_boundary_conditions(self):
        return self.__boundaries

    def set_parameters(self, sigma = 10.0, mu = 1.0, gamma = 1.0, delta = 10.0, u0 = 1.0, u0xxx = 0.0, u0x = None, u0xx = None, u0xxxx = None, initial_guess_solution = None, *, epsilon=_UNSET):
        super().set_parameters(sigma = sigma, mu = mu, gamma = gamma, delta = delta, u0 = u0, u0xxx = u0xxx, u0x = u0x, u0xx = u0xx, u0xxxx = u0xxxx, initial_guess_solution = initial_guess_solution, epsilon=epsilon)
        self.__boundaries = [ [u0, u0x, u0xx, u0xxx, u0xxxx], [0,0,None,None,None] ]
