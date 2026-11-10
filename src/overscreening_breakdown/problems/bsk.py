import numpy as np
from types import SimpleNamespace

from overscreening_breakdown.arithmetic import NumpyArithmetic
from overscreening_breakdown.coefficients import ConstantCoefficient, as_coefficient
from overscreening_breakdown.problems.flux import FluxProblem


_UNSET = object()


def _gaussian(x, sigma, arithmetic):
    # Preserve the model's existing normalization (there is no 1/sigma factor).
    half = arithmetic.scalar("0.5")
    return arithmetic.exp(-half*(x/sigma)**2)/arithmetic.sqrt(2*arithmetic.pi)


def _physical_rhs(x, u, arithmetic, sigma, mu, gamma):
    half = arithmetic.scalar("0.5")
    numerator = arithmetic.sinh(u) - half*mu*_gaussian(x, sigma, arithmetic)*arithmetic.exp(u)
    denominator = 1 + 2*gamma*arithmetic.sinh(half*u)**2
    return numerator/denominator


def _physical_rhs_u(x, u, arithmetic, sigma, mu, gamma):
    # Algebraically the existing analytic derivative, arranged to avoid
    # cancellation of cosh(u) terms when gamma=1 and u is large.
    cosh = arithmetic.cosh(u)
    denominator = 1 - gamma + gamma*cosh
    numerator = (2*(gamma + (1-gamma)*cosh)
                 + mu*_gaussian(x, sigma, arithmetic)*((gamma-1)*arithmetic.exp(u)-gamma))
    return numerator/(2*denominator**2)

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
            exp=self.__exp, sinh=self.__sinh, cosh=self.__cosh,
            sqrt=self.__sqrt, pi=self.__pi,
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
