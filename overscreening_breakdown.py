import numpy as np
import mpmath as mp

class overscreening_breakdown(object):
    def __init__(self):
        self.__domain = [0, np.inf]
        self.__sigma = 10.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__delta = 50.0
        self.__u0 = 100.0
        self.__u0xxx = 0
        self.__use_mpmath = np
        self.__use_mp_math = False        
        self.__set_functions()
        self.__delta_inverse_check = 0.0
        self.__boundaries = [[self.__u0,None,None,self.__u0xxx,None],[0,None,None,None,None]]
        self.__initial_guess_solution = lambda x: self.__u0*self.__exp(-x)

#       define axillary function to be used in function bellow for numpy and mpmath, if one needs high precision arithemtic        


#       these functions and constants that should be defined by a user 
    def __set_functions(self):
        self.__exp = np.vectorize(self.__use_mpmath.exp)
        self.__sqrt = np.vectorize(self.__use_mpmath.sqrt)
        self.__sinh = np.vectorize(self.__use_mpmath.sinh)
        self.__cosh = np.vectorize(self.__use_mpmath.cosh)
        self.__pi = self.__use_mpmath.pi
        return self


    def set_use_mp_math(self, mp_ref):
        self.__use_mp_math = True
        self.__use_mpmath = mp_ref
        self.__set_functions()

    def set_parameters(self, sigma = 10.0, mu = 1.0, gamma = 1.0, delta = 10.0, u0 = 1.0, u0xxx = 0.0, u0x = None, u0xx = None, u0xxxx = None, initial_guess_solution = None ):
        self.__sigma = sigma
        self.__mu = mu
        self.__gamma = gamma
        self.__delta = delta
        self.__u0 = u0
        self.__u0xxx = u0xxx
        self.__boundaries = [ [u0, u0x, u0xx, u0xxx, u0xxxx], [0,None,None,None,None] ]
        if type(initial_guess_solution) != type(None):
            self.__initial_guess_solution = initial_guess_solution

    def get_domain(self):
        return self.__domain

    def get_boundary_conditions(self):
        return self.__boundaries

    def __g_func(self, x):
        return 1.0/(self.__sqrt(2*self.__pi))*self.__exp(-(1/2)*((x)/self.__sigma)**2 ) 

    # x is in the problem domain,
    # q is a homotopy parameter i.e. q \in[0,1]
    def base_solution(self, x):
        return self.__initial_guess_solution(x)

    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self, x = None, u = None, ux = None, q = None):
        res = [None, None, 1, None, -self.__delta**2]
        if self.__delta>self.__delta_inverse_check:
            res = [None, None, 1/(self.__delta**2), None, -1]
        return res


    def right_hand_side(self, x, u, q):
        num = (self.__sinh(u) - self.__g_func(x)*0.5*self.__mu*self.__exp(u) )*self.__exp(-q*x)
        din = (1.0 + 2.0*self.__gamma*self.__sinh(u/2.0)**2)
        res = num/din
        if self.__delta>self.__delta_inverse_check:
            res = res/(self.__delta**2)
        return res

    def right_hand_side_linearization(self, x, u, q):
        num = self.__exp(-q*x)*(2*(self.__gamma + self.__cosh(u) - self.__gamma*self.__cosh(u)) + (self.__exp(u)*(-1 + self.__gamma) - self.__gamma)*self.__mu*self.__g_func(x))
        din = 2*(1 - self.__gamma + self.__gamma*self.__cosh(u))**2
        res = num/din
        if self.__delta>self.__delta_inverse_check:
            res = res/(self.__delta**2)
        # print("type(num) = ",type(num[0]),"type(din) = ", type(din[0]), "type(res) = ", type(res[0]))
        # print(num[0], din[0], res[0])
        return res