import numpy as np

class overscreening_breakdown(object):
    def __init__(self):
        self.__domain = [0, np.inf]
        self.__sigma = 10.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__delta = 50.0
        self.__u0 = 300.0
        self.__u0xxx = 0
        self.__delta_inverse_check = 0.0
        self.__boundaries = [[self.__u0,None,None,self.__u0xxx,None],[0,None,None,None,None]]
        self.__initial_guess_solution = lambda x: self.__u0*np.exp(-x)

    def set_parameters(self, sigma = 10.0, mu = 1.0, gamma = 1.0, delta = 10.0, u0 = 1.0, u0xxx = 0.0, u0x = None, u0xx = None, u0xxxx = None, initial_guess_solution = None ):
        self.__sigma = sigma
        self.__mu = mu
        self.__gamma = gamma
        self.__delta = delta
        self.__u0 = u0
        self.__u0xxx = u0xxx
        self.__boundaries = [ [u0, u0x, u0xx, u0xxx, u0xxxx], [None,None,None,None,None] ]
        if type(initial_guess_solution) != type(None):
            self.__initial_guess_solution = initial_guess_solution

    def get_domain(self):
        return self.__domain

    def get_boundary_conditions(self):
        return self.__boundaries

    def __g_func(self, x):
        return 1.0/(np.sqrt(2*np.pi))*np.exp(-(1/2)*((x)/self.__sigma)**2 ) 

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
        num = (np.sinh(u) - self.__g_func(x)*0.5*self.__mu*np.exp(u) )*np.exp(-q*x)
        din = (1 + 2.0*self.__gamma*np.sinh(u/2.0)**2)
        res = num/din
        if self.__delta>self.__delta_inverse_check:
            res = res/(self.__delta**2)
        return res

    def right_hand_side_linearization(self, x, u, q):

        num = np.exp(-q*x)*(2*(self.__gamma + np.cosh(u) - self.__gamma*np.cosh(u)) + (np.exp(u)*(-1 + self.__gamma) - self.__gamma)*self.__mu*self.__g_func(x))
        din = 2*(1 - self.__gamma + self.__gamma*np.cosh(u))**2
        res = num/din
        if self.__delta>self.__delta_inverse_check:
            res = res/(self.__delta**2)
        return res




