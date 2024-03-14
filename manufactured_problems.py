import numpy as np

class basic_problem(object):
    def __init__(self):
        self.__use_mpmath = np
        self.__set_functions()
        self._file_name_prefix = "problem_"
        self._L_all = [1/4,1/2,1,2,3,4,5,6,7,8]
        self._N_all = [10,20,40,60,80,100,120,140]
        self._domain = [0, np.inf]
        self._basis_domain = [0,self._pi]
        self._bondaries = [[1,None,None,None,None],[0,None,None,None,None]]
        self._operator = [None, None, 1, None, 1]
        #for nonlinear problems
        self._visualize = False
        self._use_method = "newton"
        self._tolerance = 1.0e-7
        self.__use_mp_math = False

    def file_name_prefix(self):
        return self._file_name_prefix
    def get_L(self):
        return self._L_all
    def get_N(self):
        return self._N_all
    def get_basis_domain(self):
        return self._basis_domain
    def get_domain(self):
        return self._domain
    def operator(self):
        return self._operator
    def get_boundary_conditions(self):
        return self._bondaries
    def get_tolerance(self):
        return self._tolerance
    def visualize(self):
        return self._visualize
    def nonlinear_method(self):
        return self._use_method
#       these functions and constants that should be defined by a user 
    def __set_functions(self):
        self._exp = np.vectorize(self.__use_mpmath.exp)
        self._sqrt = np.vectorize(self.__use_mpmath.sqrt)
        self._sinh = np.vectorize(self.__use_mpmath.sinh)
        self._cosh = np.vectorize(self.__use_mpmath.cosh)
        self._tanh = np.vectorize(self.__use_mpmath.tanh)
        self._log = np.vectorize(self.__use_mpmath.log)
        self._pi = self.__use_mpmath.pi
        self._tolerance = 1.0e-30
        return self

    def set_use_mp_math(self, mp_ref):
        self.__use_mp_math = True
        self.__use_mpmath = mp_ref
        self.__set_functions()

class problem_1(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_1_"
        self._bondaries = [[1,None,None,None,None],[0,None,None,None,None]]
        self._operator = [None, None, 1, None, -10]
    
    def get_name(self):
        return "$u_{xx}-10u_{xxxx}=f(x)$, $u=exp(-x^2)$"

    def solution_in_domain(self, x):
        return self._exp(-x**2)

    def rhs_in_domain(self, x):
        return -2*self._exp(-x**2)+4*self._exp(-x**2)*x**2-10*(12*self._exp(-x**2)-48*self._exp(-x**2)*x**2+16*self._exp(-x**2)*x**4)

    def operator(self):
        return self._operator
    def file_name_prefix(self):
        return self._file_name_prefix

class problem_2(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_2_"
        self._bondaries = [[1,None,None,None,None],[0,None,None,None,None]]
        # self._operator = [None, None, 1/2500, None, -1]
        self._operator = [None, None, -2500, None, 1]
    
    def get_name(self):
        return "$-2500 u_{xx}+u_{xxxx}=f(x)$, $u=exp(-x)$"

    def solution_in_domain(self, x):
        return self._exp(-x)

    def rhs_in_domain(self, x):
        # return -12*self._exp(-x**2) + 48*self._exp(-x**2)*x**2 -16*self._exp(-x**2)*x**4 + (-2*self._exp(-x**2) + 4*self._exp(-x**2)*x**2)/2500 
        return -2499*self._exp(-x)

    def operator(self):
        return self._operator
    def file_name_prefix(self):
        return self._file_name_prefix

class problem_3(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_3_"
        self._bondaries = [[1,None,None,None,None],[0,None,None,None,None]]
        self._operator = [None, None, -1, None, None]
    
    def get_name(self):
        return "$-u_{xx}=f(x)$, $u=exp(-x^2)$"

    def solution_in_domain(self, x):
        return self._exp(-x**2)

    def rhs_in_domain(self, x):
        return 2*self._exp(-x**2) - 4*self._exp(-x**2)*x**2

    def operator(self):
        return self._operator
    def file_name_prefix(self):
        return self._file_name_prefix
  

class problem_4(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_4_"
        self._bondaries = [[1,None,None,2,None],[0,None,None,None,None]]
        self._operator = [None, None, -1, None, 1]

    def get_name(self):
        return "$-u_{xx}+u_{xxxx}=f(x)$, $u=exp(-x) cos(x)$"        

    def solution_in_domain(self, x):
        return self._exp(-x)*self.__cos(x)

    def rhs_in_domain(self, x):
        return -4*self._exp(-x)*self.__cos(x) - 2*self._exp(-x)*self.__sin(x)

    def operator(self):
        return self._operator
    def file_name_prefix(self):
        return self._file_name_prefix        


class problem_5(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_5_"
        self._bondaries = [[1,None,None,None,None],[0,None,None,None,None]]
        self._operator = [None, None, 1, None, -1]

    def get_name(self):
        return "$u_{xx}-u_{xxxx}=f(x)$, $u=1/(x+1)$"        

    def solution_in_domain(self, x):
        # return 1/(x**2+1)
        return 1/(x+1)
        
    def rhs_in_domain(self, x):     
        # return (384*x**4)/(1 + x**2)**5 - (288*x**2)/(1 + x**2)**4 + 24/(1 + x**2)**3 - (8*x**2)/(1 + x**2)**3 + 2/(1 + x**2)**2 
        return -24/(1+x)**5+2/(1+x)**3

    def operator(self):
        return self._operator
    def file_name_prefix(self):
        return self._file_name_prefix  

class problem_6(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_6_"
        self._bondaries = [[self.__cos(5),None,None,None,None],[0,None,None,None,None]]
        self._operator = [None, None, -1, None, 1]

    def get_name(self):
        return "$-u_{xx}+u_{xxxx}=f(x)$, $u=1/(x+1) cos(1/(x+1/5))$"        

    def solution_in_domain(self, x):
        return 1/(x+1)*self.__cos(1/(x+1/5))

    def rhs_in_domain(self, x):   
        return (24*self.__cos(1/(1/5+x)))/(1+x)**5-(2*self.__cos(1/(1/5+x)))/(1+x)**3-(12*self.__cos(1/(1/5+x)))/((1/5+x)**4*(1+x)**3)-(24*self.__cos(1/(1/5+x)))/((1/5+x)**5*(1+x)**2)+self.__cos(1/(1/5+x))/((1/5+x)**8*(1+x))-(36*self.__cos(1/(1/5+x)))/((1/5+x)**6*(1+x))+self.__cos(1/(1/5+x))/((1/5+x)**4*(1+x))-(24*self.__sin(1/(1/5+x)))/((1/5+x)**2*(1+x)**4)-(24*self.__sin(1/(1/5+x)))/((1/5+x)**3*(1+x)**3)+(4*self.__sin(1/(1/5+x)))/((1/5+x)**6*(1+x)**2)-(24*self.__sin(1/(1/5+x)))/((1/5+x)**4*(1+x)**2)+(2*self.__sin(1/(1/5+x)))/((1/5+x)**2*(1+x)**2)+(12*self.__sin(1/(1/5+x)))/((1/5+x)**7*(1+x))-(24*self.__sin(1/(1/5+x)))/((1/5+x)**5*(1+x))+(2*self.__sin(1/(1/5+x)))/((1/5+x)**3*(1+x))

    def operator(self):
        return self._operator
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           


class problem_7(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_7_"
        self._bondaries = [[self.__cos(1),None,None,None,None],[0,None,None,None,None]]
        self._operator = [None, None, -1, None, 1]

    def get_name(self):
        return "$-u_{xx}+u_{xxxx}=f(x)$, $u=1/(x^2+1) cos(1/(x+1))$"        

    def solution_in_domain(self, x):
        return 1/(x**2+1)*self.__cos(1/(x+1))

    def rhs_in_domain(self, x):   
        return (384*x**4*self.__cos(1/(1+x)))/(1+x**2)**5-(288*x**2*self.__cos(1/(1+x)))/(1+x**2)**4+(24*self.__cos(1/(1+x)))/(1+x**2)**3-(8*x**2*self.__cos(1/(1+x)))/(1+x**2)**3-(48*x**2*self.__cos(1/(1+x)))/((1+x)**4*(1+x**2)**3)+(2*self.__cos(1/(1+x)))/(1+x**2)**2-(48*x*self.__cos(1/(1+x)))/((1+x)**5*(1+x**2)**2)+(12*self.__cos(1/(1+x)))/((1+x)**4*(1+x**2)**2)+self.__cos(1/(1+x))/((1+x)**8*(1+x**2))-(36*self.__cos(1/(1+x)))/((1+x)**6*(1+x**2))+self.__cos(1/(1+x))/((1+x)**4*(1+x**2))-(192*x**3*self.__sin(1/(1+x)))/((1+x)**2*(1+x**2)**4)-(96*x**2*self.__sin(1/(1+x)))/((1+x)**3*(1+x**2)**3)+(96*x*self.__sin(1/(1+x)))/((1+x)**2*(1+x**2)**3)+(8*x*self.__sin(1/(1+x)))/((1+x)**6*(1+x**2)**2)-(48*x*self.__sin(1/(1+x)))/((1+x)**4*(1+x**2)**2)+(24*self.__sin(1/(1+x)))/((1+x)**3*(1+x**2)**2)+(4*x*self.__sin(1/(1+x)))/((1+x)**2*(1+x**2)**2)+(12*self.__sin(1/(1+x)))/((1+x)**7*(1+x**2))-(24*self.__sin(1/(1+x)))/((1+x)**5*(1+x**2))+(2*self.__sin(1/(1+x)))/((1+x)**3*(1+x**2))

    def operator(self):
        return self._operator
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries        


class problem_8(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_8_"
        self._bondaries = [[1,None,None,None,None],[0,None,None,None,None]]
        self._operator = [None, None, None, None, 1]

    def get_name(self):
        return "$u_{xxxx}=f(x)$, $u=exp(-x^2)$"        

    def solution_in_domain(self, x):
        return self._exp(-x**2)

    def rhs_in_domain(self, x):
        return 12*self._exp(-x**2) - 48*self._exp(-x**2)*x**2 + 16*self._exp(-x**2)*x**4

    def operator(self):
        return self._operator
    def file_name_prefix(self):
        return self._file_name_prefix  


class nonlinear_problem_1(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "nonlinear_problem_1_"
        self.__sigma = 1.0
        self.__mu = 0.0
        self.__gamma = 0
        self.__u0 = 1
        self.__u0xxx = 0

        self._bondaries = [[self.__u0, None, None, None, None],[0,None,None,None,None]]
        self._operator = [None, None, 1, None, None]

    def get_name(self):
        return "$u_{xx}=sinh(u)$, u - exact."
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain
    def __g_func(self, x):
        return 1.0/(self._sqrt(2*self._pi))*self._exp(-(1/2)*((x)/self.__sigma)**2 ) 

    def base_solution(self, x):
        return self._exp(-x)
    
    def solution_in_domain(self, x):
        p1 = 1+self._tanh(self.__u0/4)*self._exp(-x)
        p2 = 1-self._tanh(self.__u0/4)*self._exp(-x)
        return 2*self._log(p1/p2)

    def __residual(self, x):
       return 0
        
    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        return res
    def right_hand_side(self, x, u, q):
        num = (self._sinh(u) - self.__g_func(x)*0.5*self.__mu*self._exp(u) )*self._exp(-q*x)
        din = (1 + 2.0*self.__gamma*self._sinh(u/2.0)**2)
        res = num/din
        return res + self.__residual(x)*self._exp(-q*x)
    def right_hand_side_linearization(self, x, u, q):
        num = self._exp(-q*x)*(2*(self.__gamma + self._cosh(u) - self.__gamma*self._cosh(u)) + (self._exp(u)*(-1 + self.__gamma) - self.__gamma)*self.__mu*self.__g_func(x))
        din = 2*(1 - self.__gamma + self.__gamma*self._cosh(u))**2
        res = num/din
        return res


class nonlinear_problem_2(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "nonlinear_problem_2_"
        self.__sigma = 1.0
        self.__mu = 1.0
        self.__gamma = 1/2
        self.__u0 = 1.0
        self.__u0xxx = 0

        self._bondaries = [[self.__u0, None, None, None, None],[0,None,None,None,None]]
        self._operator = [None, None, -1/2500, None, 1]

    def get_name(self):
        return "$1/2500u_{xx}-u_{xxxx}=f(x,u)-g(x)$, $u=exp(-x)$"        
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain
    def __g_func(self, x):
        return 1.0/(self._sqrt(2*self._pi))*self._exp(-(1/2)*((x)/self.__sigma)**2 ) 
    def base_solution(self, x):
        return self._exp(-x)
    
    def solution_in_domain(self, x):
        return self._exp(-x)

    def __residual(self, x):
        # num = -self._exp(self._exp(-x)-x**2/2)/(2*self._sqrt(2*self._pi))+self._sinh(self._exp(-x))
        # din = 1 + self._sinh(self._exp(-x)/2)**2
        # return -num/din
        return (2499*self._exp(-x))/2500-(-(self._exp(self._exp(-x)-x**2/2)/(2*self._sqrt(2*self._pi)))+self._sinh(self._exp(-x)))/(1+self._sinh(self._exp(-x)/2)**2)

    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        return res
    def right_hand_side(self, x, u, q):
        num = (self._sinh(u) - self.__g_func(x)*0.5*self.__mu*self._exp(u) )*self._exp(-q*x)
        din = (1 + 2.0*self.__gamma*self._sinh(u/2.0)**2)
        res = num/din
        return res + self.__residual(x)
    def right_hand_side_linearization(self, x, u, q):
        num = self._exp(-q*x)*(2*(self.__gamma + self._cosh(u) - self.__gamma*self._cosh(u)) + (self._exp(u)*(-1 + self.__gamma) - self.__gamma)*self.__mu*self.__g_func(x))
        din = 2*(1 - self.__gamma + self.__gamma*self._cosh(u))**2
        res = num/din
        return res


class nonlinear_problem_3(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "nonlinear_problem_3_"
        self.__sigma = 1.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__u0 = 1.0
        self.__u0xxx = 0

        self._bondaries = [[self.__u0, None, None, None, None],[0,None,None,None,None]]
        # self._operator = [None, None, -1, None, 1]
        self._operator = [None, None, -1, None, 1]

    def get_name(self):
        return "$-u_{xx}+u_{xxxx}=f(x,u)-g(x)$, $u=exp(-x^2)$"        
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain
    def __g_func(self, x):
        return 1.0/(self._sqrt(2*self._pi))*self._exp(-(1/2)*((x)/self.__sigma)**2 ) 
    def base_solution(self, x):
        return self._exp(-x**2)
    
    def solution_in_domain(self, x):
        # return self._exp(-x)
        return self._exp(-x**2)

    def __residual(self, x):
        # num = self._sinh(self._exp(-x))
        # din = 1 + self._sinh(self._exp(-x)/2)**2
        # return -num/din
        return 14*self._exp(-x**2)-52*self._exp(-x**2)*x**2+16*self._exp(-x**2)*x**4-self._sinh(self._exp(-x**2))/(1+self._sinh(self._exp(-x**2)/2)**2)

    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        return res
    def right_hand_side(self, x, u, q):
        num = (self._sinh(u) - self.__g_func(x)*0.5*self.__mu*self._exp(u) )*self._exp(-q*x)
        din = (1 + 2.0*self.__gamma*self._sinh(u/2.0)**2)
        res = num/din
        return res + self.__residual(x)
    def right_hand_side_linearization(self, x, u, q):
        num = self._exp(-q*x)*(2*(self.__gamma + self._cosh(u) - self.__gamma*self._cosh(u)) + (self._exp(u)*(-1 + self.__gamma) - self.__gamma)*self.__mu*self.__g_func(x))
        din = 2*(1 - self.__gamma + self.__gamma*self._cosh(u))**2
        res = num/din
        return res

class nonlinear_problem_4(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "nonlinear_problem_4_"
        self.__sigma = 1.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__u0 = 1.0
        self.__u0xxx = 0

        self._bondaries = [[self.__u0, None, None, None, None],[0,None,None,None,None]]
        self._operator = [None, None, -1, None, 1]


    def get_name(self):
        return "$-u_{xx}+u_{xxxx}=f(x,u)-g(x)$, $u=1/(x+1)$"        
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain
    def __g_func(self, x):
        return 1.0/(self._sqrt(2*self._pi))*self._exp(-(1/2)*((x)/self.__sigma)**2 ) 
    def base_solution(self, x):
        return self._exp(-x)
    
    def solution_in_domain(self, x):
        return 1/(x+1)

# 24/(1 + x)^5 - 2/(1 + x)^3 - Sinh[1/(1 + x)]/( 1 + Sinh[1/(2 (1 + x))]^2)

    def __residual(self, x):
        p1 = 24/(1 + x)**5 - 2/(1 + x)**3
        num = self._sinh(1/(1 + x))
        din = (1 + self._sinh(1/(2*(1 + x)))**2)
        return p1-num/din

    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        return res
    def right_hand_side(self, x, u, q):
        num = (self._sinh(u) - self.__g_func(x)*0.5*self.__mu*self._exp(u) )*self._exp(-q*x)
        din = (1 + 2.0*self.__gamma*self._sinh(u/2.0)**2)
        res = num/din
        return res + self.__residual(x)*self._exp(-q*x)
    def right_hand_side_linearization(self, x, u, q):
        num = self._exp(-q*x)*(2*(self.__gamma + self._cosh(u) - self.__gamma*self._cosh(u)) + (self._exp(u)*(-1 + self.__gamma) - self.__gamma)*self.__mu*self.__g_func(x))
        din = 2*(1 - self.__gamma + self.__gamma*self._cosh(u))**2
        res = num/din
        return res


class nonlinear_problem_5(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "nonlinear_problem_5_"
        self.__sigma = 1.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__u0 = self.__cos(5)
        self.__u0xxx = 0

        self._bondaries = [[self.__u0, None, None, None, None],[0,None,None,None,None]]
        self._operator = [None, None, -1, None, 1]


    def get_name(self):
        return "$-u_{xx}+u_{xxxx}=f(x,u)-g(x)$, $u= 1/(x+1) cos(1/(x+1/5))$"        
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain
    def __g_func(self, x):
        return 1.0/(self._sqrt(2*self._pi))*self._exp(-(1/2)*((x)/self.__sigma)**2 ) 
    def base_solution(self, x):
        return self._exp(-x)*self.__cos(5)
    
    def solution_in_domain(self, x):
        return 1/(x+1)*self.__cos(1/(x+1/5))

    def __residual(self, x):
        return (24*self.__cos(1/(1/5+x)))/(1+x)**5-(2*self.__cos(1/(1/5+x)))/(1+x)**3-(24*self.__sin(1/(1/5+x)))/((1/5+x)**2*(1+x)**4)+(2*self.__sin(1/(1/5+x)))/((1/5+x)**2*(1+x)**2)+(self.__cos(1/(1/5+x))/(1/5+x)**8-(36*self.__cos(1/(1/5+x)))/(1/5+x)**6+(12*self.__sin(1/(1/5+x)))/(1/5+x)**7-(24*self.__sin(1/(1/5+x)))/(1/5+x)**5)/(1+x)-(4*((6*self.__cos(1/(1/5+x)))/(1/5+x)**5-self.__sin(1/(1/5+x))/(1/5+x)**6+(6*self.__sin(1/(1/5+x)))/(1/5+x)**4))/(1+x)**2+(12*(-(self.__cos(1/(1/5+x))/(1/5+x)**4)-(2*self.__sin(1/(1/5+x)))/(1/5+x)**3))/(1+x)**3-(-(self.__cos(1/(1/5+x))/(1/5+x)**4)-(2*self.__sin(1/(1/5+x)))/(1/5+x)**3)/(1+x)-self._sinh(self.__cos(1/(1/5+x))/(1+x))/(1+self._sinh(self.__cos(1/(1/5+x))/(2*(1+x)))**2)

    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        return res
    def right_hand_side(self, x, u, q):
        num = (self._sinh(u) - self.__g_func(x)*0.5*self.__mu*self._exp(u) )*self._exp(-q*x)
        din = (1 + 2.0*self.__gamma*self._sinh(u/2.0)**2)
        res = num/din
        return res + self.__residual(x)*self._exp(-q*x)
    def right_hand_side_linearization(self, x, u, q):
        num = self._exp(-q*x)*(2*(self.__gamma + self._cosh(u) - self.__gamma*self._cosh(u)) + (self._exp(u)*(-1 + self.__gamma) - self.__gamma)*self.__mu*self.__g_func(x))
        din = 2*(1 - self.__gamma + self.__gamma*self._cosh(u))**2
        res = num/din
        return res



class nonlinear_problem_6(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "nonlinear_problem_6_"
        self.__sigma = 1.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__u0 = 1
        self.__u0xxx = 0

        self._bondaries = [[self.__u0, None, None, None, None],[0,None,None,None,None]]
        self._operator = [None, None, -1, None, 1]


    def get_name(self):
        return "$-u_{xx}+u_{xxxx}=f(x,u)-g(x)$, $u= exp(-x^2) cos(x)$"        
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain
    def __g_func(self, x):
        return 1.0/(self._sqrt(2*self._pi))*self._exp(-(1/2)*((x)/self.__sigma)**2 ) 
    def base_solution(self, x):
        return self._exp(-x**2)
    
    def solution_in_domain(self, x):
        return self._exp(-x**2)*self.__cos(x)

    def __residual(self, x):
        return 2*self._exp(-x**2)*self.__cos(x)-7*(-2*self._exp(-x**2)+4*self._exp(-x**2)*x**2)*self.__cos(x)+(12*self._exp(-x**2)-48*self._exp(-x**2)*x**2+16*self._exp(-x**2)*x**4)*self.__cos(x)-12*self._exp(-x**2)*x*self.__sin(x)-4*(12*self._exp(-x**2)*x-8*self._exp(-x**2)*x**3)*self.__sin(x)-self._sinh(self._exp(-x**2)*self.__cos(x))/(1+self._sinh(1/2*self._exp(-x**2)*self.__cos(x))**2)

    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        return res
    def right_hand_side(self, x, u, q):
        num = (self._sinh(u) - self.__g_func(x)*0.5*self.__mu*self._exp(u) )*self._exp(-q*x)
        din = (1 + 2.0*self.__gamma*self._sinh(u/2.0)**2)
        res = num/din
        return res + self.__residual(x)*self._exp(-q*x)
    def right_hand_side_linearization(self, x, u, q):
        num = self._exp(-q*x)*(2*(self.__gamma + self._cosh(u) - self.__gamma*self._cosh(u)) + (self._exp(u)*(-1 + self.__gamma) - self.__gamma)*self.__mu*self.__g_func(x))
        din = 2*(1 - self.__gamma + self.__gamma*self._cosh(u))**2
        res = num/din
        return res





class nonlinear_problem_7(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "nonlinear_problem_7_"
        self.__sigma = 1.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__u0 = self.__cos(1)
        self.__u0xxx = 0

        self._bondaries = [[self.__u0, None, None, None, None],[0,None,None,None,None]]
        self._operator = [None, None, -1, None, 1]


    def get_name(self):
        return "$-u_{xx}+u_{xxxx}=f(x,u)-g(x)$, $u= exp(-x) cos(1/(x+1))$"        
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain
    def __g_func(self, x):
        return 1.0/(self._sqrt(2*self._pi))*self._exp(-(1/2)*((x)/self.__sigma)**2 ) 
    def base_solution(self, x):
        return 1/(x+1)*self.__cos(1)
    
    def solution_in_domain(self, x):
        return self._exp(-x)*self.__cos(1/(x+1))

    def __residual(self, x):
        cs = self.__cos(1/(x+1))
        sn = self.__sin(1/(x+1))
        ecs = self._exp(-x)*cs
        esn = self._exp(-x)*sn
    
        num = self._sinh(ecs)
        din = 1 + self._sinh(ecs/2)**2

        p1 = 2*esn/((1+x)**2)
        p2 = self._exp(-x)*(cs/((1+x)**8)-36*cs/((1+x)**6)+12*sn/((1+x)**7)-24*sn/((1+x)**5))

        p3 = 4*self._exp(-x)*(6*cs/((1+x)**5)-sn/((1+x)**6)+6*sn/((1+x)**4))
        p4 = 5*self._exp(-x)*(-cs/((1+x)**4)-2*sn/((1+x)**3))
        return -p1+p2-p3+p4-num/din

    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        return res
    def right_hand_side(self, x, u, q):
        num = (self._sinh(u) - self.__g_func(x)*0.5*self.__mu*self._exp(u) )*self._exp(-q*x)
        din = (1 + 2.0*self.__gamma*self._sinh(u/2.0)**2)
        res = num/din
        return res + self.__residual(x)*self._exp(-q*x)
    def right_hand_side_linearization(self, x, u, q):
        num = self._exp(-q*x)*(2*(self.__gamma + self._cosh(u) - self.__gamma*self._cosh(u)) + (self._exp(u)*(-1 + self.__gamma) - self.__gamma)*self.__mu*self.__g_func(x))
        din = 2*(1 - self.__gamma + self.__gamma*self._cosh(u))**2
        res = num/din
        return res








