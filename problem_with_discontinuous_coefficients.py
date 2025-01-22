from manufactured_problems import basic_problem
import numpy as np

class problem_discontinuous_lhs_1(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_discontinuous_lhs_1_"
        self.__sigma = 1.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__u0 = 1
        self.__u0x = -1
        self.__u0xxx = 0
        self._bondaries = [[self.__u0, self.__u0x, None, None, None],[0,None,None,None,None]]
        self._operator = [None, None, 1, None, None]

    def info(self):
        return "test for the 2-nd derivative with manufactured solution exp(-x) and discontinuous coefficient if(x<1,1,10)."

    def get_name(self):
        return "$eps(x)u_{xx}=f(x,u)-g(x)$, $u=exp(-x)$"        
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain

    def solution_in_domain(self, x):
        return self._exp(-x)


    def __residual(self, x):
        lll = self.left_hand_side();
        func = np.vectorize(lll[2])
        ex = self._exp(-x)
        # -E^-x If[x < 1, 1, 10] + Sinh[E^-x]
        rrr = -ex*func(x) + self._sinh(ex)
        return rrr 

    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        def fun2(x):
            #it is assumed, that the function if only picewise-constant.
            if x<1:
                return 1
            else:
                return 10
        res[2] = fun2
        return res


    def right_hand_side(self, x, u, q):
        return (self._sinh(u) - self.__residual(x))*self._exp(-q*x)

    def right_hand_side_linearization(self, x, u, q):
        return self._cosh(u)*self._exp(-q*x)



class problem_discontinuous_lhs_2(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_discontinuous_lhs_2_"
        self.__sigma = 1.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__u0 = 1
        self.__u0x = -1
        self.__u0xxx = -1
        self._bondaries = [[self.__u0, None, None, self.__u0xxx, None],[0,None,None,None,None]]
        self._operator = [None, None, None, None, 1]

    def info(self):
        return "test for the 4-th derivative with manufactured solution exp(-x) and discontinuous coefficient if(x<1,1,10)."

    def get_name(self):
        return "$eps(x)u_{xxxx}=f(x,u)-g(x)$, $u=exp(-x)$"        
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain

    def solution_in_domain(self, x):
        return self._exp(-x)


    def __residual(self, x):
        lll = self.left_hand_side();
        func = np.vectorize(lll[4])
        ex = self._exp(-x)
        # -E^-x If[x < 1, 1, 2] + Sinh[E^-x]
        rrr = -ex*func(x) + self._sinh(ex)
        return rrr 

    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        def fun2(x):
            #it is assumed, that the function if only picewise-constant.
            if x<1:
                return 1
            else:
                return 10
        res[4] = fun2
        return res


    def right_hand_side(self, x, u, q):
        return (self._sinh(u) - self.__residual(x))*self._exp(-q*x)

    def right_hand_side_linearization(self, x, u, q):
        return self._cosh(u)*self._exp(-q*x)




class problem_discontinuous_lhs_3(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_discontinuous_lhs_3_"
        self.__sigma = 1.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__u0 = 1
        self.__u0x = 0
        self.__u0xxx = 0
        self._bondaries = [[self.__u0, None, None, None, None],[0,None,None,None,None]]
        self._operator = [None, None, 1, None, None]

    def info(self):
        return "test for the 2-nd derivative with and discontinuous coefficient if(x<1,1,1000)."

    def get_name(self):
        return "$eps(x)u_{xx}=sinh(u)$, $u=exp(-x)$"        
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain

    def solution_in_domain(self, x):
        return self._exp(-x)


    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        def fun2(x):
            #it is assumed, that the function if only picewise-constant.
            if x<1:
                return 1
            else:
                return 1000
        res[2] = fun2
        return res


    def right_hand_side(self, x, u, q):
        return (self._sinh(u))*self._exp(-q*x)

    def right_hand_side_linearization(self, x, u, q):
        return self._cosh(u)*self._exp(-q*x)


class problem_discontinuous_lhs_4(basic_problem):
    def __init__(self):
        super().__init__()
        self._file_name_prefix = "problem_discontinuous_lhs_4_"
        self.__sigma = 1.0
        self.__mu = 0.0
        self.__gamma = 1/2
        self.__u0 = 1
        self.__u0x = 0
        self.__u0xxx = 0
        self._bondaries = [[self.__u0, None, None, self.__u0xxx, None],[0,None,None,None,None]]
        self._operator = [None, None, None, None, 1]

    def info(self):
        return "test for the 4-nd derivative with and discontinuous coefficient if(x<1,1,1000)."

    def get_name(self):
        return "$eps(x)u_{xxxx}=sinh(u)$, $u=exp(-x)$"        
    def file_name_prefix(self):
        return self._file_name_prefix          
    def get_boundary_conditions(self):
        return self._bondaries           
    def get_domain(self):
        return self._domain

    def solution_in_domain(self, x):
        return self._exp(-x)


    #all linear parts are coded as funcitons of the solutoin u:
    # [cu, cu_x, cu_xx, cu_xxx, cu_xxxx]
    def left_hand_side(self):
        res = self._operator
        def fun2(x):
            #it is assumed, that the function if only picewise-constant.
            if x<1:
                return 1
            else:
                return 1000
        res[2] = fun2
        return res


    def right_hand_side(self, x, u, q):
        return (self._sinh(u))*self._exp(-q*x)

    def right_hand_side_linearization(self, x, u, q):
        return self._cosh(u)*self._exp(-q*x)
