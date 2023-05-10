from spectral1d import collocation_discretization, solve_nonlinear_problem
from matplotlib import pyplot as plt
import numpy as np

class linear_manufactured_solutions_solver(object):
    def __init__(self, default_folder = "./", show_figures = True):
        self.__default_folder = default_folder
        self._file_name_prefix = "resutls_"
        self._L_all = [1]
        self._N_all = [10]
        self._basis_domain = [0,np.pi]
        self._bondaries = [[1,None,None,None,None],[0,None,None,None,None]]
        self.__operator = [None, None, -1, None, 1]
        self._h_all_1 = np.divide((self._basis_domain[1] - self._basis_domain[0]), self._N_all)
        self.__c_c_all_L_N = []
        self.__L2_all_L_N = []
        self._problem_name = ""
        self.__show_figures = show_figures

    def solve(self, problem):
        self._file_name_prefix = problem.file_name_prefix()
        self._L_all = problem.get_L()
        self._N_all = problem.get_N()
        self._domain = problem.get_domain()
        self._basis_domain = problem.get_basis_domain()
        self._bondaries = problem.get_boundary_conditions()
        self.__operator = problem.operator()
        self._h_all = np.divide((self._basis_domain[1] - self._basis_domain[0]), self._N_all)  
        self._problem_name = problem.get_name()
        self._problem = problem
        
        self._solve_all_problems( self.__solve_one_problem )

        if self._file_name_prefix != None:
            self.save_results()

    def __solve_one_problem(self, N, L):
        col = collocation_discretization(N, self._domain)
        col.set_boundary_conditions(self._bondaries)
        col.set_mapping_parameter(L)

        def rhs_l(x, t):
            return self._problem.rhs_in_domain(x)

        b_c = col.linear_functional( rhs_l )
        b_c = col.linear_functional_boundary(b_c)

        alpha = self.__operator
        S_c  = col.bilinear_form(col.psi, 0)
        if alpha[0] != None:
            S_c  = S_c + col.bilinear_form(col.psi, alpha[0])
        if alpha[1] != None:
            S_c = S_c + col.bilinear_form(col.dpsi, alpha[1])
        if alpha[2] != None:
            S_c = S_c + col.bilinear_form(col.ddpsi, alpha[2])
        if alpha[3] != None:
            S_c = S_c + col.bilinear_form(col.dddpsi, alpha[3])
        if alpha[4] != None:
            S_c = S_c + col.bilinear_form(col.ddddpsi, alpha[4])
        S_c = col.set_boundary_bilinear_form_tau_method(S_c) 
        c_c = col.solve(S_c, b_c)

        L2_err = col.L2_error( c_c, self._problem.solution_in_domain)
        return L2_err, c_c

    def _solve_all_problems(self, poblem):
        self.__c_c_all_L_N = []
        self.__L2_all_L_N = []
        for L in self._L_all:
            L2_N = []
            c_c_N = []
            for N in self._N_all:
                L2_err, c_c = poblem(N, L);
                L2_N.append(L2_err)
                c_c_N.append(c_c)

            self.__c_c_all_L_N.append(c_c_N)
            self.__L2_all_L_N.append(L2_N)


    def save_results(self):
        self.__figures_L2_error()
        self.__figures_coefficients()
        self.__figures_solutions()

    def __figures_L2_error(self):
        file_path_name = self.__default_folder+"/"+self._file_name_prefix+"L2_vs_L.pdf"
        title = "$L_2$ error for "+self._problem_name
        legend = []
        h_all = self._h_all
        N_all = self._N_all
        L2_all = self.__L2_all_L_N

        k1 = L2_all[0][0]/h_all[0]
        k2 = L2_all[0][0]/h_all[0]**2
        k4 = L2_all[0][0]/h_all[0]**4
        k8 = L2_all[0][0]/h_all[0]**8
        print(N_all)
        print(h_all)
        plt.loglog(N_all,np.multiply(h_all,k1), linewidth=0.5)
        legend.append("1st order")
        plt.loglog(N_all,k2*np.power(h_all,2), linewidth=0.5 )
        legend.append("2nd order")
        plt.loglog(N_all,k4*np.power(h_all,4), linewidth=0.5 )
        legend.append("4th order")
        plt.loglog(N_all,k8*np.power(h_all,8), linewidth=0.5 )
        legend.append("8th order")
        for L, L2_N in zip(self._L_all,L2_all):
            k_l = L2_all[0][0]/L2_N[0]
            plt.loglog(N_all,np.multiply(L2_N,k_l),'*')
            legend.append("L = %.01f"%L)
        plt.legend(legend)
        plt.title(title)
        plt.grid()
        plt.savefig(file_path_name)
        if self.__show_figures:
            plt.show()
        plt.clf()            

    def __figures_coefficients(self):
        file_path_name = self.__default_folder+"/"+self._file_name_prefix+"N_coeffs.pdf"
        title = "$|\hat{u}}(N)$ decay for "+self._problem_name
        legend = []
        N_all = self._N_all
        L_all = self._L_all
        c_c_all = self.__c_c_all_L_N

        c_c_N = c_c_all[2]
        for c_c_ ,N in zip(reversed(c_c_N), reversed(N_all)):
            plt.semilogy( np.abs(c_c_), '.')
            legend.append("N = %i"%N)

        plt.legend(legend)
        plt.title(title)
        plt.grid()
        plt.savefig(file_path_name)
        if self.__show_figures:
            plt.show()
        plt.clf()

        file_path_name = self.__default_folder+"/"+self._file_name_prefix+"L_coeffs.pdf"
        title = "$|\hat{u}}(L)$ decay for "+self._problem_name
        legend = []
        c_c_L = []
        for c_c_ in c_c_all:
            c_c_L.append(c_c_[len(c_c_)-2])

        for L, c_c_ in zip(L_all, c_c_L):
            plt.semilogy( c_c_, '.')
            legend.append("L = %0.1f"%L)

        plt.legend(legend)
        plt.title(title)
        plt.grid()
        plt.savefig(file_path_name)
        if self.__show_figures:
            plt.show()
        plt.clf()


    def __figures_solutions(self):
        file_path_name = self.__default_folder+"/"+self._file_name_prefix+"solutions.pdf"
        title = "solutions for "+self._problem_name        
        legend = []
        t = np.arange(0, np.pi, 0.005)
        N_all = self._N_all
        L_all = self._L_all
        c_c_all = self.__c_c_all_L_N
        
        for L,c_c_N in zip(L_all, c_c_all):
            for N, c_c in zip(N_all, c_c_N):
                if N == N_all[-1]:
                    col = collocation_discretization(N, self._domain)
                    col.set_mapping_parameter(L)
                    f_c = col.solution_in_basis(c_c, t);
                    # plt.semilogy(t, np.abs(sol(col.arg_from_basis_to_domain(t))-f_c),linewidth = 0.5)
                    plt.plot(t, f_c,'*', label='_nolegend_')
                    
            plt.plot(t, self._problem.solution_in_domain( col.arg_from_basis_to_domain(t) ) ,linewidth = 1.0)
            legend.append("L = %0.1f"%L)

        plt.legend(legend)
        plt.title(title)    
        plt.grid()
        plt.savefig(file_path_name)        
        if self.__show_figures:
            plt.show()
        plt.clf()



    def get_all_coefficients(self):
        return self.__c_c_all_L_N
    
    def get_all_L2_norms(self):
        return self.__L2_all_L_N

class nonlinear_manufactured_solutions_solver(linear_manufactured_solutions_solver):
    def __init__(self, default_folder = "./", show_figures = True):
        super().__init__(default_folder, show_figures)
        self._tolerance = 1.0e-7
        self._use_method = "newton"
        self._visualize = False

    def solve(self, problem):
        self._file_name_prefix = problem.file_name_prefix()
        self._L_all = problem.get_L()
        self._N_all = problem.get_N()
        self._domain = problem.get_domain()
        self._basis_domain = problem.get_basis_domain()
        self._bondaries = problem.get_boundary_conditions()
        self._h_all = np.divide((self._basis_domain[1] - self._basis_domain[0]), self._N_all)  
        self._problem_name = problem.get_name()
        self._tolerance = problem.get_tolerance()
        self._visualize = problem.visualize()
        self._use_method = problem.nonlinear_method()
        self._problem = problem
        
        self._solve_all_problems( self.__solve_one_problem )
        if self._file_name_prefix != None:
            self.save_results()

    def __solve_one_problem(self, N, L):
        solver = solve_nonlinear_problem(N, self._domain, tolerance = self._tolerance, use_method = self._use_method, visualize = self._visualize)
        solver.set_mapping_parameter(L)
        solver.set_problem(self._problem)
        c_c = solver.solve_problem()
        L2_err = solver.L2_error( c_c, self._problem.solution_in_domain)
        return L2_err, c_c    
    