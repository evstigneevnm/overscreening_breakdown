from matplotlib import pyplot as plt
import numpy as np
from spectral1d import solve_nonlinear_problem
from overscreening_breakdown import overscreening_breakdown


def main():
    problem = overscreening_breakdown()
    L = 7
    MM = [30, 50, 70, 100]
    use_mpmath = True
    fp_prec = 100
    if use_mpmath:
        tolerance = 1.0e-30
    else:
        tolerance = 1.0e-8

    c_dict = {}
    c = None
    for j in MM:
        N = j
        print("domain = ", N)
        solver = solve_nonlinear_problem(N, [0,np.inf], tolerance = tolerance, use_method = "newton", visualize = False, use_mpmath = use_mpmath, prec = fp_prec)
        solver.set_mapping_parameter(L)
        solver.set_problem(problem)
        # solver.reset_base_solution()
        # try:
        #     solver.set_base_solution("solution.pickle")
        # except:
        #     print("solution not loaded")
        c = solver.solve_problem(c0 = c)
        c_dict[j] = c
        c_norm = np.linalg.norm(c)
        print("solution norm = {:.12e}".format(np.float128(c_norm)) )        
        # solver.save_solution("solution.pickle", c)
        solver.save_solution("solution_{:d}.pickle".format(j), c)

    t = np.arange(0, np.pi, 0.01)
    x = np.arange(0, 100, 0.1)
    for j in MM:
        c_l = c_dict[j]
        c_norm = np.linalg.norm(c_l)
        print("solution norm = {:.12e}".format(np.float128(c_norm)) )
        if c_norm<1000:
            sol = solver.obtain_solution_in_basis(c_l, t)
            # rhs = solver.obtain_rhs_in_basis(c_l, t)
            plt.plot(t, sol, label=j)
            # plt.plot(t, rhs)
    plt.legend()
    plt.show()
    for j in MM:
        c_l = c_dict[j]
        sol = solver.obtain_solution_in_domain(c_l, x)
        plt.plot(x, sol, label=j)
    plt.legend()
    plt.show()

    print("checking the quality of the last solved solution from file")
    N = MM[-1]
    solver = solve_nonlinear_problem(N, [0,np.inf], tolerance = 1.0e-30, use_method = "newton", visualize = False, use_mpmath = True)
    solver.set_mapping_parameter(L)
    solver.set_problem(problem)    
    solver.set_base_solution("solution_{:d}.pickle".format(N))
    c = solver.solve_problem()
    print("compare the results")
    c_l = c_dict[-1]
    sol = solver.obtain_solution_in_domain(c_l, x)
    plt.plot(x, sol, label="original")
    sol = solver.obtain_solution_in_domain(c, x)
    plt.plot(x, sol, label="new")
    plt.legend()
    plt.show()    

if __name__ == '__main__':
    main()