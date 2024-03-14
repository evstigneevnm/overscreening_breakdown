from matplotlib import pyplot as plt
import numpy as np
from spectral1d import solve_nonlinear_problem
from nonlinear_problem_on_segment import nonlinear_problem_on_segment


def main():
    problem = nonlinear_problem_on_segment()
    MM = [10]
    use_mpmath = False
    use_adjoint_opimization = True
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
        solver = solve_nonlinear_problem(N, problem.get_domain(), tolerance = tolerance, use_method = "newton", visualize = False, use_mpmath = use_mpmath, prec = fp_prec, use_adjoint_opimization = use_adjoint_opimization)
        solver.set_problem(problem)
        c = solver.solve_problem(c0 = c)
        c_dict[j] = c
        c_norm = np.linalg.norm(c)
        print("solution norm = {:.12e}".format(np.float128(c_norm)) )
        solver.save_solution("solution_{:d}.pickle".format(j), c)

    t = np.arange(0, np.pi, 0.01)
    x = np.arange(problem.get_domain()[0], problem.get_domain()[1], (problem.get_domain()[1]-problem.get_domain()[0])/500)
    for j in MM:
        c_l = c_dict[j]
        c_norm = np.linalg.norm(c_l)
        print("solution norm = {:.12e}".format(np.float128(c_norm)) )
        if c_norm<1000:
            sol = solver.obtain_solution_in_basis(c_l, t)
            plt.plot(t, sol, label=j)
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
    solver = solve_nonlinear_problem(N, problem.get_domain(), tolerance = 1.0e-30, use_method = "newton", visualize = False, use_mpmath = True)
    solver.set_problem(problem)    
    solver.set_base_solution("solution_{:d}.pickle".format(N))
    c = solver.solve_problem()
    print("compare the results")
    c_l = c_dict[MM[-1]]
    sol = solver.obtain_solution_in_basis(c_l, t)
    plt.plot(t, sol, 'o', label="original")
    sol = solver.obtain_solution_in_basis(c, t)
    plt.plot(t, sol, label="new")
    plt.legend()
    plt.show()    
    sol = solver.obtain_solution_in_domain(c_l, x)
    plt.plot(x, sol, 'o', label="original")
    sol = solver.obtain_solution_in_domain(c, x)
    plt.plot(x, sol, label="new")
    plt.legend()
    plt.show()      

if __name__ == '__main__':
    main()