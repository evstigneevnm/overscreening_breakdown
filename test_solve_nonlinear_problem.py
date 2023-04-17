from matplotlib import pyplot as plt
import numpy as np
from spectral1d import solve_nonlinear_problem
from overscreening_breakdown import overscreening_breakdown


def main():
    problem = overscreening_breakdown()
    MM = [50,100,150]
    c_dict = {}
    for j in MM:
        N = j
        print("domain = ", N)
        solver = solve_nonlinear_problem(N, [0,np.inf], tolerance = 1.0e-7, use_method = "newton", visualize = False)
        solver.set_mapping_parameter(4)
        solver.set_problem(problem)
        # solver.reset_base_solution()
        # solver.set_base_solution("solution.pickle")
        c = solver.solve_problem()
        c_dict[j] = c
        # solver.save_solution("solution.pickle", c)
    
    # x = np.arange(0, 30, 0.1)
    # sol = solver.obtain_solution_in_domain(c, x)
    t = np.arange(0, np.pi, 0.01)
    x = np.arange(0, 100, 0.1)
    for j in MM:
        c_l = c_dict[j]
        c_norm = np.linalg.norm(c_l)
        print(c_norm)
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


    solver.save_solution("solution.pickle", c)

if __name__ == '__main__':
    main()