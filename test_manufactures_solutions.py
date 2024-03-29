from manufactured_solutions_solver import linear_manufactured_solutions_solver, nonlinear_manufactured_solutions_solver
from manufactured_problems import problem_1, problem_2, problem_3, problem_4, problem_5, problem_6, problem_7, problem_8
from manufactured_problems import nonlinear_problem_1, nonlinear_problem_2, nonlinear_problem_3, nonlinear_problem_4, nonlinear_problem_5, nonlinear_problem_6

def main():
    solver = linear_manufactured_solutions_solver(default_folder = "./figures", show_figures = False)
    pr1 = problem_1()
    solver.solve(pr1)
    pr2 = problem_2()
    solver.solve(pr2)
    pr3 = problem_3()
    solver.solve(pr3)    
    pr4 = problem_4()
    solver.solve(pr4)    
    pr5 = problem_5()
    solver.solve(pr5)
    pr6 = problem_6()
    solver.solve(pr6)

    # pr7 = problem_7()
    # solver.solve(pr7)
    # pr8 = problem_8()
    # solver.solve(pr8)

    nonlin_solver = nonlinear_manufactured_solutions_solver(default_folder = "./figures", show_figures = False)
    nonlin_pr1 = nonlinear_problem_1()
    nonlin_solver.solve(nonlin_pr1)
    nonlin_pr2 = nonlinear_problem_2()
    nonlin_solver.solve(nonlin_pr2)
    nonlin_pr3 = nonlinear_problem_3()
    nonlin_solver.solve(nonlin_pr3)
    nonlin_pr4 = nonlinear_problem_4()
    nonlin_solver.solve(nonlin_pr4)
    nonlin_pr5 = nonlinear_problem_5()
    nonlin_solver.solve(nonlin_pr5)
    nonlin_pr6 = nonlinear_problem_6()
    nonlin_solver.solve(nonlin_pr6)


if __name__ == '__main__':
    main()