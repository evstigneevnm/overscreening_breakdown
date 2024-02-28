from spectral1d import collocation_discretization
from matplotlib import pyplot as plt
import numpy as np
import mpmath as mp
import time


def func_mp(x):
    vexp = np.vectorize(mp.exp)
    return vexp(-x)
def func_np(x):
    return np.exp(-x)

def calculate_solve_function(N, L, use_mpmath, bc_left, bc_right):
    
    def rhs(x, t):
        if use_mpmath:
            return func_mp(x)
        else:
            return func_np(x)

    col = collocation_discretization(N = N, domain = [0, np.inf], L = L, use_mpmath = use_mpmath)
    col.set_boundary_conditions([bc_left,bc_right])
    S_c = col.bilinear_form(col.ddpsi, 1)
    b_c = col.linear_functional(rhs)
    b_c = col.linear_functional_boundary(b_c)    
    S_c = S_c + col.bilinear_form(col.ddddpsi, -100)
    S_c = col.set_boundary_bilinear_form_tau_method(S_c)
    c_c = col.solve(S_c, b_c)
    t = np.arange(0, np.pi, 0.01)
    f_c = col.solution_in_basis(c_c, t)
    bf_c = col.solution_in_basis(b_c, t)
    return b_c, c_c, t, f_c, bf_c

def main():

    N = 100
    L = 1/2

    start_time_mp = time.time()
    b_mp, c_mp, t, f_mp, bf_mp = calculate_solve_function(N, L, True, [1,None,None,None,None], [0,None,None,None,None])
    end_time_mp = time.time()
    time_mp = end_time_mp-start_time_mp

    start_time_np = time.time()
    b_np, c_np, t, f_np, bf_np = calculate_solve_function(N, L, False, [1,None,None,None,None], [0,None,None,None,None])
    end_time_np = time.time()
    time_np = end_time_np-start_time_np

    print("execution time for mpmath = ", time_mp, "sec.")
    print("execution time for np = ", time_np, "sec.")
    print("ratio mpmath/np = ", time_mp/time_np)

    # plt.plot(t, func_np(col_np.arg_from_basis_to_domain(t)))
    plt.plot(t, f_mp,'o')
    plt.plot(t, f_np,'.')
    # plt.plot(t, bf_np - bf_mp,'.')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
