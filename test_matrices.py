from spectral1d import collocation_discretization
from matplotlib import pyplot as plt
import numpy as np
import mpmath as mp
import time

def rhs_np(x, t):
    return -2*np.exp(-x**2)+4*np.exp(-x**2)*x**2-10*(12*np.exp(-x**2)-48*np.exp(-x**2)*x**2+16*np.exp(-x**2)*x**4)
def rhs_mp(x, t):
    vexp = np.vectorize(mp.exp)
    return -2*vexp(-x**2)+4*vexp(-x**2)*x**2-10*(12*vexp(-x**2)-48*vexp(-x**2)*x**2+16*vexp(-x**2)*x**4)

def sol(x):
    return np.exp(-x**2)

def test_solve(N, L, use_mpmath, prec = 16): #log_10(2^53) \sim 16
    def rhs_wrap(x, t):
        if use_mpmath:
            return rhs_mp(x, t)
        else:
            return rhs_np(x, t)

    col = collocation_discretization(N = N, domain = [0, np.inf], L = L, use_mpmath = use_mpmath, prec = prec)
    col.set_boundary_conditions([[1,None,None,None,None],[0,None,None,None,None]])
    # col.set_mapping_parameter(L)
    # M_c = col.mass_matrix()
    # M_c = col.set_boundary_bilinear_form_tau_method(M_c)
    # print("M:")
    # print(M_c)
    # print("S:")
    # print(S_c)
    b_c = col.linear_functional(rhs_wrap)
    b_c = col.linear_functional_boundary(b_c)
    S_c = col.bilinear_form(col.ddpsi, 1 )
    S_c = S_c + col.bilinear_form(col.ddddpsi, -10)
    S_c = col.set_boundary_bilinear_form_tau_method(S_c)
    c_c = col.solve(S_c, b_c)
    t = np.arange(0, np.pi, 0.01)
    x = col.arg_from_basis_to_domain(t)
    f_c = col.solution_in_basis(c_c, t);
    

    return b_c, c_c, f_c, t, x

def main():
    N = 400
    L = 1/2
    
    start_time_mp = time.time()
    b_c_mp, c_c_mp, f_c_mp, t_mp, x_mp = test_solve(N, L, True, 100)
    end_time_mp = time.time()
    time_mp = end_time_mp-start_time_mp
    start_time_np = time.time()    
    b_c_np, c_c_np, f_c_np, t_np, x_np = test_solve(N, L, False)
    end_time_np = time.time()
    time_np = end_time_np-start_time_np    

    print("b:")
    print(b_c_mp)
    print(b_c_np)
    print("c:")
    print(c_c_mp)
    print(c_c_np)

    print("execution time for mpmath = ", time_mp, "sec.")
    print("execution time for np = ", time_np, "sec.")
    print("ratio mpmath/np = ", time_mp/time_np)

    plt.semilogy(np.abs(c_c_mp),'o')
    plt.semilogy(np.abs(c_c_np),'.')
    plt.legend(["mpmath", "np"])
    plt.grid()
    plt.show()

    plt.plot(t_mp, f_c_mp,'o')
    plt.plot(t_mp, f_c_np,'.')
    plt.plot(t_mp, sol(x_np) )
    plt.legend(["mpmath", "np", "exact"])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()