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

def calculate_solve_function(N, L, domain, use_mpmath, bc_left, bc_right):
    
    def rhs(x, t):
        if use_mpmath:
            return func_mp(x)
        else:
            return func_np(x)

    col = collocation_discretization(N = N, domain = domain, L = L, use_mpmath = use_mpmath)
    col.set_boundary_conditions([bc_left, bc_right])
    S_c = col.bilinear_form(col.ddpsi, 1)
    b_c = col.linear_functional(rhs)
    b_c = col.linear_functional_boundary(b_c)    
    S_c = S_c + col.bilinear_form(col.ddddpsi, -100)
    # print(np.shape(S_c))
    # print(S_c)
    S_c = col.set_boundary_bilinear_form_tau_method(S_c)
    # print(np.shape(S_c))
    # print(S_c)
    c_c = col.solve(S_c, b_c)
    t = np.arange(0, np.pi, 0.01)
    if domain[1] == np.inf:
        x_max = 100
        h = 0.1;
    else:
        x_max = domain[1]
        h = (x_max - domain[0])/1000.0

    x = np.arange(domain[0], x_max, h)
    f_c = col.solution_in_basis(c_c, t)
    fx_c = col.solution_in_domain(c_c, x)
    bf_c = col.solution_in_basis(b_c, t)
    return b_c, S_c, c_c, t, f_c, bf_c, x, fx_c

def main():

    N = 25
    L = 1/2

    #checking on semi-infinite domain
    print("=== checking on semi-infinite domain ===")
    domain = [0, np.inf]
    start_time_mp = time.time()
    b_mp, S_c_mp, c_mp, t, f_mp, bf_mp, x, fx_mp  = calculate_solve_function(N, L, domain, False, [1,None,None,0,None], [None,None,None,None,None])
    end_time_mp = time.time()
    time_mp = end_time_mp-start_time_mp

    start_time_np = time.time()
    b_np, S_c_np, c_np, t, f_np, bf_np, x, fx_np  = calculate_solve_function(N, L, domain, False, [1,None,None,0,None], [None,None,None,None,None])
    end_time_np = time.time()
    time_np = end_time_np-start_time_np

    NP = []
    MP = []
    d = []
    for a in S_c_np:
        for b in a:
            NP.append(b)
    if str(type(S_c_mp)) == "<class 'numpy.ndarray'>":
        for a in S_c_mp:
            for b in a:
                MP.append(b)
    else:
        for a in S_c_mp:
            MP.append(a)

    for a,b in zip(MP, NP):
        d.append(np.abs(np.float64(a-b)))
    print("matrix difference: ", np.linalg.norm(d))

    NP = []
    MP = []
    d = []    
    for n,m in zip(b_np, b_mp):
        NP.append(n)
        MP.append(m)
    for a,b in zip(MP, NP):
        d.append(np.abs(np.float64(a-b)))
    print("expansion coeffs difference: ",np.linalg.norm(d)) 

    NP = []
    MP = []
    d = []    
    for n,m in zip(c_np, c_mp):
        NP.append(n)
        MP.append(m)
    for a,b in zip(MP, NP):
        d = np.abs(np.float64(a-b))
    print("solution coeffs difference: ",np.linalg.norm(d))    

    NP = []
    MP = []
    d = []    
    for n,m in zip(f_np, f_mp):
        NP.append(n)
        MP.append(m)
    for a,b in zip(MP, NP):
        d.append(np.abs(np.float64(a-b)))
    print("solution pointwise difference: ",np.linalg.norm(d))


    print("execution time for mpmath = ", time_mp, "sec.")
    print("execution time for np = ", time_np, "sec.")
    print("ratio mpmath/np = ", time_mp/time_np)


    plt.plot(t, f_mp,'o')
    plt.plot(t, f_np,'.')
    plt.legend(["mp", "np"])
    plt.grid()
    plt.show()

    plt.plot(x, fx_mp,'o')
    plt.plot(x, fx_np,'.')
    plt.legend(["mp", "np"])
    plt.grid()
    plt.show()

    plt.plot(t, bf_np - bf_mp, '.')
    plt.grid()
    plt.show()

    #checking on segment domain
    print("=== checking on segment domain ===")
    domain = [-7, 4]
    start_time_mp = time.time()
    b_mp, S_c_mp, c_mp, t, f_mp, bf_mp, x, fx_mp = calculate_solve_function(N, L, domain, False, [5,-5,None,None,None], [5,-5,None,None,None])
    end_time_mp = time.time()
    time_mp = end_time_mp-start_time_mp

    start_time_np = time.time()
    b_np, S_c_np, c_np, t, f_np, bf_np, x, fx_np = calculate_solve_function(N, L, domain, False, [5,-5,None,None,None], [5,-5,None,None,None])
    end_time_np = time.time()
    time_np = end_time_np-start_time_np    


    NP = []
    MP = []
    d = []    
    for a in S_c_np:
        for b in a:
            NP.append(b)

    if str(type(S_c_mp)) == "<class 'numpy.ndarray'>":
        for a in S_c_mp:
            for b in a:
                MP.append(b)
    else:
        for a in S_c_mp:
            MP.append(a)

    for a,b in zip(MP, NP):
        d.append(np.abs(np.float64(a-b)))
    print("matrix difference: ", np.linalg.norm(d))

    NP = []
    MP = []
    d = []    
    for n,m in zip(b_np, b_mp):
        NP.append(n)
        MP.append(m)
    for a,b in zip(MP, NP):
        d.append(np.abs(np.float64(a-b)))
    print("expansion coeffs difference: ",np.linalg.norm(d)) 

    d = []
    for a,b in zip(c_np, c_mp):
        d.append(np.abs(np.float64(a-b)))
    print("solution coeffs difference: ",np.linalg.norm(d)) 



    NP = []
    MP = []
    d = []    
    for n,m in zip(f_np, f_mp):
        NP.append(n)
        MP.append(m)
    for a,b in zip(MP, NP):
        d = np.abs(np.float64(a-b))
    print("solution pointwise difference: ",np.linalg.norm(d)) 


    print("execution time for mpmath = ", time_mp, "sec.")
    print("execution time for np = ", time_np, "sec.")
    print("ratio mpmath/np = ", time_mp/time_np)

    plt.plot(t, f_mp,'o')
    plt.plot(t, f_np,'.')
    plt.legend(["mp", "np"])
    plt.grid()
    plt.show()

    plt.plot(x, fx_mp,'o')
    plt.plot(x, fx_np,'.')
    plt.legend(["mp", "np"])
    plt.grid()
    plt.show()    

    plt.plot(t, bf_np - bf_mp, '.')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
