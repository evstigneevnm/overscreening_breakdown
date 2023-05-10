from spectral1d import collocation_discretization
from matplotlib import pyplot as plt
import numpy as np

def main():
    N = 10
    col = collocation_discretization(N, [0, np.inf])
    col.set_boundary_conditions([[1,None,None,None,None],[0,None,None,None,None]])
    col.set_mapping_parameter(1/2)
    def rhs(x, t):
        return -2*np.exp(-x**2)+4*np.exp(-x**2)*x**2-10*(12*np.exp(-x**2)-48*np.exp(-x**2)*x**2+16*np.exp(-x**2)*x**4)
    def sol(x):
        return np.exp(-x**2)

    # M_c = col.mass_matrix()
    # M_c = col.set_boundary_bilinear_form_tau_method(M_c)
    # print("M:")
    # print(M_c)
    b_c = col.linear_functional(rhs)
    b_c = col.linear_functional_boundary(b_c)
    print("b:")
    print(b_c)
    S_c = col.bilinear_form(col.ddpsi, 1 )
    S_c = S_c + col.bilinear_form(col.ddddpsi, -10)
    S_c = col.set_boundary_bilinear_form_tau_method(S_c)
    # print("S:")
    # print(S_c)
    
    c_c = col.solve(S_c, b_c)
    print("c:")
    print(c_c)
    plt.semilogy(np.abs(c_c),'.')
    plt.grid()
    plt.show()
    t = np.arange(0, np.pi, 0.01)
    f_c = col.solution_in_basis(c_c, t);
    plt.plot(t, f_c,'*')
    plt.plot(t, sol(col.arg_from_basis_to_domain(t) ) )
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()