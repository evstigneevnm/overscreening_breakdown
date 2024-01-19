from spectral1d import collocation_discretization
from matplotlib import pyplot as plt
import numpy as np

def main():
    N = 31
    col = collocation_discretization(N, [0, np.inf] )
    col.set_boundary_conditions([[1,None,None,-1,None],[0,None,None,None,None]])
    
    # print(col.mass_matrix())

    def func(x):
        return np.exp(-x)

    b_c = col.function_expand(func );
    # print(b_c)


    S_c = col.bilinear_form(col.ddpsi, 1 )
    S_c = S_c + col.bilinear_form(col.ddddpsi, -5)
    S_c = col.set_boundary_bilinear_form_tau_method(S_c)

    c_c = col.solve(S_c, b_c)
    print(c_c)

    t = np.arange(0, np.pi, 0.01)
    f_c = col.solution_in_basis(c_c, t);
    plt.plot(t, func(col.arg_from_basis_to_domain(t)))
    plt.plot(t, f_c,'.')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
