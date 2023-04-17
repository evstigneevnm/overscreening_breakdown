from spectral1d import collocation_discretization, galerkin_discretization
from matplotlib import pyplot as plt
import numpy as np

def main():
    N = 31
    col = collocation_discretization(N, [0, np.inf] )
    gal = galerkin_discretization(N, [0, np.inf])
    col.set_dirichlet_boundary_conditoins_on_domain([1,0])
    gal.set_dirichlet_boundary_conditoins_on_domain([1,0])
    
    # print(col.mass_matrix())
    # print(gal.mass_matrix())

    def func(x):
        return np.exp(-x)

    b_c,y_c = col.rhs_expand(func );
    b_g = gal.rhs_expand(func );
    # print(b_c)
    # print(b_g)

    S_c = col.bilinear_form(col.ddpsi, 1 )
    S_c = S_c + col.bilinear_form(col.ddddpsi, -5)
    S_c = col.set_boundary_bilinear_form_tau_method(S_c)
    S_g = gal.bilinear_form(gal.ddpsi, gal.psi, 1)
    S_g = S_g + gal.bilinear_form(gal.dddpsi, gal.dpsi, 5)
    S_g = gal.set_boundary_bilinear_form_tau_method(S_g)

    c_c = col.solve(S_c, b_c)
    print(c_c)
    c_g = gal.solve(S_g, b_g)
    print(c_g)

    t = np.arange(0, np.pi, 0.01)
    f_c = col.solution_in_basis(c_c, t);
    f_g = gal.solution_in_basis(c_g, t);
    plt.plot(t, func(col.arg_from_basis_to_domain(t)))
    plt.plot(t, f_c,'.')
    plt.plot(t, f_g,'.')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()