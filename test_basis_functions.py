from spectral1d import basis_functions, basic_discretization
from matplotlib import pyplot as plt
import numpy as np
import mpmath as mp

def main():
    N = 10
    L = 4;
    bf_np = basis_functions(N = N, L = L)
    bf_mp = basis_functions(N = N, L = L, use_mpmath = True)

    print("testing numpy implementation...")
    bf_np.test()
    print("ok")
    print("testing mpmath implementation...")
    bf_mp.test()
    print("ok")

    bd_np_inf = basic_discretization(N = N, domain = [0, np.inf])
    bd_mp_inf = basic_discretization(N = N, domain = [0, mp.inf], L = L, use_mpmath = True)

    bd_np_seg = basic_discretization(N = N, domain = [-4, 5])
    bd_mp_seg = basic_discretization(N = N, domain = [-4, 5], use_mpmath = True)

    


if __name__ == '__main__':
    main()
