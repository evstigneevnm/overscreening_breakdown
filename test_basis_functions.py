from spectral1d import basis_functions, basic_discretization
from matplotlib import pyplot as plt
import numpy as np
import mpmath as mp

def main():
    N = 10
    L = 4;
    bf_np = basis_functions(N, L)
    bf_mp = basis_functions(N, L, True)

    print("testing numpy implementation...")
    bf_np.test()
    print("ok")
    print("testing mpmath implementation...")
    bf_mp.test()
    print("ok")

    bd_np = basic_discretization(N, [0, np.inf])
    bd_mp = basic_discretization(N, [0, mp.inf], L, True)
    


if __name__ == '__main__':
    main()
