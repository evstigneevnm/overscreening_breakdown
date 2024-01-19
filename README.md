# Overview of overscreening_breakdown

#### This is a repository for the pseudo-spectral collocation numerical method for the overscreening breakdown problem. For the discription, see bellow.

## Requirements and Installation
* #### python 3.8 or newer with modules:
- numpy
- matplotlib
- scipy
- pickle
* #### clone repository locally into your local folder

## Testing:
#### For manufactured glabal tests run the following from your local folder:
```
python3 test_manufactures_solutions.py
```
resutls will be stored in "./figures" folder.

Other tests are:
- test_basis.py
- test_matrices.py
- test_solve_nonlinear_problem.py
- test_different_mu.ipynb

## Description
#### Problem formulation
Physical problem ...

#### Numerical method
The problem is solved on the $R^+$ domain by using an analytical mapping to the domain [0, pi].
Newton's method with spectral discretization over the collocation method is used in order to find the solution from some initial guess. The method includes homotopy for difficult cases that allow one to converge to the solution.
For the example-driven manual, see:
```
execution_example.ipynb
```
For the tests on the stability of parameters, see:
```
test_different_mu.ipynb
```

