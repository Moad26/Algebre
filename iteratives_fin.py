from iter import jacobi, Gauss_Seidel, Gauss_Seidel_2
import numpy as np
import matplotlib.pyplot as plt
from verification import verification_sym
from verification import verification_pos

A = A = np.array([[6, 2, 1],
              [2, 7, 2],
              [1, 2, 8]])
b = np.array([1, 2, 3])
x_0 = np.zeros(3)

if verification_pos(A) and verification_sym(A):
    print("jacobi____________________________________________________________________________")
    solution, E = jacobi(A, b, x_0)
    print("Solution:", solution)
    Ax = A @ solution
    print("Ax:", Ax)
    print("b:", b)
    print("Différence (Ax - b):", max(Ax - b))


    print("Gauss_Seidel____________________________________________________________________________")
    solution, E = Gauss_Seidel(A, b, x_0)
    print("Solution:", solution)
    Ax = A @ solution
    print("Ax:", Ax)
    print("b:", b)
    print("Différence (Ax - b):", max(Ax - b))

    print("Gauss_Seidel_2____________________________________________________________________________")
    solution, E = Gauss_Seidel_2(A, b, x_0)
    print("Solution:", solution)
    Ax = A @ solution
    print("Ax:", Ax)
    print("b:", b)
    print("Différence (Ax - b):", max(Ax - b))

    _, E_jacobi = jacobi(A, b, x_0)
    _, E_gauss_seidel = Gauss_Seidel(A, b, x_0)
    _, E_gauss_seidel_2 = Gauss_Seidel_2(A, b, x_0)


    plt.plot(E_jacobi, label='Jacobi')
    plt.plot(E_gauss_seidel, label='Gauss-Seidel')
    plt.plot(E_gauss_seidel_2, label='Gauss-Seidel 2')
    plt.yscale('log') 
    plt.xlabel('Iterations')
    plt.ylabel('Erreur (norme ||Ax - b||)')
    plt.title('Erreur des méthodes itératives')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("pas de conditions")