import numpy as np
  

def cholesky_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n) :
        for j in range(i + 1):

            if i == j:
                sum_square = sum(L[i, k] ** 2 for k in range(j))

                L[i ,j] = np.sqrt(A[i, i] - sum_square)

            else:

                sum_products = sum(L[i, k] * L[j, k] for k in range(j))

                L[i, j] = (A[i ,j] - sum_products) / L[j, j]
  
    return L


