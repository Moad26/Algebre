import numpy as np

def res_tri_sup(A:np.array, B:np.array):
    n = len(B)
    X = np.zeros(n)
    X[n - 1] = B[n - 1] / A[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        X[i] = B[i]
        for k in range(i + 1, n):
            X[i] -= A[i][k] * X[k]
        X[i] /=A[i][i]

    return X


def gauss_elimination(A, b):
    n = len(b)

    aug_matrix = np.hstack((A, b.reshape(-1, 1))).astype(float)

    for i in range(n):
        
        if aug_matrix[i][i] == 0:
            for j in range(i + 1, n):
                if aug_matrix[j][i] != 0:
                    aug_matrix[[i, j]] = aug_matrix[[j, i]]  
                    break
        
        aug_matrix[i] /= aug_matrix[i][i]

        for j in range(i + 1, n):
            aug_matrix[j] -= (aug_matrix[j][i] / aug_matrix[i][i]) * aug_matrix[i]
    
    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        X[i] = aug_matrix[i][-1]  
        for j in range(i + 1, n):
            X[i] -= aug_matrix[i][j] * X[j]  
    
    return X


def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(n):
            somme = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - somme

        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                somme = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - somme)/U[i][i]
    return L, U



    