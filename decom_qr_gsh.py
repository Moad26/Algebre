import numpy as np
from gram_shmidt import gram_schmidt


def QR_gram_schmidt_carrre(A):
    
    n = len(A)
    
    q = gram_schmidt(A)
    r = np.zeros((n ,n))

    for i in range(n):
        for j in range(n):
            if i <= j:
                r[i][j] = A[:, i] @ q[:, j]

    return q, r



def QR_Decomposition(A):
    n, m = A.shape 

    Q = np.empty((n, n)) 
    u = np.empty((n, n)) 

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] 

        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) 

    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]

    return Q, R


def diag_sign(A):

    D = np.diag(np.sign(np.diag(A)))

    return D

def adjust_sign(Q, R):

    D = diag_sign(Q)

    Q[:, :] = Q @ D
    R[:, :] = D @ R

    return Q, R

def QR_eigval(A, tol = 1e-12, maxiter = 1000):

    A_old = np.copy(A)
    A_new = np.copy(A)

    diff = np.inf
    i = 0

    while (diff > tol) and (i < maxiter):
        A_old[:, :] = A_new

        Q, R = QR_Decomposition(A_old)

        A_new = R @ Q

        diff = np.abs(A_new - A_old).max()
        i += 1

    eigval = np.diag(A_new)

    return eigval

""" A = np.array([[81, -27],
              [-27, 9]])

print(QR_eigval(A))
 """
