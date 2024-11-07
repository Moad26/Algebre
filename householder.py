import numpy as np

def householder_projection(a):

    v = a.copy()
    v[0] += np.sign(a[0]) * np.linalg.norm(a)
    v = v / np.linalg.norm(v)
    H = np.eye(len(a)) - 2 * np.outer(v, v)

    return H

def qr_decomposition(A):
    m, n = A.shape

    Q = np.eye(m)
    R = A.copy()

    for i in range(n):
        H = np.eye(m)
        H_i = householder_projection(R[i:, i])
        H[i:,i:] = H_i
        R = H @ R
        Q = Q @ H

    return Q, R

""" A = np.array([[4, 1, -2, 2],
              [1, 2, 0, 1],
              [-2, 0, 3, -2],
              [2, 1, -2, -1]], dtype=float)

Q, R = qr_decomposition(A)

print("Matrice Q :")
print(Q)
print("\nMatrice R :")
print(R) """