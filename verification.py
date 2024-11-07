import numpy as np

def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(len(matrix)):
        minor = [row[:col] + row[col + 1:] for row in matrix[1:]]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(minor)

    return det

def verification_pos(M):
    n = len(M)
    c = True
    for k in range(1, n + 1):
        minor_k = M[:k, :k].tolist()  
        if determinant(minor_k) <= 0:
             c = False
    return c

def verification_sym(M):
    n = len(M)
    c = True
    for i in range(n):
        for j in range(i + 1, n):
            if M[i][j] != M[j][i]:
                c = False
    return c

def verification_min(M):
    n = len(M)
    c = True
    for k in range(1, n + 1):
        minor_k = M[:k, :k].tolist()  
        if determinant(minor_k) == 0:
             c = False
    return c

def verification_det(M):
    return determinant(M) != 0






