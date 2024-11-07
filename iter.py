import numpy as np



def inv(d):
    n = len(d)
    d_inv = np.diag([1 / d[i][i] for i in range(n)])
    return d_inv


def jacobi(A, b,x_0, eps=10e-12, max_iter = 1000):
    D = np.diag(np.diag(A))
    D_inv = inv(D)
    N = D - A
    e = float("inf")
    x = x_0
    iter_count = 0
    l_er=[]
    while e > eps and iter_count < max_iter:
        x = (D_inv @ N) @ x + D_inv @ b
        e = np.linalg.norm(A @ x - b)
        l_er.append(e)
        iter_count += 1
    if iter_count == max_iter:
        print("max iter")
    return x, l_er



def inverse_matrice(matrice):
    try:
        matrice_inv = np.linalg.inv(matrice)  
        return matrice_inv
    except np.linalg.LinAlgError:
        return "La matrice n'est pas inversible."

def Gauss_Seidel(A, b,x_0, eps=10e-12, max_iter = 1000):
    D = np.diag(np.diag(A))
    N = D - A
    L = np.tril(N)
    U = np.triu(N)
    l_er = []
    DL_inv = inverse_matrice(D - L)

    e = float("inf")
    x = x_0 
    iter_count = 0

    while e > eps and iter_count < max_iter:
        x = DL_inv @ (U @ x + b)
        e = np.linalg.norm(A @ x - b)
        l_er.append(e)
        iter_count += 1
    if iter_count == max_iter:
        print("max iter")

    return x, l_er


def Gauss_Seidel_2(A, b, x_0, eps=1e-12, max_iter=1000):
    n = len(b)
    x = x_0.copy()  
    iter_count = 0
    l_er = []   
    e = float("inf")
    while e > eps and iter_count < max_iter:
        x_old = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i,i+1:], x_old[i+1:])
            x[i] = (b[i] -(sum1 + sum2)) / A[i, i]
        e = np.linalg.norm(A @ x - b)
        l_er.append(e)
        iter_count += 1
    
    if iter_count == max_iter:
        print("max iter")

    return x, l_er




