import numpy as np

def E(n, i, j , a):
    E = np.eye(n)
    E[i][j] = a
    return E

def decomp_lu(A):
    n = len(A)
    U = A.copy()
    L = np.eye(n)

    for i in range(n):

        if U[i][i] == 0:
            for j in range(i + 1, n):
                if U[j][i] != 0:
                    U[[i, j]] = U[[j, i]] 
                    break
        

        for j in range(i + 1, n):
            facteur = U[j, i] / U[i, i]
            L = L @ E(n, i, j, facteur)
            U = E(n, i, j, -facteur) @ U
    return L, U 
            

    
""" A = np.array([[3  , -2 , 6 , -5  ],
              [24 , -12, 41 , -39],
              [-27, 18 , -62, 54 ],
              [9  , 14 , 15 , -47]])

L, U = decomp_lu(A)

print("Matrice L (inférieure) :\n", L)
print("Matrice U (supérieure) :\n", U)
print(L @ U) """
    

        


