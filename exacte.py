import numpy as np
from decom_qr_gsh import QR_Decomposition
from lu2 import lu_decomposition
from cholesky import cholesky_decomposition
from verification import verification_min
from verification import verification_sym
from verification import verification_pos
from verification import verification_det
# Matrice pour la décomposition LU
A_lu = np.array([[4, 3, 2],
                 [6, 3, 4],
                 [2, 1, 3]])

# Matrice pour la décomposition QR
A_qr = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Matrice pour la décomposition de Cholesky (doit être symétrique et définie positive)
A_cholesky = np.array([[6, 3, 4],
                       [3, 6, 5],
                       [4, 5, 10]])

if verification_min(A_lu):
    L, U = lu_decomposition(A_lu) 
    print(L,"\n", U)
else:
    print("pas condition lu")

if np.linalg.det(A_qr):
    Q, R = QR_Decomposition(A_qr) 
    print(Q,"\n", R)
else:
    print("pas condition qr")


if verification_sym(A_cholesky) and verification_pos(A_cholesky):
    L1= cholesky_decomposition(A_cholesky) 
    print(L1)
else:
    print("pas condition cholesky")

