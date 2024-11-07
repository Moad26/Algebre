import numpy as np

def gram_schmidt(v):
    n, m = v.shape

    ortho_v = np.zeros((n, m))

    ortho_v[0] = v[0] / np.linalg.norm(v[0])

    for i in range(1, n):
        u = v[i] - sum(np.dot(v[i], ortho_v[k]) * ortho_v[k] for k in range(i))
        ortho_v[i] = u / np.linalg.norm(u)

    return ortho_v

