import numpy as np

_RNG = np.random.default_rng(42)


def power_iteration(A, tol=1e-6, max_iter=1000):

    n = A.shape[0]
    randVec = _RNG.standard_normal(n)  # Gaussian standart normal
    randVec = randVec / np.linalg.norm(randVec)

    for i in range(max_iter):
        newVec = A @ randVec
        # Normalize every iteration for avoiding overflow
        newVec = newVec / np.linalg.norm(newVec)
        dotProduct = np.dot(newVec, randVec)
        # Used abs because if eigenvector make a 180 degree turn it does not effect it
        error = 1.0 - np.abs(dotProduct)

        if (error < tol):
            randVec = newVec
            break

        randVec = newVec

    # You can use dot product also for this equation
    eigenVal = np.dot(randVec, (A @ randVec))
    return eigenVal, randVec


def deflation(A):
    pcaPairs = []
    for i in range(A.shape[0]):
        val, vec = power_iteration(A)
        pcaPairs.append((val, vec))
        vec = vec / np.linalg.norm(vec)
        A = A - val * np.outer(vec, vec)
    return pcaPairs


def full_svd(A):
    X = A.T @ A
    pcaPairs = deflation(X)
    pcaPairs.sort(key=lambda x: x[0], reverse=True)
    nRows = A.shape[0]
    nCols = A.shape[1]

    # A = UEV^T
    V = np.zeros((nCols, nCols))
    E = np.zeros((nCols, nCols))
    U = np.zeros((nRows, nCols))

    for i in range(nCols):
        val_i, vec_i = pcaPairs[i]
        vec_i = vec_i / np.linalg.norm(vec_i)

        # np.sqrt(max(val_i, 0)) you could use
        sigma_i = np.sqrt(max(val_i, 0))
        E[i, i] = sigma_i  # Diagonals of E
        V[:, i] = vec_i  # Columns of V are eigenvectors

        if sigma_i > 1e-8:  # Fixing the zero division possibility
            u_i = (A @ vec_i) / sigma_i
            u_i = u_i / np.linalg.norm(u_i)
            U[:, i] = u_i
        else:
            U[:, i] = np.zeros(nRows)
    return U, E, V.T
