import numpy as np


def power_iteration(A, tol=1e-6, max_iter=1000):
    n = A.shape[0]
    randVec = np.random.rand(n)
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


if __name__ == "__main__":
    A = np.array([[2, 1], [1, 2]])
    val, vec = power_iteration(A)
    print("Test eigenvalue:", val)
    print("Test eigenvector:", vec)


def deflation(A):
    pca_pairs = []
    for i in range(A.shape[0]):
        val, vec = power_iteration(A)
        pca_pairs.append((val, vec))
        A = A - val * np.outer(vec, vec)
    return pca_pairs


def full_svd(A):
    X = A.T @ A
    pca_pairs = deflation(X)
    pca_pairs.sort(key=lambda x: x[0], reverse=True)
    n_rows = A.shape[0]
    n_cols = A.shape[1]

    # A = UEV^T
    V = np.zeros((n_cols, n_cols))
    E = np.zeros((n_rows, n_cols))
    U = np.zeros((n_rows, n_rows))

    for i in range(n_cols):
        val_i, vec_i = pca_pairs[i]

        sigma_i = np.sqrt(abs(val_i))
        E[i, i] = sigma_i  # Diagonals of E
        V[:, i] = vec_i  # Columns of V are eigenvectors

        if sigma_i > 1e-8:  # Fixing the zero division possibility
            u_i = (A @ vec_i) / sigma_i
            U[:, i] = u_i
        else:
            U[:, i] = np.zeros(n_rows)
    return U, E, V.T
