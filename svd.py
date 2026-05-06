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


def deflation(A):
    pca_pairs = []
    for i in range(A.shape[0]):
        val, vec = power_iteration(A)
        pca_pairs.append((val, vec))
        # Small mistakes lead to larger problems
        vec = vec / np.linalg.norm(vec)
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
    E = np.zeros((n_cols, n_cols))
    U = np.zeros((n_rows, n_cols))

    for i in range(n_cols):
        val_i, vec_i = pca_pairs[i]
        vec_i = vec_i / np.linalg.norm(vec_i)

        # np.sqrt(max(val_i, 0)) you could use
        sigma_i = np.sqrt(max(val_i, 0))  # ???????
        E[i, i] = sigma_i  # Diagonals of E
        V[:, i] = vec_i  # Columns of V are eigenvectors

        if sigma_i > 1e-8:  # Fixing the zero division possibility
            u_i = (A @ vec_i) / sigma_i
            u_i = u_i / np.linalg.norm(u_i)
            U[:, i] = u_i
        else:
            U[:, i] = np.zeros(n_rows)
    return U, E, V.T


if __name__ == "__main__":
    print("--- Power Iteration Test ---")
    A_test = np.array([[2, 1], [1, 2]])
    val, vec = power_iteration(A_test)
    print("Test eigenvalue:", val)
    print("Test eigenvector:", vec)

    print("\n--- SVD Test ---")
    U, E, Vt = full_svd(A_test)
    A_rec = U @ E @ Vt
    print("Reconstruction error:", np.linalg.norm(A_test - A_rec))
