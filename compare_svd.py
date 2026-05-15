import numpy as np
from svd import full_svd


def testBench():
    # Preparing test data
    rng = np.random.default_rng(42)
    D = rng.standard_normal((100, 3))
    centroid = np.mean(D, axis=0)
    D = D - centroid
    U, E, Vt = full_svd(D)
    if E.ndim == 2:
        sigmas = np.diag(E)
    else:
        sigmas = E

    # Numpy results
    Unp, Enp, Vtnp = np.linalg.svd(D, full_matrices=False)
    sigmasNp = Enp

    # Comparing sigma values
    for g, r in zip(sigmas, sigmasNp):
        dif = np.abs(g - r)
        print(f"fullSvd:{g:.6f} | Numpy:{r:.6f} | Difference:{dif:.6f}")
        if (dif <= 1e-5):
            print("Safe difference")
        else:
            print("Dangerous difference")
    print("="*40)

    print("Reconstruction Error Test")
    Drecons = U @ E @ Vt
    FrobNorm = np.linalg.norm(D - Drecons) / np.linalg.norm(D)
    if FrobNorm < 1e-4:
        print(f"FrobNorm:{FrobNorm:6f} | Safe reconstruction error")
    else:
        print(f"FrobNorm:{FrobNorm:6f} | Dangerous reconstruction error")
    print("="*40)

    print("Matrix Orthogonality Test")
    Iu = U.T @ U  # If orthogonal then I
    errU = np.linalg.norm(Iu - np.eye(3))

    Iv = Vt @ Vt.T
    errV = np.linalg.norm(Iv - np.eye(3))
    print(f"U Orthogonality Error (U^T @ U - I): {errU:.6e}")
    print(f"V Orthogonality Error (V @ V^T - I): {errV:.6e}")

    if errU < 1e-8 and errV < 1e-8:
        print("Result is safe Gram-Schmidt works perfectly")
    else:
        print("Result dangerous check Gram-Schmidt")
    print("="*40)


if __name__ == "__main__":
    testBench()
