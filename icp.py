import numpy as np
import open3d as o3d
from svd import full_svd
from scipy.spatial import KDTree


pcdInit = o3d.io.read_point_cloud("data/icp/bun045_initial_align.pcd")
pointsInit = np.asarray(pcdInit.points)

pcdTarget = o3d.io.read_point_cloud("data/icp/bun000_target.pcd")
pointsTarget = np.asarray(pcdTarget.points)

pcdTarget.paint_uniform_color([0, 0, 1])
pcdInit.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcdTarget, pcdInit])  # initial state

kNearTree = KDTree(pointsTarget)
maxIter = 30

# Transform Q to P


def kabsch(P, Q):
    centroidP = np.mean(P, axis=0)
    centroidQ = np.mean(Q, axis=0)

    centredP = P - centroidP
    centredQ = Q - centroidQ

    H = centredP.T @ centredQ

    U, E, Vt = full_svd(H)

    det = np.linalg.det(U @ Vt)
    R = U @ np.diag([1, 1, det]) @ Vt

    t = centroidP - (R @ centroidQ)
    return R, t


prev_err = 0
for i in range(maxIter):
    distances, indices = kNearTree.query(pointsInit, k=1)

    neighbors = pointsTarget[indices]
    R, t = kabsch(neighbors, pointsInit)
    pointsInit = (pointsInit @ R.T) + t

    # track RMSE between matched pairs as convergence criterion
    avgerror = np.sqrt(np.mean(distances**2))
    print(f"Iteration:{i+1},Average Error:{avgerror:.4f}")
    if i > 0 and abs(prev_err - avgerror) < 1e-7:
        break
    prev_err = avgerror

pcdInit.points = o3d.utility.Vector3dVector(pointsInit)

pcdTarget.paint_uniform_color([0, 0, 1])
pcdInit.paint_uniform_color([1, 0, 0])
mergedPoints = pcdInit + pcdTarget

o3d.visualization.draw_geometries([mergedPoints])

o3d.io.write_point_cloud("output/aligned_bunny.pcd", mergedPoints)
