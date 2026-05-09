import numpy as np
import open3d as o3d
from svd import full_svd
from scipy.spatial import KDTree

pcd = o3d.io.read_point_cloud("data/pca/bunny.pcd")
points = np.asarray(pcd.points)
curvatures = np.zeros(points.shape[0])

kNearTree = KDTree(points)
kNum = 100


for i in range(points.shape[0]):
    distances, indices = kNearTree.query(points[i], k=kNum)
    neighbors = points[indices]

    meanCentre = np.mean(neighbors, axis=0)
    centredNeighbors = neighbors - meanCentre

    neighborCov = centredNeighbors.T @ centredNeighbors
    U, E, Vt = full_svd(neighborCov)  # E is 3x3 matrix

    eigen1 = E[0, 0]
    eigen2 = E[1, 1]
    eigen3 = E[2, 2]

    # curvVal = eigen3 / (eigen1 + eigen2 + eigen3) check zero division.Covariance matric is semi-positive definite

    totalDenom = eigen1 + eigen2 + eigen3
    if totalDenom > 1e-12:
        curvatures[i] = eigen3 / totalDenom
    else:
        curvatures[i] = 0.0

# Normalize for ColorMap
minCurve = np.min(curvatures)
maxCurve = np.max(curvatures)
normalizedCurves = (curvatures - minCurve) / (maxCurve - minCurve)

colors = np.zeros((points.shape[0], 3))
colors[:, 0] = normalizedCurves
colors[:, 2] = 1.0 - normalizedCurves

pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("output/bunny_curvatures.pcd", pcd)
