import numpy as np
import open3d as o3d
from svd import full_svd
from scipy.spatial import KDTree

pcd = o3d.io.read_point_cloud("data/pca/bunny.pcd")
points = np.asarray(pcd.points)

normalVecs = np.zeros(points.shape)

kNearTree = KDTree(points)
kNum = 100
centroid = np.mean(points, axis=0)

for i in range(points.shape[0]):
    distances, indices = kNearTree.query(points[i], k=kNum)

    neighbors = points[indices]

    meanCentre = np.mean(neighbors, axis=0)
    centredNeigbors = neighbors - meanCentre

    neighborCov = centredNeigbors.T @ centredNeigbors
    U, E, Vt = full_svd(neighborCov)
    normalVecs[i] = Vt[-1]  # Smallest sigma last row vector of Vt

    if (np.dot(points[i] - centroid, normalVecs[i]) < 0):
        normalVecs[i] = normalVecs[i] * -1

pcd.normals = o3d.utility.Vector3dVector(normalVecs)
nColors = (normalVecs + 1.0) / 2.0
pcd.colors = o3d.utility.Vector3dVector(nColors)
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
o3d.io.write_point_cloud("output/bunny_normals.pcd", pcd)
