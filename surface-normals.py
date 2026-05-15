import numpy as np
import open3d as o3d
from svd import full_svd
from scipy.spatial import KDTree

pcd = o3d.io.read_point_cloud("data/pca/bunny.pcd")
points = np.asarray(pcd.points)

normalVecs = np.zeros(points.shape)

kNearTree = KDTree(points)
kNum = 30
centroid = np.mean(points, axis=0)

for i in range(points.shape[0]):
    distances, indices = kNearTree.query(points[i], k=kNum)

    neighbors = points[indices]

    meanCentre = np.mean(neighbors, axis=0)
    centredNeigbors = neighbors - meanCentre
    N = centredNeigbors.shape[0]
    neighborCov = (centredNeigbors.T @ centredNeigbors) / (N-1)

    U, E, Vt = full_svd(neighborCov)
    normalVecs[i] = Vt[-1]  # Smallest sigma last row vector of Vt

    if (np.dot(points[i] - centroid, normalVecs[i]) < 0):
        normalVecs[i] = normalVecs[i] * -1

pcd.normals = o3d.utility.Vector3dVector(normalVecs)
pColors = (normalVecs + 1.0) / 2.0
pcd.colors = o3d.utility.Vector3dVector(pColors)

normalVecLen = 0.007  # Scaling normal vectors

lineStarts = points
lineEnds = points + (normalVecs * normalVecLen)
# vstack first place starts then ends
linePoints = np.vstack([lineStarts, lineEnds])
N = points.shape[0]  # i and i + N makes a line
lineIndices = np.array([[i, i+N] for i in range(N)])  # i and i + N is a set

lineSet = o3d.geometry.LineSet()
lineSet.points = o3d.utility.Vector3dVector(linePoints)
lineSet.lines = o3d.utility.Vector2iVector(lineIndices)
lineSet.colors = o3d.utility.Vector3dVector(pColors)


o3d.visualization.draw_geometries([pcd, lineSet])
o3d.io.write_point_cloud("output/bunny_normals.pcd", pcd)
