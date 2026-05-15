import numpy as np
import open3d as o3d
from svd import full_svd


pcd = o3d.io.read_point_cloud("data/pca/box_5_noisy.pcd")  # Read data
points = np.asarray(pcd.points)  # Use as matrix

meanCentre = np.mean(points, axis=0)  # Mean of every colum
# Now points centered (0,0) PCA works according to origin
centeredPoints = points - meanCentre
U, E, Vt = full_svd(centeredPoints)
principleCom = Vt

# Projected to the principle components
new_local_points = centeredPoints @ principleCom.T

min_bounds = np.min(new_local_points, axis=0)  # Min of every axis
max_bounds = np.max(new_local_points, axis=0)  # Max of every axis

cornerBounds = np.array([[min_bounds[0], min_bounds[1], min_bounds[2]],
                        [max_bounds[0], min_bounds[1], min_bounds[2]],
                        [min_bounds[0], max_bounds[1], min_bounds[2]],
                        [max_bounds[0], max_bounds[1], min_bounds[2]],
                        [min_bounds[0], min_bounds[1], max_bounds[2]],
                        [max_bounds[0], min_bounds[1], max_bounds[2]],
                        [min_bounds[0], max_bounds[1], max_bounds[2]],
                        [max_bounds[0], max_bounds[1], max_bounds[2]]])

# Corners projected to principle components.Rotate and put back to old location
new_cornerBounds = (cornerBounds @ principleCom) + meanCentre

edgesOfObb = [[0, 1], [1, 3], [3, 2], [2, 0],
              [4, 5], [5, 7], [7, 6], [6, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

obbColor = [[0, 0, 1] for i in range(len(edgesOfObb))]  # Edges are blue
obb_lineset = o3d.geometry.LineSet()
obb_lineset.points = o3d.utility.Vector3dVector(
    new_cornerBounds)  # 8 corner to draw lines
obb_lineset.lines = o3d.utility.Vector2iVector(
    edgesOfObb)  # Connections of edges
obb_lineset.colors = o3d.utility.Vector3dVector(obbColor)


# AABB
min_bounds_aabb = np.min(points, axis=0)
max_bounds_aabb = np.max(points, axis=0)

cornerBoundsOfaabb = np.array([
    [min_bounds_aabb[0], min_bounds_aabb[1], min_bounds_aabb[2]],
    [max_bounds_aabb[0], min_bounds_aabb[1], min_bounds_aabb[2]],
    [min_bounds_aabb[0], max_bounds_aabb[1], min_bounds_aabb[2]],
    [max_bounds_aabb[0], max_bounds_aabb[1], min_bounds_aabb[2]],
    [min_bounds_aabb[0], min_bounds_aabb[1], max_bounds_aabb[2]],
    [max_bounds_aabb[0], min_bounds_aabb[1], max_bounds_aabb[2]],
    [min_bounds_aabb[0], max_bounds_aabb[1], max_bounds_aabb[2]],
    [max_bounds_aabb[0], max_bounds_aabb[1], max_bounds_aabb[2]]
])

aabb_colors = [[1, 0, 0] for i in range(len(edgesOfObb))]
aabb_lineset = o3d.geometry.LineSet()
aabb_lineset.points = o3d.utility.Vector3dVector(cornerBoundsOfaabb)
aabb_lineset.lines = o3d.utility.Vector2iVector(edgesOfObb)
aabb_lineset.colors = o3d.utility.Vector3dVector(aabb_colors)

o3d.visualization.draw_geometries([pcd, obb_lineset, aabb_lineset])

# Sampling process to draw points along the edges


def sampling(corners, edges, num_points=100):
    times = np.linspace(0, 1, num_points)[:, np.newaxis]
    sampledPoints = []
    for edge in edges:
        p1 = corners[edge[0]]
        p2 = corners[edge[1]]
        points = p1 + times * (p2 - p1)
        sampledPoints.extend(points)
    return np.array(sampledPoints)


# Making lines of OBB
obb_sampled = sampling(new_cornerBounds, edgesOfObb, 100)
obb_pcd = o3d.geometry.PointCloud()  # Invert to pcd all points
obb_pcd.points = o3d.utility.Vector3dVector(obb_sampled)
obb_pcd.paint_uniform_color([0, 0, 1])

# Makine lines of AABB
aabb_sampled = sampling(cornerBoundsOfaabb, edgesOfObb, 100)
aabb_pcd = o3d.geometry.PointCloud()
aabb_pcd.points = o3d.utility.Vector3dVector(aabb_sampled)
aabb_pcd.paint_uniform_color([1, 0, 0])

# Merge both of them
mergedPoints = aabb_pcd + obb_pcd + pcd

o3d.io.write_point_cloud("output/box5.pcd", mergedPoints)
print("Points sampled and saved")
