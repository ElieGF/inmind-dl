import open3d as od
import numpy as np

# Load the point cloud from a file
pcd = od.io.read_point_cloud("C:\\Users\\Elie_\\Desktop\\pcs\\1.ply")

# Display the original point cloud (before applying the color gradient)
print("Displaying the original point cloud...")
od.visualization.draw_geometries([pcd])

# Convert the point cloud to a NumPy array
points = np.asarray(pcd.points)

# Compute the centroid of the point cloud
centroid = np.mean(points, axis=0)

# Calculate distances from the centroid
distances = np.linalg.norm(points - centroid, axis=1)

# Normalize the distances to range between 0 and 1
max_distance = distances.max()
normalized_distances = distances / max_distance

# Create colors based on the normalized distances (intensity decreases with distance)
colors = np.zeros((points.shape[0], 3))
colors[:, 0] = 1 - normalized_distances
colors[:, 1] = 1 - normalized_distances
colors[:, 2] = 1 - normalized_distances
# Assign the colors to the point cloud
pcd.colors = od.utility.Vector3dVector(colors)

# Display the colored point cloud (after applying the color gradient)
od.visualization.draw_geometries([pcd])
