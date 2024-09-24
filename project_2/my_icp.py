import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def load_ply(file_path): # ready to test -TJS
    # Load a .ply file using open3d as an numpy.array
    pcd = o3d.io.read_point_cloud("v1.ply")
    point_cloud = np.asarray(pcd)
    return point_cloud

def find_closest_points(source_points, target_points):
    # Align points in the souce and target point cloud data
    kdtree = cKDTree(target_points)
    distances, indices = kdtree.query(source_points)
    return indices

def estimate_normals(points, k_neighbors=30): # ready to test - TJS
    """
    Use open3d to do it, e.g. estimate_normals()
    k_neighbors: The number of nearest neighbors to consider when estimating normals (you can change the value)
    """
    points = o3d.geometry.estimate_normals(search_param=KDTreeSearchParamKNN with knn = k_neighbors)
    pass

def normal_shooting(source_points, source_normals, target_points, target_normals):
    pass

def point_to_plane(source_points, target_points, target_normals):
    pass
 
def compute_transformation(source_points, target_points): # ready to test - TJS (I keep seeing .T be used with np.dot, investigate)
    # Compute the optimal rotation matrix R and translation vector t that align source_points with matched_target_points
    source_mean = np.mean(source_points)
    target_mean = np.mean(target_points)
    source_difference = source_points - source_mean
    target_difference = target_points - target_mean
    h_matrix = np.dot(source_difference, target_difference)
    svd = np.linalg.svd(h_matrix) # this may be wrong -TJS
    R = np.dot(svd) # this may also be wrong
    t = target_mean - np.dot(R, source_mean)
    return R, t

def apply_transformation(source_points, R, t): # Ready to test - T (ensure order of R and t is right)
    # Apply the rotation R and translation t to the source points
    rotated_points = np.dot(source_points, R)
    new_source_points = rotated_points + t
    return new_source_points

def compute_mean_distance(source, target): # ready to test - TJS
    # Compute the mean Euclidean distance between the source and the target
    mean_distance = np.mean(((target[0]-source[0])**2)+((target[1]-source[1])**2)+((target[2]-source[2])**2)**0.5)
    return mean_distance

def calculate_mse(source_points, target_points): # ready to test, ensure source points is nx3 array - TJS
    # Follow the equation in slides 
    # You may find cKDTree.query function helpful to calculate distance between point clouds with different number of points
    print(source_points)
    print(target_points)
    tree = cKDTree(source_points)
    distance, _ = tree.query(target_points)
    mse = np.mean(distance**2)
    return mse


def icp(source_points, target_points, max_iterations=100, tolerance=1e-6, R_init=None, t_init=None, strategy="closest_point"):
    # Apply initial guess if provided
    if R_init is not None and t_init is not None:
        source_points = apply_transformation(source_points, R_init, t_init)
        
    # ICP algorithm, return the rotation R and translation t
    for i in range(max_iterations):
        # The number of points may be different in the source and the target, so align them first
        if strategy == 'closest_point':
            indices = find_closest_points(source_points, target_points)
            matched_target_points = target_points[indices]
            pass
        elif strategy == 'normal_shooting':
            pass
        elif strategy == "point-to-plane":
            pass
        else:
            raise ValueError("Invalid strategy. Choose 'closest_point', 'normal_shooting', or 'point_to_plane'")
        
        # Complete the rest of code using source_points and matched_target_points
        pass
    
    aligned_source_points = source_points
    return R, t, aligned_source_points

if __name__ == "__main__":
    source_file = 'source.ply'
    target_file = 'target.ply'
    output_file = 'merged.ply'
    strategy = ""
    
    source_points = load_ply(source_file)
    target_points = load_ply(target_file)
    
    # Initial guess (modifiable)
    R_init = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t_init = np.array([0, 0, 0])
    
    print("Starting ICP...")
    R, t, aligned_source_points = icp(source_points, target_points, R_init=R_init, t_init=t_init, strategy=strategy)
    
    print("ICP completed.")
    print("Rotation Matrix:")
    print(R)
    print("Translation Vector:")
    print(t)
    mse = calculate_mse(aligned_source_points, target_points)
    print(f"Mean Squared Error (MSE): {mse}")
    
    # Combine aligned source and target points
    combined_points = np.vstack((aligned_source_points, target_points))
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    
    # Save
    o3d.io.write_point_cloud(output_file, combined_pcd)
    print(f"Combined point cloud saved to '{output_file}'")
    
    # Visualization
    source_pcd_aligned = o3d.geometry.PointCloud()
    source_pcd_aligned.points = o3d.utility.Vector3dVector(aligned_source_points)
    
    target_pcd = o3d.io.read_point_cloud(target_file)
    o3d.visualization.draw_geometries([source_pcd_aligned.paint_uniform_color([0, 0, 1]), target_pcd.paint_uniform_color([1, 0, 0])],
                                      window_name='ICP Visualization')
