import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def load_ply(file_path): # ready to test -TJS
    # Load a .ply file using open3d as an numpy.array
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud = np.asarray(pcd.points) # adc changed pdc to pcd.points
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
    normals_estimate = o3d.geometry.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k_neighbors))
    return normals_estimate

def normal_shooting(source_points, source_normals, target_points, target_normals): # ready to test but I think this equation is wrong - TJS
    # similar to point to plane, find normal and then intersection between another point cloud and based on the intersection we'll find closest point to intersection.
    # calculate intersection and then find closest point
    # normal is vertical vector orientation (use estimate_normals)
    matched_target_points = min(np.sum(((source_points - target_points)*target_normals*source_normals)**2))
    return matched_target_points
    
def point_to_plane(source_points, target_points, target_normals): # ready to test (ensure equation!) - TJS
    matched_target_points = min(np.sum(((source_points - target_points)*target_normals)**2))
    return matched_target_points

def compute_transformation(source_points, target_points): # ready to test - TJS (I keep seeing .T be used with np.dot, investigate)
    # Compute the optimal rotation matrix R and translation vector t that align source_points with matched_target_points
    source_mean = np.mean(source_points)
    target_mean = np.mean(target_points)
    source_difference = source_points - source_mean
    target_difference = target_points - target_mean
    h_matrix = np.dot(source_difference.T, target_difference)
    U, _, V = np.linalg.svd(h_matrix)
    R = np.dot(V, U.T)
    t = -R*source_mean.T + target_mean.T
    return R, t

def apply_transformation(source_points, R, t): # Ready to test - TJS
    # Apply the rotation R and translation t to the source points
    rotated_points = np.dot(source_points, R.T) # adc fact check transformation .T source_points.T, R
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
    # all losses should be about similar
    for i in range(max_iterations):
        # The number of points may be different in the source and the target, so align them first
        if strategy == 'closest_point':
            indices = find_closest_points(source_points, target_points)
            matched_target_points = target_points[indices]
            pass

        elif strategy == 'normal_shooting':
            source_normals = estimate_normals(source_points)
            target_normals = estimate_normals(target_points)
            # fix what comes out
            matched_target_points = normal_shooting(source_points, target_points, source_normals, target_normals)
            pass

        elif strategy == "point-to-plane":
            target_normals = estimate_normals(target_points)
            matched_target_points= point_to_plane(source_points, target_points, target_normals)
            pass

        else:
            raise ValueError("Invalid strategy. Choose 'closest_point', 'normal_shooting', or 'point_to_plane'")
        
        # Complete the rest of code using source_points and matched_target_points
        R,t = compute_transformation(source_points, matched_target_points)
        aligned_source_points = apply_transformation(source_points, R, t)
        mean_dist = compute_mean_distance(aligned_source_points, matched_target_points)

        if mean_dist < tolerance:
            print(f"ICP converged gracefully at {i+1} iterations")
            return R, t, aligned_source_points
    
        aligned_source_points = source_points
    print(f"ICP did not converge gracefully after {max_iterations} iterations")
    return R, t, aligned_source_points

if __name__ == "__main__":
    source_file = 'project_2/test_case/v1.ply'
    target_file = 'project_2/test_case/v2.ply'
    output_file = 'merged.ply'
    strategy = ""
    
    source_points = load_ply(source_file)
    target_points = load_ply(target_file)
    
    # Initial guess (modifiable) # good initial guess is important, feel free to go higher
    # if we do our own data, we can get our initial guess from the ground truth on that and input it here for the original tests!
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