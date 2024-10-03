import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def load_ply(file_path):
    # Load a .ply file using open3d as an numpy.array
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud = np.asarray(pcd.points)
    return point_cloud, pcd

def find_closest_points(source_points, target_points):
    # Align points in the souce and target point cloud data
    kdtree = cKDTree(target_points)
    distances, indices = kdtree.query(source_points)
    return indices

def estimate_normals(points, k_neighbors=30):
    """
    Use open3d to do it, e.g. estimate_normals()
    k_neighbors: The number of nearest neighbors to consider when estimating normals (you can change the value)
    """
    points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k_neighbors))
    point_norms = points.normals
    return point_norms

def normal_shooting(source_points, target_points, target_normals):
    # Find normal of point clouds, then find intersection of normals

    matched_target_points = []

    # Nearest neighbor tree of target points
    kdtree = cKDTree(target_points)
    
    # Loop through each point
    for i in range(source_points.shape[0]):
        source_point = source_points[i]

        # Identify initial source guess 
        _, closest_index = kdtree.query(source_point)
        
        target_normal = target_normals[closest_index]
        # Normalize target normal to get direction instead of magnitude
        ray_direction = target_normal / np.linalg.norm(target_normal)
        # Move source point in direction of target normal
        projected_point = source_point + ray_direction * 0.1 
        # Find the closest point in the target cloud to the projected point
        _, final_index = kdtree.query(projected_point)
        matched_target_points.append(target_points[final_index])
    
    return np.array(matched_target_points)
    
def point_to_plane(source_points, target_points, target_normals):
    # Use closest vanilla closest point matching; match target points
    indices = find_closest_points(source_points, target_points)
    matched_target_points = target_points[indices]
    
    target_normals_np = np.asarray(target_normals)
    # Align target normals with closest points
    matched_target_normals = target_normals_np[indices]

    # Find distance between point and normals; project points
    dist = np.sum((matched_target_points - source_points) * matched_target_normals, axis =1)
    projected = source_points + dist[:, np.newaxis] * matched_target_normals

    return projected

def compute_transformation(source_points, target_points):
    # Compute rotation matrix R and translation vector t that align source_points with matched_target_points
    print("computing transform")
    source_mean = np.mean(source_points, axis = 0)
    target_mean = np.mean(target_points, axis = 0) 

    # Center points
    source_difference = source_points - source_mean
    target_difference = target_points - target_mean

    # Compute covariance matrix and use in SVD
    H = source_difference.T @ target_difference
    U, _, V = np.linalg.svd(H)
    R = V.T @ U.T
    
    # if there is a reflection, flip
    if np.linalg.det(R) < 0:
        V[-1, :] *= -1
        R = V.T @ U.T

    t = target_mean - R @ source_mean
    return R, t

def apply_transformation(source_points, R, t):
    # Apply rotation R and translation t to the source points
    rotated_points = np.dot(source_points, R.T)
    new_source_points = rotated_points + t
    return new_source_points

def compute_mean_distance(source, target):
    # Calculate  mean Euclidean distance between the source and the target
    mean_distance = np.mean((((source[0]-target[0])**2)+((source[1]-target[1])**2)+((source[2]-target[2])**2))**0.5)
    return mean_distance

def calculate_mse(source_points, target_points):
    # Follow the equation in slides 
    # You may find cKDTree.query function helpful to calculate distance between point clouds with different number of points
    tree = cKDTree(target_points)
    distance, _ = tree.query(source_points)
    mse = np.mean(distance**2)
    return mse


def icp(source_points, source_pcd, target_points, target_pcd,  max_iterations=300, tolerance=1e-6, R_init=None, t_init=None, strategy="closest_point"):
    # Apply initial guess if provided
    if R_init is not None and t_init is not None:
        source_points = apply_transformation(source_points, R_init, t_init)

    for i in range(max_iterations):
        if strategy == 'closest_point':
            indices = find_closest_points(source_points, target_points)
            matched_target_points = target_points[indices]
            pass

        elif strategy == 'normal_shooting':
            target_normals = estimate_normals(target_pcd, 30)
            matched_target_points = normal_shooting(source_points, target_points, target_normals)
            pass

        elif strategy == "point-to-plane":
            target_normals = estimate_normals(target_pcd)
            matched_target_points= point_to_plane(source_points, target_points, target_normals)
            pass

        else:
            raise ValueError("Invalid strategy. Choose 'closest_point', 'normal_shooting', or 'point_to_plane'")
        
        # Complete the rest of code using source_points and matched_target_points
        R,t = compute_transformation(source_points, matched_target_points)
        aligned_source_points = apply_transformation(source_points, R, t)
        mean_dist = compute_mean_distance(aligned_source_points, matched_target_points)

        mse = calculate_mse(aligned_source_points, matched_target_points)
        print(f'iter {i} MSE {mse}')
        if mean_dist < tolerance:
            print(f"ICP converged gracefully at {i+1} iterations")
            return R, t, aligned_source_points
    
        source_points = aligned_source_points

    print(f"ICP did not converge gracefully after {max_iterations} iterations")
    return R, t, aligned_source_points

if __name__ == "__main__":
    # Easy example merge
    # source_file = 'C:\\Users\\tiann\\robot_perception\\CMPE789\\project_2\\test_case\\v1.ply'
    # target_file = 'C:\\Users\\tiann\\robot_perception\\CMPE789\\project_2\\test_case\\v2.ply'
    # output_file = 'C:\\Users\\tiann\\robot_perception\\CMPE789\\project_2\\test_case\\merged.ply'

    # Merge 3 ply
    source_file = 'C:\\Users\\tiann\\robot_perception\\CMPE789\\project_2\\test_case\\v3.ply'
    target_file = 'C:\\Users\\tiann\\robot_perception\\CMPE789\\project_2\\test_case\\merged.ply'
    output_file = 'C:\\Users\\tiann\\robot_perception\\CMPE789\\project_2\\test_case\\merged2.ply'

    strategy = "point-to-plane"
    
    source_points, source_pcd = load_ply(source_file)
    target_points, target_pcd = load_ply(target_file)
    
    # Initial guess (modifiable) - different for easy vs hard example
    R_init = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # t_init = np.array([ 15,  -15, 0]) # for first merge
    t_init = np.array([ -5,  0, 0]) # second merge

    print("Starting ICP...")
    R, t, aligned_source_points = icp(source_points, source_pcd, target_points, target_pcd, R_init=R_init, t_init=t_init, strategy=strategy)
    
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
    # o3d.visualization.draw_geometries([source_pcd_aligned, target_pcd],
    #                                   window_name='ICP Visualization')