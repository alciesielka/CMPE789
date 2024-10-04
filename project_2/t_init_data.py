import open3d as o3d
import numpy as np\


def load_ply(file_path): # ready to test -TJS
    # Load a .ply file using open3d as an numpy.array
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud = np.asarray(pcd.points)
    return point_cloud, pcd

source_file = 'output_x_15_y_N15.ply' # my output from Carla simulationn
target_file = 'output3_x_15_y_N15.ply'
source_points, source_pcd = load_ply(source_file)
target_points, target_pcd = load_ply(target_file)

source_bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(source_points)) 
target_bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(target_points)) 
# AABB is box that surrounds all points in point cloud and is aliged with xyz axes. edges are // to xyz. 
# -  get bounding box by alignig axes (AABB)
# -  convert np array to open3d format

t_init = target_bounding_box.get_center() - source_bounding_box.get_center()  # Translation


print(t_init)


