import open3d as o3d
import numpy as np


def load_ply(file_path): # ready to test -TJS
    # Load a .ply file using open3d as an numpy.array
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud = np.asarray(pcd.points) # adc changed pdc to pcd.points
    return point_cloud, pcd

source_file = 'output_x_15_y_N15.ply'
target_file = 'output2_x_15_y_N15.ply'

    # # Merge 3 ply
    # source_file = 'project_2\\test_case\\v3.ply'
    # target_file = 'project_2\\test_case\\merged.ply'
    # output_file = 'project_2\\test_case\\merged2.ply'

    
source_points, source_pcd = load_ply(source_file)
target_points, target_pcd = load_ply(target_file)


o3d.visualization.draw_geometries([source_pcd, target_pcd],
                                      window_name='ICP Visualization')