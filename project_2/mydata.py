import carla
import numpy as np


def save_ply_file(file, points):
    with open(file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def lidar_callback(data):   
    # Extract point cloud data
    point_cloud = np.frombuffer(data.raw_data, dtype=np.float32)
    point_cloud = point_cloud.reshape((-1, 4))  # Each point has x, y, z, intensity
    points = point_cloud[:, :3]  # Keep only x, y, z

    # Save to PLY file
    save_ply_file('test_output.ply', points)


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    world = client.get_world()

    # spawn a vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprint = blueprint_library.find('vehicle.testla.model3')
    sp = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_blueprint, sp)

    # attach lidar sensor
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('points_per_second', '50000')
    lidar_bp.set_attribute('range', '100')
    lidar_spawn_point = carla.Transform(carla.Location(x=0, z=2))  # Adjust height
    lidar = world.spawn_actor(lidar_bp, lidar_spawn_point, attach_to=vehicle)

    # Listen for LiDAR data
    lidar.listen(lidar_callback)

    try:
        while True:
            world.tick()  # Update the simulation
    finally:
        vehicle.destroy()
        lidar.destroy()

if __name__ == '__main__':
    main()
