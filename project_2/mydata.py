import carla
import numpy as np
import time

def save_ply_file(file, points):
    with open(file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")  # Use ASCII format instead of binary
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for point in points:
            x, y, z = point
            f.write(f"{x} {y} {z}\n")

    # try with top        
    with open(file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")  # Use ASCII format instead of binary
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for point in points:
            min_z = -3.0
            max_z = 3.0
            x, y, z = point
            # Normalize z to be within 0-255 for color
            z_normalized = (z - min_z) / (max_z - min_z)
            # Ensure normalization is within 0 to 1
            z_normalized = min(max(z_normalized, 0), 1)

            # Create a color gradient from blue (0, 0, 255) to red (255, 0, 0)
            r = int(z_normalized * 255)  # Red component increases with height
            g = 0                         # Green component remains zero
            b = int((1 - z_normalized) * 255)

            f.write(f"{x} {y} {z} {r} {g} {b}\n")

def lidar_callback(data, points):   
    print("LiDAR data received.")
    # Extract point cloud data
    point_cloud = np.frombuffer(data.raw_data, dtype=np.float32)
    point_cloud = point_cloud.reshape((-1, 4))  # Each point has x, y, z, intensity
    new_points = point_cloud[:, :3]  # Keep only x, y, z

    # Save to PLY file'
    points.extend(new_points.tolist())  # Collect points


def set_spectator_view(world, vehicle):
    spectator = world.get_spectator()  
    transform = vehicle.get_transform()  
    
    spectator_location = transform.location + carla.Location(x=-10, z=5)  # Behind and above the vehicle
    spectator_transform = carla.Transform(spectator_location, transform.rotation)
    
    spectator.set_transform(spectator_transform)
    print("Spectator camera set to follow the vehicle.")


def move_vehicle(vehicle):
    control = carla.VehicleControl()
    control.throttle = 0.5  # Apply 50% throttle to move forward
    control.steer = 0.0     # No steering (move straight)
    control.brake = 0.0     # No brake
    vehicle.apply_control(control)
    print("Vehicle moving.")

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    world = client.get_world()

    # spawn a vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprint = blueprint_library.find('vehicle.audi.a2')
    sp = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_blueprint, sp)
    print("Vehicle spawned successfully.")

    # spwn vehicle two
    vehicle2_blueprint = blueprint_library.find('vehicle.audi.a2')

    sp2 = carla.Transform(carla.Location(x=sp.location.x - 40, y=sp.location.y - 10, z=sp.location.z), sp.rotation)
    vehicle2 = world.spawn_actor(vehicle2_blueprint, sp2)
    print("Vehicle2 spawned successfully.")


    # attach lidar sensor to vehicle 1
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '50000')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', '10')

    lidar_spawn_point = carla.Transform(carla.Location(x=0, z=2))  # Adjust height
    lidar = world.spawn_actor(lidar_bp, lidar_spawn_point, attach_to=vehicle)

    # Listen for LiDAR data
    points = []
    lidar.listen(lambda data: lidar_callback(data, points))

    # attach lidar sensor to vehicle 2
    lidar2_spawn_point = carla.Transform(carla.Location(x=0, z=2))  # Adjust height
    lidar2 = world.spawn_actor(lidar_bp, lidar2_spawn_point, attach_to=vehicle2)

    # Listen for LiDAR data
    points2 = []
    lidar2.listen(lambda data: lidar_callback(data, points2))



    try:
        print("Simulation running for 30 seconds...")
        start_time = time.time()

        # Run the simulation for 5 seconds
        while time.time() - start_time < 5:
            world.tick()  # Keep the simulation running
            
            set_spectator_view(world, vehicle)  # Update the spectator camera to follow the vehicle
    
            move_vehicle(vehicle)
            move_vehicle(vehicle2)


            time.sleep(0.1)  # Small delay to allow smooth camera movement


    finally:
        if points: 
            save_ply_file('test_output_x_40_y_10.ply', points)
            print("Point cloud data saved to PLY file.")

        if points2: 
            save_ply_file('test2_output_x_40_y_10.ply', points2)
            print("Point2 cloud data saved to PLY file.")
        vehicle.destroy()
        lidar.destroy()
        vehicle2.destroy()
        lidar2.destroy()
        print("Vehicle and LiDAR destroyed.")
        print("Simulation terminated gracefully.")

if __name__ == '__main__':
    main()
