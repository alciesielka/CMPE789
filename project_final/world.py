import carla
import random
import cv2
import numpy as np
import threading

sensor_data = {'lane_camera':None}
sensor_lock = threading.Lock()
points_spawned = []

def build_world(client):
    # load minimum world
    world = client.load_world("Town02", carla.MapLayer.Buildings)

   # world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    
    map = world.get_map()
    return world, map

def setup_main_vehicle(world, spawn_point, blueprint_name = 'vehicle.tesla.model3'):
  
   blueprint_library = world.get_blueprint_library()
   vehicle_bp = blueprint_library.find(blueprint_name)
   vehicle = world.spawn_actor(vehicle_bp, spawn_point)
   print(f'spawn_point {spawn_point}')

   return vehicle


def setup_vehicle(world, blueprint_name = 'vehicle.tesla.model3', spawn_point = None, autopilot=False):
   global points_spawned
  
   blueprint_library = world.get_blueprint_library()
   vehicle_bp = blueprint_library.find(blueprint_name)

   spawn_points = world.get_map().get_spawn_points()
  
   spawn_point = spawn_point if spawn_point else random.choice(spawn_points)

   points_spawned.append(spawn_point)

   print(f'spawn_point {spawn_point}')
   vehicle = world.spawn_actor(vehicle_bp, spawn_point)

   if autopilot:
        vehicle.set_autopilot(True)
   return vehicle

def setup_traffic_lights(world, duration=10):
    # Get traffic lights and set all / some of them
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')
    for i, light in enumerate(traffic_lights):
        if i < 2:  # Configure only two traffic lights
            light.set_green_time(duration)
            light.set_yellow_time(duration)
            light.set_red_time(duration)
    return traffic_lights


def setup_stop_sign(world):
    blueprint_library = world.get_blueprint_library()


    stop_sign_bp = blueprint_library.find('static.prop.streetsign')
    #stop_sign_bp = blueprint_library.find('static.prop.street_sign.stop')
    
    # Use a random spawn point for the stop sign
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    
    stop_sign = world.spawn_actor(stop_sign_bp, spawn_point)
    return stop_sign

def populate_sensors(img):
    global sensor_data
    global sensor_lock
    with sensor_lock:
        sensor_data['lane_camera'] = img



def camera_callback(data):   
    # every time we get a new image from the camera, run it through yolo
    image_array = np.frombuffer(data.raw_data, dtype=np.uint8)
    image_array = image_array.reshape((data.height, data.width, 4))  # BGRA format
    
    # Convert to BGR for OpenCV (optional: remove alpha channel)
    image_bgr = image_array[:, :, :3]
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    populate_sensors(img)
    # print(f"Image size: Width = {img.shape[1]}, Height = {img.shape[0]}")
    return img

def setup_sensors(world, vehicle):
   global sensor_data
   blueprint_library = world.get_blueprint_library()

   # lane detection camera
   camera_bp = blueprint_library.find('sensor.camera.rgb')
   camera_bp.set_attribute("image_size_x", "640") # 800
   camera_bp.set_attribute("image_size_y", "640") # 600
   camera_transform = carla.Transform(carla.Location(x=1.6, z=2.4))
   camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

   sensor_data['lane_camera'] = camera.listen(camera_callback)

   return sensor_data, camera



def setup_stop_pedestrian(world, sp):
    blueprint_library = world.get_blueprint_library()
    walker_bp_list = blueprint_library.filter('walker.pedestrian.*')
    #controller_bp = blueprint_library.find('controller.ai.walker')

    # Spawn walker
    walker_bp = random.choice(walker_bp_list)
        
    world.spawn_actor(walker_bp, sp)
    print("Spawned walker")

    # Spawn controller
    # controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
    # print("Spawned controller")

    # Start the controller and make the walker move
    #TODO: It might walk
    # try:
    #     controller.start()
    #     # target_location = world.get_random_location_from_navigatgition()
    #     controller.go_to_location(tp.location)
    #     controller.set_max_speed(random.uniform(1.0, 2.5))
    #     print(f"Controller started for walker at {walker.get_location()}")
    # except RuntimeError as e:
    #     print(f"Error during controller start: {e}")
    #     walker.destroy()
    #     controller.destroy()


def setup_peds_rand(world, num_pedestrians=1, min_distance=5.0):
    global points_spawned
    blueprint_library = world.get_blueprint_library()
    walker_bp_list = blueprint_library.filter('walker.pedestrian.*')
    controller_bp = blueprint_library.find('controller.ai.walker')

    sp = carla.Transform(carla.Location(x=25.530020, y=110.549988, z=0.240557))
    spawn_points = world.get_map().get_spawn_points()
    pedestrian_actors = []

    for _ in range(num_pedestrians):

        #spawn_point = world.get_random_location_from_navigation()
        #target_point = random.choice(spawn_points)
        spawn_point = random.choice(spawn_points)
        while( spawn_point not in points_spawned):
            spawn_point = random.choice(spawn_points)
        points_spawned.append(spawn_point)

        # Spawn walker
        walker_bp = random.choice(walker_bp_list)
        
        walker = world.spawn_actor(walker_bp, spawn_point)
        print(f'Spawned walker {_}')
        print(f'spawn_point {spawn_point}')

        # Spawn cameracontroller
        controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
        print(f'Spawned controller{_}')
        pedestrian_actors.append(walker)

        # Start the controller and make the walker move
        #TODO: It might walk
        # try:
        #     controller.start()
        #     target_location = world.get_random_location_from_navigation()
        #     controller.go_to_location(target_point.location)
        #     print(f'target_point location {target_location}')
        #     controller.set_max_speed(random.uniform(1.0, 2.5))
        #     pedestrian_actors.append((walker, controller))
        #     print(f"Controller started for walker at {walker.get_location()}")
        # except RuntimeError as e:
        #     print(f"Error during controller start: {e}")
        #     walker.destroy()
        #     controller.destroy()

    return pedestrian_actors


def set_spectator_view_veh(world, vehicle):
    spectator = world.get_spectator()  
    transform = vehicle.get_transform()  
    
    # Compute a position offset behind and above the vehicle based on its forward vector
    forward_vector = transform.get_forward_vector()
    spectator_location = transform.location - forward_vector * 10  # Move 10 units behind the vehicle
    spectator_location.z += 5  # Raise the spectator 5 units above the vehicle

    spectator_transform = carla.Transform(spectator_location, transform.rotation)

    spectator.set_transform(spectator_transform)
    print("Spectator camera set to follow the vehicle.")


def main():
    global points_spawned
    client = carla.Client('localhost', 2000)
    client.set_timeout(30)
    world, map = build_world(client)

    # car location
    #sp = carla.Transform(carla.Location(x=43.581200, y=-190.137695, z=0.300000))
    sp = carla.Transform(carla.Location(x=76, y=105, z=0.5), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
    points_spawned.append(sp)
    setup_stop_pedestrian(world, sp)
    
    sp = carla.Transform(carla.Location(x=76, y=106, z=0.5), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
    points_spawned.append(sp)
    setup_stop_pedestrian(world, sp)


    sp = carla.Transform(carla.Location(x=76, y=107, z=0.5), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
    points_spawned.append(sp)
    setup_stop_pedestrian(world, sp)

    sp = carla.Transform(carla.Location(x=76, y=108, z=0.5), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
    points_spawned.append(sp)
    setup_stop_pedestrian(world, sp)

    sp = carla.Transform(carla.Location(x=76, y=109, z=0.5), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
    points_spawned.append(sp)
    setup_stop_pedestrian(world, sp)
    
    #sp = carla.Transform(carla.Location(x=100, y=105, z=0.5), carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000))
    sp = carla.Transform(carla.Location(x=117, y=187, z=0.5), carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000))

    points_spawned.append(sp)
    main_veh = setup_main_vehicle(world, sp, 'vehicle.tesla.model3')

    setup_vehicle(world, 'vehicle.audi.tt', autopilot=True)
    setup_vehicle(world, 'vehicle.bmw.grandtourer', autopilot=True)
    
    # traffic lights
    setup_traffic_lights(world, duration=5)

    #setup_peds_rand(world)
    # sensors
    sensors, camera = setup_sensors(world, main_veh)

    print("world created!")
    print(f'Points Spawned: {points_spawned}' )
    return world, main_veh, sensors, map, camera

if __name__ == '__main__':
    main()
