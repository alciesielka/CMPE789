import carla
import random
import cv2
import numpy as np
import threading

sensor_data = {'lane_camera':None}
sensor_lock = threading.Lock()

def build_world(client):
    # load minimum world
    world = client.load_world("Town05_Opt", carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
    nav_point = world.get_random_location_from_navigation() 
    if not nav_point:
        print("Navigation data unavailable on this map.")

    print("Navigation point:", nav_point)
    
    
    # Toggle Buildings Off
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    # Toggle Buildings On
    world.load_map_layer(carla.MapLayer.Buildings)
    map = world.get_map()

    return world, map

def setup_vehicle(world, blueprint_name = 'vehicle.tesla.model3', spawn_point = None, autopilot= False):
  
   blueprint_library = world.get_blueprint_library()
   vehicle_bp = blueprint_library.find(blueprint_name)

   spawn_points = world.get_map().get_spawn_points()
   spawn_point = spawn_point if spawn_point else random.choice(spawn_points)

   vehicle = world.spawn_actor(vehicle_bp, spawn_point)

   if (autopilot):
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
    #print("recieved new frame")
    populate_sensors(img)
    return img

def setup_sensors(world, vehicle):
   global sensor_data
   blueprint_library = world.get_blueprint_library()

   # lane detection camera
   camera_bp = blueprint_library.find('sensor.camera.rgb')
   camera_bp.set_attribute("image_size_x", "800")
   camera_bp.set_attribute("image_size_y", "600")
   camera_transform = carla.Transform(carla.Location(x=1.6, z=2.4))
   camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

   sensor_data['lane_camera'] = camera.listen(camera_callback)

#    # object detection (can use LIDAR)
#    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
#    lidar_bp.set_attribute('range', '50')
#    lidar_transform = carla.Transform(carla.Location(x=1.6, z=2.4))
#    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
#    sensor_data['lidar'] = lidar
   return sensor_data, camera
  
#    collision_bp = self.blueprint_library.find('sensor.other.collision')
#    self.collision_sensor = self.world.spawn_actor(
#        collision_bp,
#        carla.Transform(),
#        attach_to=self.vehicle)


def setup_pedestrian(world, sp, tp):
    blueprint_library = world.get_blueprint_library()
    walker_bp_list = blueprint_library.filter('walker.pedestrian.*')
    controller_bp = blueprint_library.find('controller.ai.walker')

    # Spawn walker
    walker_bp = random.choice(walker_bp_list)
        
    walker = world.spawn_actor(walker_bp, sp)
    print("Spawned walker")

    # Spawn controller
    controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
    print("Spawned controller")

    # Start the controller and make the walker move
    #TODO: It might walk
    try:
        controller.start()
        # target_location = world.get_random_location_from_navigatgition()
        controller.go_to_location(tp.location)
        controller.set_max_speed(random.uniform(1.0, 2.5))
        print(f"Controller started for walker at {walker.get_location()}")
    except RuntimeError as e:
        print(f"Error during controller start: {e}")
        #walker.destroy()
        controller.destroy()

    return walker


def setup_peds_rand(world, num_pedestrians=1, min_distance=5.0):
    blueprint_library = world.get_blueprint_library()
    walker_bp_list = blueprint_library.filter('walker.pedestrian.*')
    controller_bp = blueprint_library.find('controller.ai.walker')

    sp = carla.Transform(carla.Location(x=25.530020, y=110.549988, z=0.240557))

    saved_spawn_points = [carla.Transform(carla.Location(x=193.779999, y=293.540009, z=0.500000),
                            carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000))
                            
                        ]
    
    target_spawn_points = [carla.Transform(carla.Location(x=193.779999, y=293.540009, z=0.500000),
                            carla.Rotation(pitch=0.000000, yaw=180, roll=0.000000))
                            
                        ]

    spawn_points = world.get_map().get_spawn_points()
    pedestrian_actors = []

    for _ in range(num_pedestrians):
        spawn_point = random.choice(spawn_points)
        target_point = random.choice(spawn_points)
        random_location = world.get_random_location_from_navigation()

        print(f'random_location {random_location}')
        
        # Spawn walker
        walker_bp = random.choice(walker_bp_list)

        
        #walker = world.spawn_actor(walker_bp, saved_spawn_points[0])\
        walker = world.spawn_actor(walker_bp, carla.Transform(random_location))

        set_spectator_view_veh(world, walker)
        print("Spawned walker")

        controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
        # Start the controller and make the walker move
        #TODO: It might walk
        try:
            controller.start()
            print("controller started")
            #random_location = world.get_random_location_from_navigation()
            #controller.go_to_location(random_location)
            print("go_to_location")
            controller.set_max_speed(random.uniform(1.0, 2.5))
            print("set_max_speed")
            pedestrian_actors.append((walker, controller))
            print(f"Controller started for walker at {walker.get_location()}")

            while True:
                # Pick a random location within the map
                random_location = world.get_random_location_from_navigation()
                print(f'random_location {random_location}')

                if random_location:
                    # Move the walker to this random location
                    controller.go_to_location(random_location)
                    print(f"Walker moving to random location: {random_location}")

                # Wait before moving to a new random location
                #time.sleep(5) 
        except Exception  as e:
            print(f"Error during controller start: {e}")
            walker.destroy()
            controller.destroy()
    return pedestrian_actors



def setup_spectator_view(world, target_actor=None):
    # Get the spectator camera
    spectator = world.get_spectator()

    if target_actor:
        target_location = target_actor.get_location()

        location = carla.Location(x=target_location.x, y=target_location.y, z=target_location.z + 10)  # 10 meters above the target
        rotation = carla.Rotation(pitch=-30, yaw=0)  # Slight tilt to view the actor better
        spectator.set_transform(carla.Transform(location, rotation))
        print(f"Spectator view set above the stop sign at location: {location}")
    else:
        # default view of the town (above)
        location = carla.Location(x=0, y=0, z=100)  # Height of 100 meters above the town
        rotation = carla.Rotation(pitch=-90)  # Looking straight down at the town

        # Apply the transformation
        spectator.set_transform(carla.Transform(location, rotation))
        print("Spectator view set above the town.")


def clear_world(world):
    # Get all actors in the world
    actors = world.get_actors()

    for actor in actors:
        # Exclude the spectator (if necessary)
        if actor.type_id == 'spectator':
            continue
        # Destroy the actor
        actor.destroy()

    print("All actors have been cleared.")

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
    client = carla.Client('localhost', 2000)
    client.set_timeout(30)
    world, map = build_world(client)

    blueprint_library = world.get_blueprint_library()
  

    # car location
    #sp = carla.Transform(carla.Location(x=20.530020, y=105.549988, z=0.240557))

    main_veh = setup_vehicle(world, 'vehicle.tesla.model3')


    #TODO: will need there own spawn points
    test_veh = setup_vehicle(world, 'vehicle.audi.tt', autopilot=True)   

    # other_veh = [setup_vehicle(world, 'vehicle.audi.tt', autopilot=True),
    #    setup_vehicle(world, 'vehicle.bmw.grandtourer', autopilot=True)]
    
    # traffic lights
    traffic_ligts = setup_traffic_lights(world, duration=5)

    # pedestrians ( "having issues")
    # sp = carla.Transform(carla.Location(x=25.530020, y=110.549988, z=0.240557))
    # tp = carla.Transform(carla.Location(x=40, y=75.549988, z=0.240557))
    # walker = setup_pedestrian(world, sp, tp)

    #pedestrian_actors = setup_peds_rand(world)


    # stop sign
    sign = setup_stop_sign(world)

    # sensors
    sensors, camera = setup_sensors(world, test_veh)

    print("world created!")



    try:
        while True:
            # Run the world tick (update the simulation state)
            world.tick()           

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        # Clean up and destroy all actors when the simulation ends
        clear_world(world)
        print("World and actors cleared.")

    
    return world, main_veh, sensors, map, camera, test_veh

if __name__ == '__main__':
    main()
