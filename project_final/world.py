import carla
import random


def build_world(client):
    # load minimum world
    world = client.load_world("Town02", carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
    nav_point = world.get_random_location_from_navigation() 
    if not nav_point:
        print("Navigation data unavailable on this map.")

    print("Navigation point:", nav_point)
    
    
    # Toggle Buildings Off
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    # Toggle Buildings On
    world.load_map_layer(carla.MapLayer.Buildings)

    return world

def setup_vehicle(world, blueprint_name = 'vehicle.tesla.model3', spawn_point = None):
  
   blueprint_library = world.get_blueprint_library()
   vehicle_bp = blueprint_library.find(blueprint_name)

   spawn_points = world.get_map().get_spawn_points()
   spawn_point = spawn_point if spawn_point else random.choice(spawn_points)

   vehicle = world.spawn_actor(vehicle_bp, spawn_point)
   return vehicle

def setup_traffic_lights(world, duration=10):
    # Get traffic lights and set all / some of them
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')
    for i, light in enumerate(traffic_lights):
        if i < 2:  # Configure only two traffic lights
            light.set_green_time(duration)
            light.set_yellow_time(duration)
            light.set_red_time(duration)

def setup_peds(world, num_pedestrians=5):
    blueprint_library = world.get_blueprint_library()
    walker_bp_list = blueprint_library.filter('walker.pedestrian.*')
    controller_bp = blueprint_library.find('controller.ai.walker')

    pedestrian_actors = []
    for _ in range(num_pedestrians):
        spawn_point = world.get_random_location_from_navigation()
        if spawn_point:
            # Spawn walker
            walker_bp = random.choice(walker_bp_list)
            walker = world.spawn_actor(walker_bp, carla.Transform(spawn_point))
            if not walker:
                print("Failed to spawn walker")
                continue
            else:
                print("spawned walker")
            
            # Spawn controller
            controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
            if not controller:
                print("Failed to spawn controller")
                walker.destroy()  # Clean up walker if controller fails
                continue
            else:
                print("spawned controller")


            try:
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                controller.set_max_speed(random.uniform(1.0, 2.5))  # Random speed
                pedestrian_actors.append((walker, controller))
            except RuntimeError as e:
                print(f"Controller error: {e}")
                walker.destroy()
                controller.destroy()
                continue
            
            # Start walking
            # controller.start()
            # controller.go_to_location(world.get_random_location_from_navigation())
            # controller.set_max_speed(random.uniform(1.0, 2.5))  # Random speed between 1.0 and 2.5 m/s
            
            # pedestrian_actors.append((walker, controller))
    return pedestrian_actors

    # spawn_point = world.get_random_location_from_navigation()
    # # spawn walker
    # walker = world.spawn_actor(walker_bp, spawn_point)
    # # spawn controller
    # controller = world.spawn_actor(controller_bp, carla.Transform(), walker.id)
    # # start walking
    # controller.start()
    # # set destination
    # controller.go_to_location(destination_point)
    # # set walking speed (in m/s)
    # controller.set_max_speed(speed)
    # # stop walking
    # controller.stop()


def setup_stop_sign(world):
    blueprint_library = world.get_blueprint_library()
    # static.prop.streetsign01
    # static.prop.streetsign04
    stop_sign_bp = blueprint_library.find('static.prop.streetsign01')
    #stop_sign_bp = blueprint_library.find('static.prop.street_sign.stop')
    
    # Use a random spawn point for the stop sign
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    
    stop_sign = world.spawn_actor(stop_sign_bp, spawn_point)
    return stop_sign

def setup_sensors(world, vehicle):
   sensors = {}
   blueprint_library = world.get_blueprint_library()

   # lane detection camera
   camera_bp = blueprint_library.find('sensor.camera.rgb')
   camera_bp.set_attribute("image_size_x", "800")
   camera_bp.set_attribute("image_size_y", "600")
   camera_transform = carla.Transform(carla.Location(x=1.6, z=2.4))
   camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
   sensors['lane_camera'] = camera #sensors['camera'] = camera

   # object detection (can use LIDAR)
   lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
   lidar_bp.set_attribute('range', '50')
   lidar_transform = carla.Transform(carla.Location(x=1.6, z=2.4))
   lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
   sensors['lidar'] = lidar
   return sensors
  
#    collision_bp = self.blueprint_library.find('sensor.other.collision')
#    self.collision_sensor = self.world.spawn_actor(
#        collision_bp,
#        carla.Transform(),
#        attach_to=self.vehicle)

    

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(30)
    world = build_world(client)

    
    main_veh = setup_vehicle(world, 'vehicle.tesla.model3')
    other_veh = [setup_vehicle(world, 'vehicle.audi.tt'),
        setup_vehicle(world, 'vehicle.bmw.grandtourer')]
    
    # traffic lights
    setup_traffic_lights(world, duration=15)

    # pedestrians ( "having issues")
    setup_peds(world, num_pedestrians=2)

    # stop sign
    setup_stop_sign(world)

    # sensors
    sensors = setup_sensors(world, main_veh)

    print("world created!")

if __name__ == '__main__':
    main()
