import carla


def build_world():
    # load minimum world
    world = client.load_world("Town02_Opt", carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
    
    # Toggle Buildings Off
    world.unload_map_layer(carla.MapLayer.Buildings)

    # Toggle Buildings On
    world.load_map_layer(carla.MapLayer.Buildings)

    return world

def setup_vehicle(world, blueprint_name = 'vehicle.tesla.model3'):
   blueprint_library = world.get_blueprint_library()
   vehicle_bp = blueprint_library.find(blueprint_name)
   spawn_point = world.get_map().get_spawn_points()[0]
   vehicle = world.spawn_actor(vehicle_bp, spawn_point)
   return vehicle

def setup_traffic_lights(world, duration):
    # Get traffic lights and set all / some of them
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')
    light = traffic_lights[i] # fix i

    light.set_green_time(duration)
    light.set_yellow_time(duration)
    light.set_red_time(duration)

def setup_peds(world, destination_point, speed):
    walker_bp = world.get_blueprint_library().filter('walker.pedestrian.*')
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')

    spawn_point = world.get_random_location_from_navigation()
    # spawn walker
    walker = world.spawn_actor(walker_bp, spawn_point)
    # spawn controller
    controller = world.spawn_actor(controller_bp, carla.Transform(), walker.id)
    # start walking
    controller.start()
    # set destination
    controller.go_to_location(destination_point)
    # set walking speed (in m/s)
    controller.set_max_speed(speed)
    # stop walking
    controller.stop()

def setup_sensors(world, vehicle):
   sensors = {}
   blueprint_library = world.get_blueprint_library()

   # lane detection camera
   camera_bp = blueprint_library.find('sensor.camera.rgb')
   camera_bp.set_attribute("image_size_x", "800")
   camera_bp.set_attribute("image_size_y", "600")
   camera_transform = carla.Transform(carla.Location(x=1.6, z=2.4))
   camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
   sensors['lane_camera'] = camera

   # object detection (can use LIDAR)
   lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
   lidar_bp.set_attribute('range', '50')
   lidar_transform = carla.Transform(carla.Location(x=1.6, z=2.4))
   lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
   sensors['lidar'] = lidar
  
   collision_bp = self.blueprint_library.find('sensor.other.collision')
   self.collision_sensor = self.world.spawn_actor(
       collision_bp,
       carla.Transform(),
       attach_to=self.vehicle)

