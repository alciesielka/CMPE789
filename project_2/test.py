import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprint_library = world.get_blueprint_library()

# List all vehicle blueprints
vehicles = blueprint_library.filter('vehicle.*')
for vehicle in vehicles:
    print(vehicle.id)


spawn_points = world.get_map().get_spawn_points()
print(f'Number of spawn points: {len(spawn_points)}')
for i, spawn_point in enumerate(spawn_points):
    print(f'Spawn Point {i}: spawn_point.location')