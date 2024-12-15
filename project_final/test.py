import carla

client = carla.Client('localhost', 2000)  # Adjust the IP and port as necessary
client.set_timeout(10.0)

# List all available maps
maps = client.get_available_maps()

world = client.load_world("Town05_Opt", carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

nav_point = world.get_random_location_from_navigation() 
if not nav_point:
        print("Navigation data unavailable on this map.")

print("Navigation point:", nav_point)

print(f"Current world map: {world.get_map().name}")
