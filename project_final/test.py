import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10)
try:
    world = client.get_world()
    print("Connected to CARLA!")
except RuntimeError as e:
    print(f"Failed to connect: {e}")
