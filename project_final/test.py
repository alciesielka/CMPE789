import carla

def list_traffic_blueprints(world):
    # Get the blueprint library from the world
    blueprint_library = world.get_blueprint_library()

    # List all traffic-related blueprints by filtering with 'traffic'
    traffic_blueprints = blueprint_library.filter('traffic')

    # Print out the ID and type of each traffic-related blueprint
    for blueprint in traffic_blueprints:
        print(f"ID: {blueprint.id}, Type: {blueprint.type_id}")

# Example usage (this assumes you have an active Carla client connection)
client = carla.Client('localhost', 2000)  # Connect to Carla server
client.set_timeout(10)  # Optional timeout in seconds for the connection
world = client.get_world()  # Get the world

# Now call the function to list traffic-related blueprints
list_traffic_blueprints(world)
