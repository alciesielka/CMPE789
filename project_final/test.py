import carla


def list_all_blueprints(client):
    # Connect to the Carla world
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Get all blueprints in the library (using an empty filter to list everything)
    blueprints = blueprint_library.filter('stop')

    # Print the ID of each blueprint
    for bp in blueprints:
        print(f"Blueprint ID: {bp.id}")

# Example usage (make sure the Carla server is running and the client is connected)
client = carla.Client('localhost', 2000)  # Change IP/port if needed
list_all_blueprints(client)
