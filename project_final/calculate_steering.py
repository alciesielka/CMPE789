# use waypoint api
import carla
import math


def calculate_steering(lane_boundaries):
    pass

def calculate_steering_to_waypoint(waypoint_detection, current_location, vehicle_heading):
    direction_vector = carla.Vector3D(
        waypoint_detection.transform.location.x - current_location.transform.location.x,
        waypoint_detection.transform.location.y - current_location.transform.location.y
    )

    waypoint_angle = math.atan2(direction_vector.y, direction_vector.x)
    steering_angle = waypoint_angle - vehicle_heading

    # Normalize angle to [-pi, pi]
    steering_angle = (steering_angle + math.pi) % (2 * math.pi) - math.pi

    return steering_angle


def ultra_fast_lane_detection(lane_bound):
    lane_bound