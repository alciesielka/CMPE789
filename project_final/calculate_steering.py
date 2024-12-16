# use waypoint api
import carla
import math

def calculate_steering(lane_boundaries, current_location, vehicle_heading):
    #print("calculating lane steering")
    
    # Check if lane_boundaries is valid
    if lane_boundaries is None or len(lane_boundaries) < 2:
        #print("One or both lane boundaries missing. Steering calculation skipped.")
        return 0.0  # Default to no steering if lane boundaries are incomplete

    # Safely extract left and right lane points
    left_lane_points = lane_boundaries[0] if lane_boundaries[0] is not None else None
    right_lane_points = lane_boundaries[1] if lane_boundaries[1] is not None else None

    if left_lane_points is None or right_lane_points is None:
        #print("One or both lane boundaries missing. Steering calculation skipped.")
        return 0.0

    # Calculate lane center points
    lane_center_points = [
        carla.Location(
            x=(left_point[0] + right_point[0]) / 2,
            y=(left_point[1] + right_point[1]) / 2
        )
        for left_point, right_point in zip(left_lane_points, right_lane_points)
    ]

    if not lane_center_points:
        #print("No valid lane center points. Steering calculation skipped.")
        return 0.0

    # Choose a target point ahead of the vehicle
    target_index = min(5, len(lane_center_points) - 1)  # 5 points ahead or the last point
    target_point = lane_center_points[target_index]

    direction_vector = carla.Vector3D(
        target_point.x - current_location.x,
        target_point.y - current_location.y
    )

    # Compute the desired angle to the target point
    desired_angle = math.atan2(direction_vector.y, direction_vector.x)
    steering_angle = desired_angle - vehicle_heading

    # Normalize the steering angle to [-pi, pi]
    steering_angle = (steering_angle + math.pi) % (2 * math.pi) - math.pi

    return steering_angle


def calculate_steering_to_waypoint(waypoint_detection, current_location, vehicle_heading):
    #print("calculate waypoint steering")
    direction_vector = carla.Vector3D(
        waypoint_detection.x - current_location.x,
        waypoint_detection.y - current_location.y
    )

    waypoint_angle = math.atan2(direction_vector.y, direction_vector.x)
    steering_angle = waypoint_angle - vehicle_heading

    # Normalize angle to [-pi, pi]
    steering_angle = (steering_angle + math.pi) % (2 * math.pi) - math.pi

    return steering_angle


    