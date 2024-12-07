from calculate_steering import calculate_steering, calculate_steering_to_waypoint, ultra_fast_lane_detection
from yolo import detect_objects
import carla


def plan_action(lane_boundaries, objects, traffic_light_state, current_location, next_waypoint_location):
    action = {'steer':0.0, 'throttle':0.5, 'brake':0.0}

    # follow lanes
    if lane_boundaries:
        action['steer'] = calculate_steering(lane_boundaries)
    
    # avoid obstacles

    if any([obj.distance < 10 for obj in objects]): # we can adjust threshold
        action['throttle'] = 0.0
        action['brake'] = 1.0

    if traffic_light_state == carla.TrafficLightState.Red:
        action['throttle'] = 0.0
        action['brake'] = 1.0
    
    if traffic_light_state == carla.TrafficLightState.Yellow:
        action['throttle'] *= 0.5

    # steer towards next waypoint
    waypoint_direction = next_waypoint_location - current_location
    action['steer'] = calculate_steering_to_waypoint(waypoint_direction)

    return action

# compute control signals (steer, throttle, break) for vehicle
def compute_control(action):
    control = carla.VehicleControl()
    control.steer = action['steer']
    control.throttle = action['throttle']
    control.brake = action['brake']

    return control

