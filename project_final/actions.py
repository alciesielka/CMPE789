from calculate_steering import calculate_steering, calculate_steering_to_waypoint
from yolo import detect_objects
import carla
import math

def plan_action(lane_boundaries, objects, traffic_light_state, current_location, next_waypoint_location, vehicle_heading):
    debug_print = False
    action = {'steer':0.0, 'throttle':0.5, 'brake':0.0}
    print("plan action")

    # follow lanes
    if lane_boundaries:
        print("yes lane boundaires")
        action['steer'] = calculate_steering(lane_boundaries, current_location, vehicle_heading)
    
        # Calculate steering angle
        steering_angle = calculate_steering(
            lane_boundaries=lane_boundaries,
            current_location=current_location,
            vehicle_heading=vehicle_heading
        )

        print(f"Steering Angle: {math.degrees(steering_angle)} degrees")

    
    # avoid obstacles
    if objects:
        if any([obj.boxes.conf > .2 for obj in objects]): # we can adjust threshold
            print("yes object")
            if any([obj.boxes.cls == 0 or obj.boxes.cls == 1 for obj in objects]):
                print("yes ped or car")
                action['throttle'] = 0.0
                action['brake'] = 1.0
            # Check for traffic Light
            if any([obj.boxes.cls == 0 or obj.boxes.cls == 1 for obj in objects]): 
                action['throttle'] = 0.0
                action['brake'] = 1.0

    # need to isolate traffic light we are closest to!
    # if traffic_light_state == carla.TrafficLightState.Red:
    #     action['throttle'] = 0.0
    #     action['brake'] = 1.0
    #    if debug_print == True:
    #       print("Red Light")
    
    # if traffic_light_state == carla.TrafficLightState.Yellow:
    #     action['throttle'] *= 0.5
    #    if debug_print == True:
    #       print("Yellow Light")

    # steer towards next waypoint
    # waypoint_direction = next_waypoint_location - current_location
    action['steer'] = calculate_steering_to_waypoint(next_waypoint_location, current_location, vehicle_heading)

    return action

# compute control signals (steer, throttle, break) for vehicle
def compute_control(action):
    debug_print = False

    control = carla.VehicleControl()
    control.steer = action['steer']
    control.throttle = action['throttle']
    control.brake = action['brake']

    if debug_print == True:
        print("--"*5)
        print(control)
        print(f"steer: {action['steer']}")
        print(f"throttle: {action['throttle']}")
        print(f"brake: {action['brake']}")

    return control

