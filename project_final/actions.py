from calculate_steering import calculate_steering, calculate_steering_to_waypoint
from yolo import detect_objects
import carla
import math

def plan_action(lane_boundaries, objects, traffic_light_state, current_location, next_waypoint_location, vehicle_heading):
    debug_print = False
    action = {'steer':0.0, 'throttle':0.5, 'brake':0.0}
    print("plan action")

    # Follow lanes
    if lane_boundaries is not None and lane_boundaries.nelement() > 0:  # Ensure lane_boundaries is non-empty
        print("Yes lane boundaries")
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
        for obj in objects:
            # Ensure confidence check produces a Python bool
            if (obj.boxes.conf > 0.2).any().item():  
                print("Object detected")
                # Check for pedestrian or car (class 0 or 1)
                if any([cls in [9, 2] for cls in obj.boxes.cls.int().tolist()]):  
                    print("Pedestrian or car detected")
                    action['throttle'] = 0.0
                    action['brake'] = 1.0
                    
                if any([cls in [3, 4, 5] for cls in obj.boxes.cls.int().tolist()]):
                    if any([cls in [4] for cls in obj.boxes.cls.int().tolist()]):
                        print("YELLOW")
                        action['throttle'] *= 0.5

                    elif any([cls in [3] for cls in obj.boxes.cls.int().tolist()]):
                        print("GREEN")
                        action['throttle'] = 0.5
                        action['brake'] = 0.0
                    else:
                        print("RED")
                        action['throttle'] = 0.0
                        action['brake'] = 1.0

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

