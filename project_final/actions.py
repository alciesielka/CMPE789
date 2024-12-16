from calculate_steering import calculate_steering, calculate_steering_to_waypoint
from yolo import detect_objects
import carla
import math
import time

stop_until = None

def plan_action(lane_boundaries, objects, traffic_light_state, current_location, next_waypoint_location, vehicle_heading):
    global stop_until
    debug_print = False
    action = {'steer':0.0, 'throttle':0.4, 'brake':0.0}
    current_time = time.time()
    print("plan action")


    # Check if the car is in a stop state
    if stop_until is not None:
        if current_time < stop_until:
            # Maintain stop state
            action['throttle'] = 0.0
            action['brake'] = 1.0
            return action
        else:
            # Stop state expired
            stop_until = None
    # Follow lanes
    if lane_boundaries is not None and lane_boundaries.nelement() > 0:  # Ensure lane_boundaries is non-empty
        action['steer'] = calculate_steering(lane_boundaries, current_location, vehicle_heading)

        # Calculate steering angle
        # TODO: What is this used for, why is calculate steering calculted twice
        steering_angle = calculate_steering(
            lane_boundaries=lane_boundaries,
            current_location=current_location,
            vehicle_heading=vehicle_heading
        )    

    # avoid obstacles
    if objects:
        for obj in objects:
            print("Objects detected:")
            print(obj.boxes.cls.int().tolist())
            # Ensure confidence check produces a Python bool
            if (obj.boxes.conf > 0.01).any().item():
                if any([cls in [4, 5] for cls in obj.boxes.cls.int().tolist()]):

                    if any([cls in [3] for cls in obj.boxes.cls.int().tolist()]):
                        print("GREEN")
                        action['throttle'] = 0.4
                        action['brake'] = 0.0
                    
                    elif any([cls in [4] for cls in obj.boxes.cls.int().tolist()]):
                        stop_until =  current_time + 2 
                        print("YELLOW")
                        action['throttle'] = 0.0
                        action['brake'] = 1.0
                        return action

                    else:
                        print("RED")
                        stop_until = current_time + 2
                        action['throttle'] = 0.0
                        action['brake'] = 1.0
                        return action

            if (obj.boxes.conf > 0.95).any().item():  
                # Check for pedestrian or car (class 0 or 1)
                if any([cls in [9, 2] for cls in obj.boxes.cls.int().tolist()]):  
                    if (len(obj.boxes.cls.int().tolist()) < 3):
                        print("Pedestrian or car detected")
                        action['throttle'] = 0.0
                        action['brake'] = 1.0
                        return action

            

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

