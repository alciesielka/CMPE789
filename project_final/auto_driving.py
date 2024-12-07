from calculate_steering import calculate_steering, calculate_steering_to_waypoint, ultra_fast_lane_detection
from yolo import detect_objects
import carla
import world
from actions import plan_action, compute_control

def camera_callback(data):   
    # every time we get a new image from the camera, run it through yolo
    yolo_detections = detect_objects(data)
    print("------------new frame------------")
    print(yolo_detections)
    
    return yolo_detections


def autonomous_driving(world, vehicle, sensors, destination):
    current_waypoint_index = 0 
    objects = []

    while True:

        # Get Sensor Data
        camera_sensor = sensors['lane_camera'] # may need to pull out of loop - T

        # Procecss Sensor Data        
        camera_sensor.listen(lambda data: camera_callback(data)) # may need to pull out of loop - T

        # Get the Current and Next Waypoint
        current_location = vehicle.get_location()
        start_waypoint = world.get_waypoint(current_location)
        final_waypoint = world.get_waypoint(destination)
        next_waypoint_location = world.waypoint.next(2.0)

        print(f"Starting Route from : {start_waypoint.transform.location}")
        print(f"-> -> -> to : {destination.transform.location}")
        print(f"Next waypoint (2m ahead): {destination.transform.location}")

        lane_boundaries = 1

        # Plan Action (consider depth/distancce for objects on road)
        action = plan_action(lane_boundaries = lane_boundaries, obkjects = objects, traffic_light_state= traffic_light_state, current_location= current_location, next_waypoint_location=next_waypoint_location)
        control_signal = compute_control(action)

        # Execute Control
        vehicle.apply_control(control_signal)

        # Check if Waypoint is Reahed after movement
        if current_location == next_waypoint_location:
            # Checkc if Final Waypoint is Reached
            if current_location == final_waypoint:
                print("Arrived")
                vehicle.destroy()
                break
            else:
                print("Moved 2m")
        else:
            print("ERROR: WAYPOINT NOT LOCATED")


def main(world):
    vehicle = world.main_veh # TODO - implement main vehicle -T
    sensors = world.sensors # TODO - implement sensors -T

    destination = carla.Location(x = 100, y = 100, z = 0)
    autonomous_driving(world, vehicle, sensors, destination)
    


