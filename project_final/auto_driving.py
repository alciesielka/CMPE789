from calculate_steering import calculate_steering, calculate_steering_to_waypoint, ultra_fast_lane_detection
from yolo import detect_objects
import carla
import world
from world import set_spectator_view_veh
import math
from actions import plan_action, compute_control
import numpy as np
import cv2


def autonomous_driving(world, carla_map, vehicle, sensors, destination):
    global sensor_data
    sensor_data = {}
    debug_prints = True

    current_waypoint_index = 0 
    objects = []
    # Get Sensor Data
    camera_sensor = sensors['lane_camera'] # may need to pull out of loop - T

    
    while True:
        set_spectator_view_veh(world, vehicle)
        print(f"sensor data {sensor_data}")

        if 'lane_camera' in sensor_data and sensor_data['lane_camera'] is not None:
            print("is not None!")
            lane_image = sensor_data['lane_camera']
            objects = detect_objects(lane_image)

        # Get the Current and Next Waypoint
        current_location = vehicle.get_location()
        start_waypoint = carla_map.get_waypoint(current_location)
        final_waypoint = carla_map.get_waypoint(destination)
        # next_waypoint_location = carla_map.waypoint.next(2.0)

        next_waypoint = start_waypoint.next(2.0)[0]  # Get the first waypoint 2 meters ahead
        next_waypoint_location = next_waypoint.transform.location

        if debug_prints == True:
            print(f"Starting Route from : {current_location}")
            print(f"-> -> -> to : {final_waypoint}")
            print(f"Next waypoint (2m ahead): {next_waypoint}")

        current_location = start_waypoint
        vehicle_heading = math.radians(vehicle.get_transform().rotation.yaw)


        lane_boundaries = False
        traffic_light_state = False

        # Plan Action (consider depth/distancce for objects on road)
        action = plan_action(lane_boundaries = lane_boundaries, objects = objects, traffic_light_state= traffic_light_state, current_location= current_location, next_waypoint_location=next_waypoint_location, vehicle_heading =  vehicle_heading)
        control_signal = compute_control(action)

        # Execute Control
        vehicle.apply_control(control_signal)

        # Check if Waypoint is Reahed after movement
        if (current_location.transform.location.y <= next_waypoint_location.transform.location.y+1) or (current_location.transform.location.y >= next_waypoint_location.transform.location.y-1):
            # Checkc if Final Waypoint is Reached
            if (current_location.transform.location.x <= final_waypoint.transform.location.x+1) or (current_location.transform.location.x >= final_waypoint.transform.location.x-1):
                if (current_location.transform.location.y <= final_waypoint.transform.location.y+1) or (current_location.transform.location.y >= final_waypoint.transform.location.y-1):
                    print("Arrived")
                    vehicle.destroy()
                    break
                else:
                    print("Moved 2m")  
            else:
                print("Moved 2m")
    
        
        if (current_location.transform.location.x <= next_waypoint_location.transform.location.x+1) or (current_location.transform.location.x >= next_waypoint_location.transform.location.x-1):
            # Checkc if Final Waypoint is Reached
            if (current_location.transform.location.x <= final_waypoint.transform.location.x+1) or (current_location.transform.location.x >= final_waypoint.transform.location.x-1):
                if (current_location.transform.location.y <= final_waypoint.transform.location.y+1) or (current_location.transform.location.y >= final_waypoint.transform.location.y-1):
                    print("Arrived")
                    vehicle.destroy()
                    break
                else:
                    print("Moved 2m")  
            else:
                print("Moved 2m")
    

def main(world, carla_map, vehicle, sensors):
    destination = carla.Location(x = 100, y = 100, z = 0)
    autonomous_driving(world, carla_map, vehicle, sensors, destination)
    

if __name__ == "__main__":
    carla_world, vehichle, sensors, map = world.main()
    main(carla_world, map, vehichle, sensors)


#TODO: implement lane detection; fix sensor callback; fix traffic light; seems like waypoints are not being reached? updating too fast?