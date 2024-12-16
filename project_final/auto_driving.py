from calculate_steering import calculate_steering, calculate_steering_to_waypoint
from yolo import detect_objects, init_yolo, load_model, run_inference, preprocess_image_ufld
import carla
import world
from world import set_spectator_view_veh
import math
import torch
from actions import plan_action, compute_control
import numpy as np
import cv2
from world import sensor_data
import random

def autonomous_driving(world, carla_map, vehicle, sensors, destination, camera):
    global sensor_data
    debug_prints = False
    model, device = init_yolo()
    lane_model = load_model()
    #current_waypoint_index = 0 
    objects = []
    set_spectator_view_veh(world, vehicle)
    while True:
       # camera.listen(camera_callback)
       # set_spectator_view_veh(world, vehicle)

        if 'lane_camera' in sensor_data and sensor_data['lane_camera'] is not None:
            lane_image = sensor_data['lane_camera']
            
            objects = detect_objects(lane_image, model, device)
            lane_img_pp = preprocess_image_ufld(lane_image)
            lane_boundaries = run_inference(lane_model, lane_img_pp)        
           

        # Get the Current and Next Waypoint
        current_location = vehicle.get_location()
        start_waypoint = carla_map.get_waypoint(current_location)
        final_waypoint = carla_map.get_waypoint(destination)

        next_waypoint = start_waypoint.next(3.0)[0]  # Get the first waypoint 2 meters ahead
        next_waypoint_location = next_waypoint.transform.location

        if debug_prints:
            print(f"Starting Route from : {current_location}")
            print(f"-> -> -> to : {final_waypoint}")
            print(f"Next waypoint: {next_waypoint_location}")

        vehicle_heading = math.radians(vehicle.get_transform().rotation.yaw)

        #lane_boundaries = False
        traffic_light_state = False

        # Plan Action
        action = plan_action(
            lane_boundaries=lane_boundaries,
            objects=objects,
            traffic_light_state=traffic_light_state,
            current_location=current_location,
            next_waypoint_location=next_waypoint_location,
            vehicle_heading=vehicle_heading
        )
        control_signal = compute_control(action)

        # Execute Control
        vehicle.apply_control(control_signal)

        # Check if Waypoint is Reached after movement
        print(f'current_location {current_location.x}, {current_location.y}')
        print(f'destination {destination.x}, {destination.y}')
        if abs(current_location.x - next_waypoint_location.x) <= 10 or abs(current_location.y - next_waypoint_location.y) <= 10:
            if abs(current_location.x - destination.x) <= 10 and abs(current_location.y - destination.y) <= 10:
                print("Arrived")
                print(f'current_location {current_location.x}, {current_location.y}')
                print(f'destination {destination.x}, {destination.y}')
                vehicle.destroy()
                world.destroy()
                break
            else:
                print("Moved 3m")

def main(world, carla_map, vehicle, sensors, camera):
    spawn_points = world.get_map().get_spawn_points()
    destination = random.choice(spawn_points).location
    #sp = carla.Transform(carla.Location(x=43.581200, y=-190.137695, z=0.300000))

   
    destination = carla.Location(x=76, y = 105, z = 0)
    #destination = carla.Location(x=100, y=100, z=0)
    autonomous_driving(world, carla_map, vehicle, sensors, destination, camera)

if __name__ == "__main__":
    carla_world, vehicle, sensors, carla_map, camera = world.main()
    main(carla_world, carla_map, vehicle, sensors, camera)

# TODO: fix vehicle spawn


# TODO: Run rain and fog and see how it does!!!
# TODO: Potentially run 60 signs as stop signs.
