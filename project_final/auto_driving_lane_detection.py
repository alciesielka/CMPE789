from calculate_steering import calculate_steering, calculate_steering_to_waypoint, ultra_fast_lane_detection
from yolo import detect_objects
import carla
import torch
import world
from world import set_spectator_view_veh
import math
from actions import plan_action, compute_control
import numpy as np
import cv2

from ultra_fast_lane_detection.model import UltraFastLaneDetector

lane_model = UltraFastLaneDetector('culane_18.pth', device='cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_lane_image(image):
    resized = cv2.resize(image, (800, 288))
    normalized = resized / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
    return tensor

def detect_lanes(image, model):
    preprocessed = preprocess_lane_image(image)
    with torch.no_grad():
        lane_predictions = model(preprocessed)
    return lane_predictions


def autonomous_driving(world, carla_map, vehicle, sensors, destination):
    global sensor_data
    sensor_data = {}
    debug_prints = True

    current_waypoint_index = 0 
    objects = []

    while True:
        set_spectator_view_veh(world, vehicle)
        print(f"sensor data {sensor_data}")

        if 'lane_camera' in sensor_data and sensor_data['lane_camera'] is not None:
            print("is not None!")
            lane_image = sensor_data['lane_camera']
            objects = detect_objects(lane_image)

            # lane Detection
            lane_data = detect_lanes(lane_image, lane_model)
            lane_boundaries = ultra_fast_lane_detection.process_output(lane_data)

            # debug lane visualization (doesn't load here)
            debug_lane_image = ultra_fast_lane_detection.visualize_lanes(lane_image, lane_boundaries)
            cv2.imshow('Lane Detection', debug_lane_image)
            cv2.waitKey(1)


        # get current and next waypoint
        current_location = vehicle.get_location()
        start_waypoint = carla_map.get_waypoint(current_location)
        final_waypoint = carla_map.get_waypoint(destination)

        # get the waypoint 2 meters ahead
        next_waypoint = start_waypoint.next(2.0)[0] 
        next_waypoint_location = next_waypoint.transform.location

        if debug_prints:
            print(f"Starting Route from : {current_location}")
            print(f"-> -> -> to : {final_waypoint}")
            print(f"Next waypoint (2m ahead): {next_waypoint_location}")

        vehicle_heading = math.radians(vehicle.get_transform().rotation.yaw)
        traffic_light_state = False

        # plan Action
        action = plan_action(
            lane_boundaries=lane_boundaries,
            objects=objects,
            traffic_light_state=traffic_light_state,
            current_location=current_location,
            next_waypoint_location=next_waypoint_location,
            vehicle_heading=vehicle_heading
        )
        control_signal = compute_control(action)

        # execute Control
        vehicle.apply_control(control_signal)

        # check if Waypoint is Reached after movement
        if abs(current_location.x - next_waypoint_location.x) <= 1 and abs(current_location.y - next_waypoint_location.y) <= 1:
            if abs(current_location.x - destination.x) <= 1 and abs(current_location.y - destination.y) <= 1:
                print("Arrived")
                vehicle.destroy()
                break
            else:
                print("Moved 2m")

def main(world, carla_map, vehicle, sensors):
    destination = carla.Location(x=100, y=100, z=0)
    autonomous_driving(world, carla_map, vehicle, sensors, destination)

if __name__ == "__main__":
    carla_world, vehicle, sensors, carla_map = world.main()
    main(carla_world, carla_map, vehicle, sensors)

# TODO: implement lane detection; fix sensor callback; fix traffic light; seems like waypoints are not being reached? updating too fast?
