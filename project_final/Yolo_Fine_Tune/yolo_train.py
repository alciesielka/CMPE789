import os
import world
import numpy as np
import carla
import random
import threading
import os
import numpy as np

sensor_data = {'lane_camera': None}
sensor_lock = threading.Lock()


def save_image_and_annotations(image, objects, output_dir):
    image_width, image_height = image.width, image.height

    # Ensure output directories exist
    image_dir = os.path.join(output_dir, "images")
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Save image
    image_file = os.path.join(image_dir, f"{image.frame}.png")
    image.save_to_disk(image_file)

    # Save annotations
    annotation_file = os.path.join(label_dir, f"{image.frame}.txt")
    with open(annotation_file, 'w') as f:
        for obj in objects:
            class_id = obj['class_id']
            x_min, y_min, x_max, y_max = obj['bbox']
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def detect_objects(image):
    # Simulate detection of pedestrians and traffic lights
    detected_objects = [
        {'class_id': 0, 'bbox': (100, 150, 200, 300)},  # Example pedestrian
        {'class_id': 1, 'bbox': (250, 300, 400, 450)},  # Example traffic light
    ]
    return detected_objects

def main(camera, output_dir, test_veh):
    # Attach the camera to save images and annotations
    camera.listen(lambda image: save_image_and_annotations(image, detect_objects(image), output_dir))

if __name__ == "__main__":
    # Setup Carla world and simulation
    carla_world, main_veh, sensors, carla_map, camera, test_veh = world.main()

    # Define the output directory for dataset
    output_dir = "dataset"

    # Run the main process
    # test_veh is the one to follow
    main(camera, output_dir, test_veh)
