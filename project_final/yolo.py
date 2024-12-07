import ultralytics
from ultralytics import YOLO


def detect_objects(image):
    # Load the YOLOv8 Tiny model
    model = YOLO("yolov8n.pt")  # 'n' stands for Nano, the smallest YOLOv8 model

    # Perform inference on an image
    results = model(image)

    # Show the results
    results.show()
    print(results)
    return results

def detect_lane(image):
    # TODO: Add Ultra fast lane detection - T
    pass