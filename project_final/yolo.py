import ultralytics
from ultralytics import YOLO


def detect_objects(image):
    print("NEW IMAGE")
    # Load the YOLOv8 Tiny model
    model = YOLO("yolov8n.pt")  # 'n' stands for Nano, the smallest YOLOv8 model

    # Perform inference on an image
    results = model(image)

    # Show the results
    results.show()
    print(results)
    return results

def lane_detection(image):
    pass