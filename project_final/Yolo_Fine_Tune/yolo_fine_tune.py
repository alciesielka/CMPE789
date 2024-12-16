import torch
from ultralytics import YOLO

def init_yolo(model_path="yolov8n.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_path).to(device)
    return model, device

def train_yolo(model, data_yaml, epochs=50, batch_size=16, img_size=640):

    print(f"Starting training for {epochs} epochs...")
    model.train(
        data=data_yaml,       # Path to dataset YAML
        epochs=epochs,        # Number of epochs
        batch=batch_size,     # Batch size
        imgsz=img_size,       # Image size
        workers=4             # Number of data loading workers
    )
    print("Training completed.")

def validate_yolo(model, data_yaml):
    print("Running validation...")
    metrics = model.val(data=data_yaml)
    print(metrics)

if __name__ == "__main__":
    # Initialize YOLO model
    model_path = "yolov8n.pt"  # Pre-trained YOLOv8 Nano model
    model, device = init_yolo(model_path)

    # Path to the data YAML configuration file
    data_yaml = "project_final\Yolo_Fine_Tune\data.yaml"

    # Fine-tune the model
    train_yolo(model, data_yaml, epochs=3, batch_size=16, img_size=640)

    # Validate the model
    # can remove if needed
    validate_yolo(model, data_yaml)


# How data needs to be organized: 

# /dataset
#   /images
#     /train
#       img1.jpg
#       img2.jpg
#       ...
#     /val
#       img1.jpg
#       img2.jpg
#       ...
#   /labels
#     /train
#       img1.txt
#       img2.txt
#       ...
#     /val
#       img1.txt
#       img2.txt
#       ...

# .txt files are organized: 
# <class_id> <x_center> <y_center> <width> <height>
#  x_center, y_center, width, and height are normalized between 0 and 1 relative to the image dimensions.
