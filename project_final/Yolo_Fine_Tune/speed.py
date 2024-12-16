from ultralytics import YOLO

# Load original model
original_model = YOLO("yolov8n.pt")
# Load fine-tuned model
fine_tuned_model = YOLO("runs\\detect\\train2\\weights\\best.pt")

# Compare inference speeds
import time
img_path = "project_final\\Yolo_Fine_Tune\\train\images\\001357_png.rf.5b9e805023fe27cf99a9d3f14b1b8319.jpg"

start = time.time()
original_model(img_path)
print(f"Original model inference time: {time.time() - start:.2f} seconds")

start = time.time()
fine_tuned_model(img_path)
print(f"Fine-tuned model inference time: {time.time() - start:.2f} seconds")