import cv2
import torch
from my_tracker import load_faster_rcnn2, Siamese_Network
from torchvision import transforms
import os
from PIL import Image
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# Load Faster R-CNN model
print("Loading Faster R-CNN model...")
feature_extractor = load_faster_rcnn2("fasterrcnn_mots_epoch2.pth")
feature_extractor.eval()
print("Faster R-CNN model loaded successfully.")

# Set directory and list of images
frames_dir = 'project_3\\MOT16-01\\img1'
images = sorted(os.listdir(frames_dir))
annotated_frames = []



choose_id = True
# Loop through each image in the directory
for idx in range(len(images)):
    frames_path = os.path.join(frames_dir, images[idx])
    frame = Image.open(frames_path)
    frame = np.array(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV display

    # Convert frame to tensor
    frame_tensor = transforms.ToTensor()(frame).unsqueeze(0)

    # Perform detection
    with torch.no_grad():
        detections = feature_extractor(frame_tensor)

    boxes, labels, scores = detections[0]['boxes'], detections[0]['labels'], detections[0]['scores']

    # Draw all bounding boxes with a confidence score > 0.5
    for box, score in zip(boxes, scores):
        if score > 0.3:
            x1, y1, x2, y2 = map(int, box)
            # Draw bounding box
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # Display confidence score
            frame = cv2.putText(frame, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            

    annotated_frames.append(frame)
    print(f"Processed frame {idx + 1}/{len(images)}")

    # Show the current frame
    cv2.imshow("Frame", frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the output video
print("Saving video...")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Get frame dimensions dynamically
frame_height, frame_width = annotated_frames[0].shape[:2]
out = cv2.VideoWriter("tracking_video.mp4", fourcc, 30, (frame_width, frame_height))

for f in annotated_frames:
    out.write(f)
out.release()

print("Video saved successfully.")
cv2.destroyAllWindows()
print("Success")
