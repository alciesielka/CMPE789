# not implemented yet

import cv2
import torch
from my_tracker import load_faster_rcnn2, Siamese_Network
from torchvision import transforms
import torch.nn.functional as F
import os
from PIL import Image
import warnings
import numpy as np

warnings.filterwarnings('ignore')


print("Loading Faster R-CNN model...")
# feature_extractor = load_faster_rcnn2("fasterrcnn_mots_epoch2.pth")
feature_extractor = load_faster_rcnn2("epoch4weights.pth")
feature_extractor.eval()
print("Faster R-CNN model loaded successfully.")

print("Loading Siamese Network model...")
siamese_net = Siamese_Network()
siamese_net_weights = torch.load("siamese_network_reid_epoch3.pth", map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=True)
siamese_net.load_state_dict(siamese_net_weights)
siamese_net.eval()
print("Siamese Network model loaded successfully.")

# video from google drive
# video_path = "output_with_mask.mp4"
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Error: Video file could not be opened.")
#     exit()
# else:
#     print("Video file opened successfully.")

object_tracker = {}
next_object_id = 0
matched_id = None

annotated_frames = []

lower_purple = (120, 50, 50)   
upper_purple = (150, 255, 255) 

frames_dir = 'MOT16-01/img1'

images = os.listdir(frames_dir)
images = sorted(images)

for idx in range(0, len(images)):
    frames_path = os.path.join(frames_dir, images[idx])
    frame = Image.open(frames_path)

    frame = np.array(frame)
    # Convert frame to tensor
    frame_tensor = transforms.ToTensor()(frame).unsqueeze(0)  

    # Detection PortioN:
    with torch.no_grad():
        detections = feature_extractor(frame_tensor)

    boxes, labels, scores = detections[0]['boxes'], detections[0]['labels'], detections[0]['scores']

    for box, label, score in zip(boxes, labels, scores):
        # print(f"Confience score: {score}")
        if score > 0.2: 
         
            x1, y1, x2, y2 = map(int, box)
            object_region = frame[y1:y2, x1:x2]
    
            object_region = cv2.resize(object_region, (100, 100))
            object_region_tensor = transforms.ToTensor()(object_region).unsqueeze(0)

            # Pass the feature tensor to the Siamese Network
            try:
                object_features = siamese_net.forward_one(object_region_tensor)  # Use forward_one here
            except Exception as e:
                print(f"Error during feature extraction: {e}")
                continue

            # Matching IDs
            object_features = F.normalize(object_features, p=2, dim=1)

            # Match and track only for ID 0
            if 0 in object_tracker:
                # Get features of the already tracked object (ID 0)
                tracked_features = F.normalize(object_tracker[0], p=2, dim=1)
                distance = F.pairwise_distance(object_features, tracked_features).item()

                if distance < 0.6:
                    matched_id = 0  # ID 0 is matched
                    object_tracker[0] = object_features  # Update features for ID 0
                else:
                    matched_id = None  # No match for ID 0
            else:
                # Initialize ID 0 if not yet assigned
                object_tracker[0] = object_features
                matched_id = 0
                next_object_id += 1

            # Draw bounding box and ID only for matched ID 0
            if matched_id == 0:
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                frame = cv2.putText(frame, f'ID: {matched_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if matched_id == 0:
        annotated_frames.append(frame)

    annotated_frames.append(frame)
    print(len(annotated_frames))
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #exit
        break


print("saving")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("tracking_video.mp4", fourcc, 30, (1080, 1920))
for f in annotated_frames:
    out.write(f)
out.release()

print("saved")
cv2.destroyAllWindows()
print("success")
