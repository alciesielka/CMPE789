# not implemented yet

import cv2
import torch
from project_3.my_simnet import load_faster_rcnn2, Siamese_Network
from torchvision import transforms
import torch.nn.functional as F
import os
from PIL import Image
import warnings
import numpy as np
warnings.filterwarnings('ignore')


print("Loading Faster R-CNN model...")
feature_extractor = load_faster_rcnn2("fasterrcnn_mots_epoch2.pth")
feature_extractor.eval()
print("Faster R-CNN model loaded successfully.")

print("Loading Siamese Network model...")
siamese_net = Siamese_Network()
siamese_net_weights = torch.load("siamese_network_reid_epoch_margin3.pth", map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=True)
siamese_net.load_state_dict(siamese_net_weights)
siamese_net.eval()
print("Siamese Network model loaded successfully.")


object_tracker = {}
next_object_id = 0
matched_id = None
last_position = None

annotated_frames = []
choose_id = True


frames_dir = 'project_3\\MOT16-01\\img1'
images = os.listdir(frames_dir)
images = sorted(images)

########### CHOOSE PERSON TO TRACK ##################
frame_height, frame_width = None, None

first_frame_path = os.path.join(frames_dir, images[0])
frame = Image.open(first_frame_path)
frame = np.array(frame)
frame_tensor = transforms.ToTensor()(frame).unsqueeze(0)

with torch.no_grad():
    detections = feature_extractor(frame_tensor)

boxes, labels, scores = detections[0]['boxes'], detections[0]['labels'], detections[0]['scores']
id_map = {}
for i, (box, score) in enumerate(zip(boxes, scores)):
    if score > 0.3:
        x1, y1, x2, y2 = map(int, box)
        id_map[i] = (x1, y1, x2, y2)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        frame = cv2.putText(frame, f'ID: {i}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("Select ID", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

selected_id = int(input("Enter the ID of the object you want to track from the displayed bounding boxes: "))
last_position = ((id_map[selected_id][0] + id_map[selected_id][2]) // 2, (id_map[selected_id][1] + id_map[selected_id][3]) // 2)
########### CHOOSE PERSON TO TRACK ##################

# Periodicially save video 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

for idx in range(1, len(images)):
    frames_path = os.path.join(frames_dir, images[idx])
    frame = Image.open(frames_path)
    frame = np.array(frame)
    if frame_height is None or frame_width is None:
        frame_height, frame_width = frame.shape[:2]
        out = cv2.VideoWriter("tracking_video_checkpoint.mp4", fourcc, 30, (frame_width, frame_height))


    frame_tensor = transforms.ToTensor()(frame).unsqueeze(0)  

    with torch.no_grad():
        detections = feature_extractor(frame_tensor)

    boxes, labels, scores = detections[0]['boxes'], detections[0]['labels'], detections[0]['scores']
    
    for box, label, score in zip(boxes, labels, scores):
        #print(f"Confience score: {score}")
        if score > 0.2: 
         
            x1, y1, x2, y2 = map(int, box)
            object_region = frame[y1:y2, x1:x2]
            center_position = ((x1 + x2) // 2, (y1 + y2) // 2)
            object_region = cv2.resize(object_region, (100, 100))
            object_region_tensor = transforms.ToTensor()(object_region).unsqueeze(0)

            # TEST
            if last_position:
                distance_to_last = np.linalg.norm(np.array(center_position) - np.array(last_position))
                if distance_to_last > 30: # position thresh
                    continue  # Skip if too far from last known location
    
        
            # Pass the feature tensor to the Siamese Network
            object_features = siamese_net.forward_one(object_region_tensor)

            # Matching IDs
            with torch.no_grad():
                object_features = F.normalize(object_features, p=2, dim=1)

            # Match and track only for ID 0 
            if selected_id in object_tracker:
                # Get features of the already tracked object (ID 0)
                tracked_features = F.normalize(object_tracker[selected_id], p=2, dim=1)
                distance = F.pairwise_distance(object_features, tracked_features).item()

                if distance < 0.5:
                    matched_id = selected_id  # ID 0 is matched
                    last_position = center_position
                    object_tracker[selected_id] = object_features  # Update features for ID 0
                else:
                    matched_id = None  # No match for ID 0
            elif selected_id not in object_tracker:
                # Initialize ID 0 if not yet assigned
                object_tracker[selected_id] = object_features
                matched_id =selected_id

            if matched_id == selected_id: 
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                frame = cv2.putText(frame, f'ID: {selected_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                break# exit after finding first BB

    annotated_frames.append(frame)

    #print(len(annotated_frames))
    print(f"Processed frame {idx + 1}/{len(images)}")
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #exit
        break

    # Save video every 10 frames
    if idx % 10 == 0:
        print(f"Saving checkpoint at frame {idx}")
        for f in annotated_frames:
            out.write(f)
        annotated_frames = []


print("saving")
for f in annotated_frames:
    out.write(f)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("tracking_video.mp4", fourcc, 30, (1080, 1920))
# for f in annotated_frames:
#     out.write(f)
# out.release()

print("saved")
cv2.destroyAllWindows()
print("success")
