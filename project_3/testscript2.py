# not implemented yet

import cv2
import torch
from my_tracker import load_faster_rcnn2, Siamese_Network
from torchvision import transforms
import torch.nn.functional as F


print("Loading Faster R-CNN model...")
feature_extractor = load_faster_rcnn2("best.pth")
feature_extractor.eval()
print("Faster R-CNN model loaded successfully.")

print("Loading Siamese Network model...")
siamese_net = Siamese_Network()
siamese_net_weights = torch.load(
        "siamese_network_reid.pth", map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=True
    )
siamese_net.load_state_dict(siamese_net_weights)
siamese_net.eval()
print("Siamese Network model loaded successfully.")

# video from google drive
video_path = "output_with_mask.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Video file could not be opened.")
    exit()
else:
    print("Video file opened successfully.")

object_tracker = {}
next_object_id = 0

annotated_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break

    # Convert frame to a suitable format
    frame_tensor = transforms.ToTensor()(frame).unsqueeze(0)  

    # Detection PortioN:
    with torch.no_grad():
        detections = feature_extractor(frame_tensor)

    boxes, labels, scores = detections[0]['boxes'], detections[0]['labels'], detections[0]['scores']

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5: 
         
            x1, y1, x2, y2 = map(int, box)
            object_region = frame[y1:y2, x1:x2]

            # Siamese NEtwork here: grabw
            object_region_tensor = transforms.ToTensor()(object_region).unsqueeze(0)

            with torch.no_grad():
                object_features = feature_extractor(object_region_tensor)  # Use the backbone model ?? not sure

            # The output should contain features with the expected channel size
            if '0' in object_features:  # Check the keys in the output
                feature_tensor = object_features['0']  # Get the feature tensor
            else:
                print("No valid features found for the object region.")
                continue

            # Pass the feature tensor to the Siamese Network
            try:
                object_features = siamese_net.forward_one(feature_tensor)  # Use forward_one here
            except Exception as e:
                print(f"Error during feature extraction: {e}")
                continue

            # Matching and tracking logic
            matched_id = None
            min_distance = float('inf')

            # Compare 
            for obj_id, features in object_tracker.items():
                distance = F.pairwise_distance(object_features, features).item()
                if distance < min_distance and distance < 0.5:  # 0.5 is the threshold
                    min_distance = distance
                    matched_id = obj_id

            if matched_id is not None:
                # Update
                object_tracker[matched_id] = object_features
            else:
                # Assign new object ID
                object_tracker[next_object_id] = object_features
                matched_id = next_object_id
                next_object_id += 1

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {matched_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    annotated_frames.append(frame)
    # cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #exit
        break



height, width = annotated_frames.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("tracking_video.mp4", fourcc, 30, (width, height))
for f in annotated_frames:
    out.write(f)
out.release()

cap.release()
cv2.destroyAllWindows()
