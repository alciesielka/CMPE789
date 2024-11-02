# not implemented yet

import cv2
import torch
from my_tracker import load_faster_rcnn, Siamese_Network
from torchvision import transforms
import torch.nn.functional as F

# Load your models
feature_extractor = load_faster_rcnn("fasterrcnn_mots_epoch3.pth")
feature_extractor.eval()

siamese_net = Siamese_Network()
siamese_net_weights = torch.load("siamese_network_reid.pth", map_location=torch.device('cuda'), weights_only=True)  # or 'cuda' if available
siamese_net.load_state_dict(siamese_net_weights)
siamese_net.eval()

# Open the video file or webcam
video_path = "project3\\output_with_mask.mp4"  # Change this to your video file
cap = cv2.VideoCapture(video_path)

# Initialize variables for tracking
object_tracker = {}
next_object_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to a suitable format
    frame_tensor = transforms.ToTensor()(frame).unsqueeze(0)  # Add batch dimension

    # Detect objects using Faster R-CNN
    with torch.no_grad():
        detections = feature_extractor(frame_tensor)

    # Assuming detections are in the format [boxes, labels, scores]
    boxes, labels, scores = detections['boxes'], detections['labels'], detections['scores']

    # Iterate over detections
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Confidence threshold
            # Extract the object region
            x1, y1, x2, y2 = map(int, box)
            object_region = frame[y1:y2, x1:x2]

            # Extract features using Siamese Network
            object_region_tensor = transforms.ToTensor()(object_region).unsqueeze(0)
            object_features = siamese_net(object_region_tensor)

            # Matching and tracking logic
            matched_id = None
            min_distance = float('inf')

            # Compare with tracked objects
            for obj_id, features in object_tracker.items():
                distance = F.pairwise_distance(object_features, features).item()
                if distance < min_distance and distance < 0.5:  # 0.5 is the threshold
                    min_distance = distance
                    matched_id = obj_id

            if matched_id is not None:
                # Update existing object
                object_tracker[matched_id] = object_features
            else:
                # Assign new object ID
                object_tracker[next_object_id] = object_features
                matched_id = next_object_id
                next_object_id += 1

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {matched_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
