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
        "siamese_network_reid.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
siamese_net.load_state_dict(siamese_net_weights)
siamese_net.eval()
print("Siamese Network model loaded successfully.")

# Video path
video_path = "output_with_mask.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Video file could not be opened.")
    exit()
else:
    print("Video file opened successfully.")

object_tracker = {}
next_object_id = 0

# Define the transformations for the object region
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),  # Resize as needed
    transforms.ToTensor()
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break

    # Convert frame to a tensor
    frame_tensor = transforms.ToTensor()(frame).unsqueeze(0)

    # Detection portion
    with torch.no_grad():
        detections = feature_extractor(frame_tensor)

    # Access the detection results
    boxes, labels, scores = detections[0]['boxes'], detections[0]['labels'], detections[0]['scores']

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box.tolist())
            # Ensure bounding box coordinates are within frame dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            object_region = frame[y1:y2, x1:x2]
            object_region = cv2.resize(object_region, (800, 800))  # Resize to match expected input size


            # Transform the object region
            object_region_tensor = transform(object_region).unsqueeze(0)  # Shape (1, C, H, W)
            print(f"Bounding box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"Object region shape: {object_region.shape}")
            # Feature extraction from the Faster R-CNN backbone
            with torch.no_grad():
                object_features = feature_extractor(object_region_tensor)

            if isinstance(object_features, list):
                object_features = object_features[0]  # Get the first set of features
            # Print to inspect the type and contents
            print(f"Object features output type: {type(object_features)}")
            print(f"Object features output content: {object_features}")

            # Now check the keys
            if isinstance(object_features, dict) and '0' in object_features:
                feature_tensor = object_features['0']
                print(f"Extracted features shape: {feature_tensor.shape}")
            else:
                print("No valid features found for the object region.")
                continue

            # Pass the feature tensor to the Siamese Network
            try:
                object_features = siamese_net.forward_one(feature_tensor)  # Use forward_one here
            except Exception as e:
                print(f"Error during Siamese Network feature extraction: {e}")
                continue

    

            # Matching and tracking logic
            matched_id = None
            min_distance = float('inf')

            # Compare with existing tracked objects
            for obj_id, features in object_tracker.items():
                distance = F.pairwise_distance(object_features, features).item()
                if distance < min_distance and distance < 0.5:  # 0.5 is the threshold
                    min_distance = distance
                    matched_id = obj_id

            # Update the tracker
            if matched_id is not None:
                object_tracker[matched_id] = object_features
            else:
                object_tracker[next_object_id] = object_features
                matched_id = next_object_id
                next_object_id += 1

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {matched_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break

cap.release()
cv2.destroyAllWindows()
