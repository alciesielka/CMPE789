# not implemented yet

import cv2
import torch
from my_tracker import load_faster_rcnn, Siamese_Network
from torchvision import transforms
import torch.nn.functional as F
import warnings

# remove security warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# load models and weights
feature_extractor = load_faster_rcnn("best.pth")
feature_extractor.eval().to(device)

siamese_net = Siamese_Network()
siamese_net_weights = torch.load("siamese_network_reid.pth", map_location=torch.device('cuda'))
siamese_net.load_state_dict(siamese_net_weights)
siamese_net.eval().to(device)

# Alex:
# video_path = "project3\\output_with_mask.mp4"

# Tianna:
video_path = "./output_with_mask.mp4"

cap = cv2.VideoCapture(video_path)

object_tracker = {}
next_object_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # put img in tensor of correct size
    frame_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)  # Add batch dimension

    # pass through model
    with torch.no_grad():
        detections = feature_extractor(frame_tensor)

    print(detections)
    boxes, labels, scores = detections["boxes"], detections["labels"], detections["scores"]

    # track over detections
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5: # only look at detections over 50 % confidence

            # get location of object
            x1, y1, x2, y2 = map(int, box)
            object_region = frame[y1:y2, x1:x2]

            # pass object through tracker model
            object_region_tensor = transforms.ToTensor()(object_region).unsqueeze(0).to(device)
            object_features = siamese_net(object_region_tensor)

            # match objects
            matched_id = None
            min_distance = float('inf')

            for obj_id, features in object_tracker.items():
                distance = F.pairwise_distance(object_features, features).item()
                if distance < min_distance and distance < 0.5:  # 0.5 is the threshold
                    min_distance = distance
                    matched_id = obj_id

            if matched_id is not None:
                object_tracker[matched_id] = object_features
            else:
                # normalize object ID between frames
                object_tracker[next_object_id] = object_features
                matched_id = next_object_id
                next_object_id += 1

            # draw annotations
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {matched_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            torch.cuda.empty_cache()
            print("done?")

    # display annotations
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("done?")