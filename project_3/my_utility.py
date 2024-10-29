#from pycocotools import mask as mask_utils # pip install pycocotools
from torchvision import transforms
from PIL import Image
import random
import json
import numpy as np
    
def parse_gt_file(file_path):
    gt_data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split the line by commas and strip whitespace
            parts = line.strip().split(',')
            # Ensure there are at least 8 elements in the line
            if len(parts) < 8:
                continue  # Skip incomplete lines

            frame_id = int(parts[0])  # Frame number
            obj_id = int(parts[1])     # ID
            bb_left = int(parts[2])     # BB left
            bb_top = int(parts[3])      # BB top
            bb_width = int(parts[4])    # BB width
            bb_height = int(parts[5])   # BB height
            conf = float(parts[6])       # Confidence

            # X and Y coordinates, need to hgandle float
            x = float(parts[7]) if len(parts) > 7 else 0.0  
            y = float(parts[8]) if len(parts) > 8 else 0.0  

            gt_data.append({
                'frame_id': frame_id,
                'object_id': obj_id,
                'bb_left': bb_left,
                'bb_top': bb_top,
                'bb_width': bb_width,
                'bb_height': bb_height,
                'conf': conf,  # FLOAT!
                'x': x,
                'y': y
            })
    return gt_data

def save_gt_data(gt_data, filename="gt_data.json"):
    with open(filename, 'w') as f:
        json.dump(gt_data, f)

def load_gt_data(filename="gt_data.json"):
    with open(filename, 'r') as f:
        return json.load(f)


def prepare_data(gt_data, image_folder, frame_id):
    image_path = f"{image_folder}/{str(frame_id).zfill(6)}.jpg"
    image = Image.open(image_path).convert("RGB")

    # Extract objects for the specific frame
    frame_objects = [obj for obj in gt_data if obj['frame_id'] == frame_id]

    boxes = []
    labels = []
    for obj in frame_objects:
        # Extract bounding box coordinates
        xmin = obj['bb_left']
        ymin = obj['bb_top']
        xmax = xmin + obj['bb_width']
        ymax = ymin + obj['bb_height']
        boxes.append([xmin, ymin, xmax, ymax])

        labels.append(obj['object_id'])
    
    boxes = np.array(boxes)
    return image, boxes, labels

def augment_data(original_image):
    augmentation = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=(original_image.height, original_image.width), scale=(0.8, 1.0))
    ])
    
    augmented_image = augmentation(original_image)
    return augmented_image


def prepare_triplet_data(gt_data):
    triplets = []
    
    for obj in gt_data:
        anchor_frame = obj['frame_id']
        anchor_id = obj['object_id']

        positives = [o for o in gt_data if o['frame_id'] == anchor_frame and o['object_id'] == anchor_id]
        negatives = [o for o in gt_data if o['frame_id'] == anchor_frame and o['object_id'] != anchor_id]

        # triplets
        for pos in positives:
            if negatives:  # Ensure there are negatives to choose from
                neg = negatives[np.random.choice(len(negatives))]
                triplet = {
                    'anchor': (anchor_frame, anchor_id),
                    'positive': (pos['frame_id'], pos['object_id']),
                    'negative': (neg['frame_id'], neg['object_id'])
                }
                triplets.append(triplet)
    
    return triplets