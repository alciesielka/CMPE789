import ultralytics
from ultralytics import YOLO
import torch

import cv2
import torch
import numpy as np
from ultra_fast_lane_detection.model.model import parsingNet
import torchvision.transforms as transforms

def load_model():
    backbone='18'
    model = parsingNet(backbone=backbone, cls_dim=(100+1, 56, 4), use_aux=False).cuda()
    state_dict = torch.load('C:\\Users\\django\\Documents\\Alex\\CMPE789-Github\\CMPE789\\project_final\\culane_18.pth', map_location='cuda')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def preprocess_image(img, img_size=(800, 600)): # may need to swap to 600 x 800
    img = cv2.resize(img, (img_size[1], img_size[0]))
    img = img.astype(np.float32) / 255.0
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    img = transform(img).unsqueeze(0).cuda()
    return img

def run_inference(model, img):
    with torch.no_grad():
        outputs = model(img)
        lane_points = outputs  # Modify based on the model's output structure
        print(lane_points)
    
    return lane_points

def init_yolo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = YOLO('C:\\Users\\django\\Documents\\Alex\\CMPE789-Github\\CMPE789\\train\\weights\\best.pt').to(device)  # 'n' stands for Nano, the smallest YOLOv8 model
    # state_dict = torch.load('C:\\Users\\django\\Documents\\Alex\\CMPE789-Github\\CMPE789\\train\\weights\\best.pt', map_location='cuda')
    # model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device

def detect_objects(image, model, device):
    
    # Perform inference on an image
    results = model(image, device = device)
    print(results)
    return results

def lane_detection(image):
    pass