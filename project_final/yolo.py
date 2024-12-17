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

def preprocess_image_yolo_center_crop(img, img_size=(600, 200)):
    # find center of box
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    crop_x1 = max(center_x - img_size[1] // 2, 0)
    crop_x2 = min(center_x + img_size[1] // 2, img.shape[1])
    crop_y1 = max(center_y - img_size[0] // 2, 0)
    crop_y2 = min(center_y + img_size[0] // 2, img.shape[0])

    # crop and  resize
    img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    img = cv2.resize(img, (img_size[1], img_size[0]))

    # normalize and transform
    img = img.astype(np.float32) / 255.0
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    img = transform(img).unsqueeze(0).cuda()
    return img


def preprocess_image_yolo(img, img_size=(600, 100)):
    img = cv2.resize(img, (img_size[1], img_size[0]))
    img = img.astype(np.float32) / 255.0
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    img = transform(img).unsqueeze(0).cuda()
    return img

def preprocess_image_ufld(img, img_size=(800, 288)):
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
        lane_points = outputs
        print(lane_points)
    
    return lane_points

def init_yolo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load in our fine tuned yolo weights
    model = YOLO('C:\\Users\\django\\Documents\\Alex\\CMPE789-Github\\CMPE789\\train\\weights\\best.pt', verbose=False).to(device)
    model.eval()
    return model, device

def detect_objects(image, model, device):
    preprocess_image_yolo_center_crop(image)

    results = model(image, verbose=False, device = device)
    print("results")
    return results

def lane_detection(image):
    pass