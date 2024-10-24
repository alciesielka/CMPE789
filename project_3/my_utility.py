import numpy as np
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F


# data preparation
def decode_rle(encoded_mask, height, width):
   decoded_mask = 0
   pass
   return decoded_mask

def parse_gt_file(file_path):
   data = []
   with open(file_path, 'r') as f:
       pass
   return data

# data augmentation
def augment_data(original_image):
    color_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ])

    augmented_image = color_aug(original_image)
    return augmented_image

def split_data():
    pass

def main():
    pass

if __name__ == "__main__":
    main()