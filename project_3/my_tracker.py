import numpy as np
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese_Network(nn.Module):
    def __init__(self):
        super(Siamese_Network, self).__init__()
       
        # CNN layers for feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 22 * 22, 256)
        self.fc2 = nn.Linear(256, 256)
def forward_one(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
   
def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


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

# fine tune a pre-trained Faster R-CNN
# Freeze backbone layers
def fine_tune(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Only fine-tune the heads for classification and mask prediction
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    return params_to_optimize



if __name__ == '__main__':
    pass
    # do stuff here