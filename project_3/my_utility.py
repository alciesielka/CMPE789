import numpy as np
import torchvision
from pycocotools import mask as mask_utils # pip install pycocotools
from sklearn.model_selection import train_test_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F


# data preparation
def decode_rle(encoded_mask, height, width):
   binary_mask = mask_utils.decode(encoded_mask)
   decoded_mask = np.array(binary_mask).reshape(height, width)
   return decoded_mask

def parse_gt_file(file_path):
   data_arr = []
   data = {
       'time_frame':[],
       'object_id':[],
       'class_id':[],
       'image_height':[],
       'image_width':[],
       'rle':[]
   }
   # may use dictionary? - TJS
   with open(file_path, 'r') as file:
       for line in file:
           # will create a 2D array where each line has its own set 
           data_arr[line] = line.strip().split()
           for item_num in range(0, len(data_arr[line])):
                if item_num != len(data_arr): # may need to do len -1? - TJS
                    data[item_num].append(data_arr[line][item_num])
                else:
                   decoded_mask = decode_rle(data_arr[line][item_num])
                   data[item_num].append(decoded_mask)                       
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