import numpy as np
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: pass in training data from other file

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
