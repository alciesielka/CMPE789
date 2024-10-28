import numpy as np
import torchvision
import torch.optim as optim
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_utility import parse_gt_file, prepare_data, augment_data


if __name__ == '__main__':

    gt_file_path = './MOT16-02/gt/gt.txt'  # Path to the MOTS ground truth file
    image_folder = './MOT16-02/img1'  # Path to the folder containing images

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    num_classes = 70

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    for param in model.backbone.parameters():
        param.requires_grad = False

    # Only fine-tune the heads for classification and mask prediction
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(params_to_optimize, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_epochs = 3

    transform = T.ToTensor()


    gt_data = parse_gt_file(gt_file_path)

    frame_ids = sorted(set(obj['frame_id'] for obj in gt_data))
    total_loss = 0.0
    train_batch_count = 0
    
    for epoch in range(num_epochs):
        model.train()
        model = model.to(device)
        for frame_id in frame_ids:
        
            # image will be the same for each pass of frame_id, iterate through bboxes

            image, boxes, labels = prepare_data(gt_data, image_folder, frame_id)
        
            image_cuda = [transform(image).to(device)]

            box_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)
            label_tensor = torch.tensor(labels, dtype=torch.int64).to(device)
            targets = [{"boxes" : box_tensor, "labels" : label_tensor}]

            loss_dict = model(image_cuda, targets)

            optimizer.zero_grad()

            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            train_batch_count += 1

            torch.cuda.empty_cache()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / train_batch_count}")
