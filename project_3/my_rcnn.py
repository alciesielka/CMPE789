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
import matplotlib.pyplot as plt

def plot_loss(test_losses, val_losses, epoch):
    epoch = int(epoch)
    x = np.arange(1, epoch+1)
    plt.plot(x, test_losses, 'r--', label = 'test_loss')
    plt.plot(x, val_losses, 'b--', label = 'val_loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()

    plt.savefig("./test_val_loss.png")

def validate(model, val_data_path, image_folder):
    print("validating")
    model.eval()
    total_loss = 0.0
    val_batch_count = 0

    transform = T.ToTensor()

    val_data = parse_gt_file(val_data_path)
    frame_ids = sorted(set(obj['frame_id'] for obj in val_data))
    
    with torch.no_grad():
        for frame_id in frame_ids:
            image, boxes, labels = prepare_data(val_data, image_folder, frame_id) 
            
            image_cuda = [transform(image).to(device)]

            box_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)
            label_tensor = torch.tensor(labels, dtype=torch.int64).to(device)

            targets = [{"boxes" : box_tensor, "labels" : label_tensor}]

            outputs = model(image_cuda)

            if len(outputs) > 0:
                boxes = outputs[0]['boxes']
                labels = outputs[0]['labels']
                scores = outputs[0]['scores']

            else:
                print("No outputs")
                continue

            losses = sum(loss for loss in loss_dict.values())

            val_batch_count += 1

            total_loss += losses.item()
        
        avg_val_loss = total_loss/val_batch_count

    return avg_val_loss
    

if __name__ == '__main__':
    # Tianna
    gt_file_path = ['./MOT16-02/gt/gt.txt', './MOT16-11/gt/gt.txt']  # Path to the MOTS ground truth file
    image_folder = ['./MOT16-02/img1', './MOT16-11/img1']  # Path to the folder containing images

    val_file_path = './MOT16-04/gt/gt.txt'  # Path to the MOTS validation files
    val_image_folder = './MOT16-04/img1'  # Path to the folder containing val images

    # Alex
    # gt_file_path = 'project_3\\MOT16-02\\gt\\gt.txt'  # Path to the MOTS ground truth file
    # image_folder = 'project_3\\MOT16-02\\img1'  # Path to the folder containing images

    # val_file_path = 'project_3\\MOT16-02\\gt\\gt.txt'  # Path to the MOTS validation files
    # val_image_folder = 'project_3\\MOT16-02\\img1'  # Path to the folder containing val images

    # MOTS - 80 classes + 1 background class
    num_classes = 81

    # get pretrained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    for param in model.backbone.parameters():
        param.requires_grad = False

    # Only fine-tune the heads for classification and mask prediction
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(params_to_optimize, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    num_epochs = 5

    # init losses for later calculations
    total_loss = 0.0
    train_batch_count = 0

    # for plotting analysis after training
    test_loss_arr = []
    val_loss_arr = []
    
    for epoch in range(num_epochs):
        model.train()
        model = model.to(device)

        for dataset in range(0, len(gt_file_path)):

            print(gt_file_path[dataset])
            print(image_folder[dataset])
            # load data
            transform = T.ToTensor()
            gt_data = parse_gt_file(gt_file_path[dataset])
            frame_ids = []
            frame_ids = sorted(set(obj['frame_id'] for obj in gt_data))

            for frame_id in frame_ids:
                image, boxes, labels = prepare_data(gt_data, image_folder[dataset], frame_id)
            
                augmented = augment_data(image)
                image_cuda = [T.ToTensor()(augmented).to(device)]

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


        epoch_loss = total_loss / train_batch_count
        
        val_loss = validate(model, val_file_path, val_image_folder)

        test_loss_arr.append(epoch_loss)
        val_loss_arr.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}, Val Loss {val_loss}")

        model_name = "epoch" + str(epoch) + "weights.pth"
        torch.save(model.state_dict(), model_name)

    plot_loss(test_loss_arr, val_loss_arr, num_epochs)


