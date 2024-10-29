import numpy as np
import torchvision
import torch.optim as optim
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from my_utility import prepare_triplet_data, parse_gt_file
from PIL import Image  

class Siamese_Network(nn.Module):
    def __init__(self):
        super(Siamese_Network, self).__init__()
       
        # CNN layers for feature extraction
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3) # 256, 64?
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)

        self.fc1 = None
        #self.fc1 = nn.Linear(128 * 22 * 22, 256)
        self.fc2 = nn.Linear(256, 256)

    def forward_one(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))

        if self.fc1 is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(flattened_size, 256).to(x.device)

        x = x.view(x.size(0), -1)  #  x = x.view(-1, 128 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
   
    def forward(self, input1, input2, input3):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)
        return output1, output2, output3


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        return loss

class TripletDataset(Dataset):
    def __init__(self, triplet_data, image_folder, transform=None):
        self.triplet_data = triplet_data
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.triplet_data)

    def __getitem__(self, idx):
        triplet = self.triplet_data[idx]
        anchor_frame, anchor_id = triplet['anchor']
        positive_frame, positive_id = triplet['positive']
        negative_frame, negative_id = triplet['negative']

        anchor_image = Image.open(f"{self.image_folder}/{str(anchor_frame).zfill(6)}.jpg").convert("RGB")
        positive_image = Image.open(f"{self.image_folder}/{str(positive_frame).zfill(6)}.jpg").convert("RGB")
        negative_image = Image.open(f"{self.image_folder}/{str(negative_frame).zfill(6)}.jpg").convert("RGB")

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image


def load_faster_rcnn(faster_rcnn_path):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 81
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(faster_rcnn_path))
    for param in model.parameters():
        param.requires_grad = False
    return model.backbone  


def train_siamese(siamese_net, feature_extractor, dataloader, optimizer, criterion, device):
    siamese_net.train()
    feature_extractor.eval()
    
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        # Extract features from Faster R-CNN backbone
        with torch.no_grad():
            anchor_feat = feature_extractor(anchor)["0"]
            positive_feat = feature_extractor(positive)["0"]
            negative_feat = feature_extractor(negative)["0"]
        
        # Forward pass through Siamese Network
        out_anchor, out_positive, out_negative = siamese_net(anchor_feat, positive_feat, negative_feat)
        
        # Compute Triplet Loss
        loss = criterion(out_anchor, out_positive, out_negative)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the feature extractor
    feature_extractor = load_faster_rcnn("fasterrcnn_mots_epoch3.pth")
    feature_extractor.to(device)

    # Initialize Siamese Network and optimizer
    siamese_net = Siamese_Network().to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(siamese_net.parameters(), lr=1e-4)

    # Prepare triplet data and create DataLoader
    gt_file_path = 'project_3\\MOT16-02\\gt\\gt.txt'
    gt_data = parse_gt_file(gt_file_path)
    triplet_data = prepare_triplet_data(gt_data) 
    transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])  # Adjust size as necessary
    triplet_dataset = TripletDataset(triplet_data, "project_3\\MOT16-02\\img1", transform)
    dataloader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)

    # Train the Siamese Network
    for epoch in range(1):
        print(f"Epoch {epoch + 1}")
        train_siamese(siamese_net, feature_extractor, dataloader, optimizer, criterion, device)
    
    # Save the trained Siamese Network
    torch.save(siamese_net.state_dict(), "siamese_network_reid.pth")
    print("Siamese Network model saved.")