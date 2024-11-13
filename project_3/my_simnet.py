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
from my_utility import load_market1501_triplets, plot_loss
from PIL import Image  
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

class Siamese_Network(nn.Module):
    def __init__(self):
        super(Siamese_Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3) # 3
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)

        self.fc_input_size = None  # Placeholder

        self.fc1 = nn.Linear(56448, 256)
        self.fc2 = nn.Linear(256, 256)

    def forward_one(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
   
    def forward(self, input1, input2, input3):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)
        return output1, output2, output3


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = torch.nn.Parameter(torch.tensor(1.0))

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

        anchor_image = Image.open(f"{self.image_folder}/{ anchor_frame}").convert("RGB")
        positive_image = Image.open(f"{self.image_folder}/{ positive_frame}").convert("RGB")
        negative_image = Image.open(f"{self.image_folder}/{ negative_frame}").convert("RGB")
 
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image


# Unused
def load_faster_rcnn(faster_rcnn_path):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    num_classes = 81
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    
    model.load_state_dict(torch.load(faster_rcnn_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    for param in model.parameters():
        param.requires_grad = False
    return model.backbone  


# Used (move to utility)
def load_faster_rcnn2(faster_rcnn_path):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    num_classes = 81 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(faster_rcnn_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    
    for param in model.parameters():
        param.requires_grad = False  # Freeze 

    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = True 
    model.eval()
    return model  # Return entire model



def train_siamese(siamese_net, dataloader, optimizer, criterion, device):
    siamese_net.train()
    total_loss = 0.0
    
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        out_anchor, out_positive, out_negative = siamese_net(anchor, positive, negative)
        # Triplet Loss
        loss = criterion(out_anchor, out_positive, out_negative)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")
    return total_loss / len(dataloader) 


def validate_siamese(siamese_net, dataloader, criterion, device):
    siamese_net.eval() 
    total_loss = 0.0
    with torch.no_grad(): 
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            out_anchor, out_positive, out_negative = siamese_net(anchor, positive, negative)
            loss = criterion(out_anchor, out_positive, out_negative)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss}")
    return average_loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Siamese Network and optimizer
    siamese_net = Siamese_Network().to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(siamese_net.parameters(), lr=1e-4)

    # Prepare triplet data and create DataLoader
    # NEED TO USE MARKET AND NOT MOTS
    data = 'project_3\\bounding_box_train'

    triplet_data = load_market1501_triplets(data) 
    transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])
    # Split data into training and validation sets (80-20 split)
    split_idx = int(0.8 * len(triplet_data))
    train_data = triplet_data[:split_idx]
    val_data = triplet_data[split_idx:]
    
    train_dataset = TripletDataset(train_data, "project_3\\bounding_box_train", transform)
    val_dataset = TripletDataset(val_data, "project_3\\bounding_box_train", transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_batches = len(train_loader)  
    print(f"Number of batches per epoch: {num_batches}")
    train_losses = []
    val_losses = []
    num_epochs = 30

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train_siamese(siamese_net, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        val_loss = validate_siamese(siamese_net, val_loader, criterion, device)
        val_losses.append(val_loss)

        torch.save(siamese_net.state_dict(), f"siamese_network_reid_epoch_margin{epoch+1}.pth")
        print(f"Model saved after epoch {epoch + 1} with validation loss {val_loss:.4f}")
 
    plot_loss(train_losses, val_losses, num_epochs, "Siamese_Net_Loss" )
    print("Training and validation complete.")
