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

    def similarity(self, output1, output2):
        return F.cosine_similarity(output1, output2)


def rcnn_feature_extraction(model, image):
    with torch.no_grad():
        features = model.backbone(image)
    return features

# Incomplete .. just thoughts
def train_model(CNN, RCNN, frame):
    with torch.no_grad():
        # Pass frame through RCNN to get detections and features
        features = rcnn_feature_extraction(rcnn, frame)
    
    # need to loop through all features....
        # Pass through the Siamese network
        feature1 = features[0].unsqueeze(0)  
        feature2 = features[1].unsqueeze(0)  
    
        # Generate embeddings for comparison
        embedding1 = CNN.forward_one(feature1)
        embedding2 = CNN.forward_one(feature2)
    
        # Compute similarity between embeddings for tracking association
        similarity_score = CNN.similarity(embedding1, embedding2)
        print("Similarity Score:", similarity_score.item())


if __name__ == '__main__':
    rcnn = fasterrcnn_resnet50_fpn(pretrained=False)
    rcnn.load_state_dict(torch.load("rcnn_weights.pth"))  # Load your trained weights
    rcnn.eval()  # Set RCNN to evaluation mode


    # do stuff here


'''
STEPS:
1) load data -- yes?
2) augment data -- yes?
3) fine tune pretrained model (rcnn) on MOTS? -- next step
4) pass info from rcnn and train siamese network 
5) create tracking pipeline
'''
