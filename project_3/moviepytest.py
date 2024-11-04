import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms

# Load the model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Assume 'image' is your input image tensor, properly preprocessed.
# Example preprocessing for a single image:
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
])
image = preprocess(image).unsqueeze(0)  # Add batch dimension

# Forward pass through the model
with torch.no_grad():
    detections = model(image)  # Get the output from the model

# Extract the output tensors from the OrderedDict
outputs = detections.values()  # Get the tensor values from the OrderedDict

# Loop through each output tensor and visualize
for idx, output_tensor in enumerate(outputs):
    # Move tensor to CPU and convert to numpy array
    output_array = output_tensor.cpu().detach().numpy()
    
    # If output_tensor has shape (N, C, H, W), process it accordingly
    # Here we take the first output (typically there's a batch dimension)
    output_array = output_array[0]  # Get the first output tensor (N=1)

    # If the tensor has more than 1 channel, transpose it for visualization
    if output_array.shape[0] > 1:
        output_array = output_array.transpose(1, 2, 0)  # Rearrange to (H, W, C)

    # Normalize values to [0, 1] if needed (assuming the range is not [0, 1])
    output_array = (output_array - output_array.min()) / (output_array.max() - output_array.min())

    # Plot the output
    plt.imshow(output_array)
    plt.title(f"Output Tensor {idx}")
    plt.axis('off')  # Turn off axis
    plt.show()
