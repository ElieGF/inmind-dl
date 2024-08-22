import torch
import os
from DeepLabV3PLUS_model import *

# Load the model checkpoint
checkpoint = torch.load(r'C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\Segmentation\Checkpoints Resnet\deeplabv3_epoch_100.pth.tar')

# Load the model and move it to the CPU
model = DeepLabV3Plus(num_classes=11)
model.load_state_dict(checkpoint['state_dict'])
model.to('cpu')  # Ensure the model is on CPU
model.eval()

# Create a dummy input tensor with a dynamic size
input_tensor = torch.randn(1, 3, 512, 512).to('cpu')  # Ensure the input tensor is also on CPU

# Export the model to ONNX with dynamic axes for height and width
onnx_model_path = r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\Segmentation\Dynamic_Segmentation_model.onnx"
torch.onnx.export(
    model, 
    input_tensor, 
    onnx_model_path, 
    opset_version=12, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 
                  'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
)

print(f"Segmentation model has been exported to {onnx_model_path}")
