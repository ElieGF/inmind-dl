import torch
import cv2  
import numpy as np
from DeepLabV3PLUS_model import *
import os
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = DeepLabV3Plus(num_classes=11).to(device)
checkpoint = torch.load("C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\Segmentation\\Checkpoints Resnet\\deeplabv3_epoch_100.pth.tar")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Load the image
image_path = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\test_images\\rgb_0235.png"  
image = cv2.imread(image_path) 
original_name = os.path.basename(image_path)

# Convert image to tensor
image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # Change to CHW format and add batch dimension

# Time the inference
start_time = time.time()

# Perform inference
with torch.no_grad():
    output = model(image_tensor)
    output = torch.argmax(output, dim=1).cpu().numpy()  # Get the predicted class for each pixel

end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")


# Convert to color image
output = output.squeeze()  # Remove batch dimension
output_colored = np.zeros((output.shape[0], output.shape[1], 4), dtype=np.uint8)  # Create an empty RGBA image

# Map each class to an RGBA color
class_to_rgba = {
    0: (0, 0, 0, 0),  # Background
    1: (140, 255, 25, 255),  # Rack
    2: (0, 0, 0, 255),  # Unlabelled
    3: (140, 25, 255, 255),  # Crate
    4: (255, 197, 25, 255),  # Forklift
    5: (25, 255, 82, 255),  # Iwhub
    6: (25, 82, 255, 255),  # Dolly
    7: (255, 25, 197, 255),  # Pallet
    8: (255, 111, 25, 255),  # Railing
    9: (226, 255, 25, 255),  # Floor
    10: (54, 255, 25, 255),  # Stillage
}

# Apply the color mapping
for class_idx, rgba_color in class_to_rgba.items():
    mask = output == class_idx
    output_colored[mask] = rgba_color

# Save the resulting image
output_image_path = f"C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\Segmentation\\Inference Results\\Segmented {original_name}"
cv2.imwrite(output_image_path, output_colored)

print(f"Segmented image saved to {output_image_path}")
