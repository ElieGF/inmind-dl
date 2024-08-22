import onnxruntime as ort
import numpy as np
import cv2
import time

# Load the ONNX model
onnx_model_path = r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\Segmentation\Dynamic_Segmentation_model.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Specify GPU provider first
ort_session = ort.InferenceSession(onnx_model_path, providers=providers)

# Prepare the input image 
image_path = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\test_images\\rgb_0235.png"
original_img = cv2.imread(image_path)

# Convert image to tensor format without resizing
img = original_img.transpose(2, 0, 1)  # Change to CHW format
img = img[np.newaxis, :, :, :].astype(np.float32)

# Run inference
start_time = time.time()
outputs = ort_session.run(None, {'input': img})
end_time = time.time()

print(f"Inference time: {end_time - start_time:.4f} seconds")

# Post-process the output to get the segmentation mask
output = np.argmax(outputs[0], axis=1).squeeze()

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

# Create an empty RGBA image with original dimensions
output_colored = np.zeros((output.shape[0], output.shape[1], 4), dtype=np.uint8)

# Apply the color mapping
for class_idx, rgba_color in class_to_rgba.items():
    mask = output == class_idx
    output_colored[mask] = rgba_color

# Save the output image
output_image_path = r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\Segmentation\Inference Results\onnx_inference_output.png"
cv2.imwrite(output_image_path, output_colored)

print(f"Output image saved to {output_image_path}")
