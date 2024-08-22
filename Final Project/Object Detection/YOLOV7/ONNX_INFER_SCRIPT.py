import onnxruntime as ort
import numpy as np
import cv2
import time

# Load the ONNX model
onnx_model_path = r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\YOLOV7\yolov7\yolov7_model.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Use GPU first, fall back to CPU if necessary
ort_session = ort.InferenceSession(onnx_model_path, providers=providers)

# Print the input names for the model
input_names = [input.name for input in ort_session.get_inputs()]

# Prepare the input image
image_path = r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\test\Images\rgb_0165.png"
img = cv2.imread(image_path)
img = cv2.resize(img, (640, 640))
img = img.transpose(2, 0, 1)  # Change to CHW format
img = img[np.newaxis, :, :, :].astype(np.float32)

# Use the correct input name 
input_feed = {input_names[0]: img}  

# Run inference
start_time = time.time()
outputs = ort_session.run(None, input_feed)
end_time = time.time()

print(f"Inference time for YOLOv7 ONNX: {end_time - start_time:.4f} seconds")

# Print the outputs
print("Model Outputs:")
for i, output in enumerate(outputs):
    print(f"Output {i}: {output.shape}, {output}")
