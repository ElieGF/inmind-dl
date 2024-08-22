import torch
import os
from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
import cv2
from fastapi.responses import Response
from typing import List, Dict
from fastapi.responses import FileResponse
import shutil
import glob

app = FastAPI()

# Define a dictionary to map class IDs to names
class_id_to_name = {
    0: "Background",
    1: "Rack",
    2: "Unlabelled",
    3: "Crate",
    4: "Forklift",
    5: "Iwhub",
    6: "Dolly",
    7: "Pallet",
    8: "Railing",
    9: "Floor",
    10: "Stillage",
}


# Define the directory where images will be saved
output_dir = r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\temp output images"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Load the ONNX models
segmentation_model = ort.InferenceSession("Segmentation/Dynamic_Segmentation_model.onnx")

# Model Listing Endpoint
@app.get("/models", response_model=List[str])
async def list_models():
    models = ["YOLOv7 Object Detection", "DeepLabV3+ Semantic Segmentation"]
    return models

# Segmentation Inference Endpoint
@app.post("/segmentation")
async def segmentation_inference(image: UploadFile = File(...)):
    # Read and process the input image
    file = await image.read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :].astype(np.float32)

    # Run inference with the model
    output = segmentation_model.run(None, {'input': img})
    output = np.argmax(output[0], axis=1).squeeze()

    # Create an empty RGBA image to hold the color-mapped output
    output_img = np.zeros((output.shape[0], output.shape[1], 4), dtype=np.uint8)

    # Class to Color Mapping
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

    # Map each class to its RGBA color
    for class_idx, rgba_color in class_to_rgba.items():
        mask = output == class_idx
        output_img[mask] = rgba_color

    # Encode the output image to PNG format
    _, buffer = cv2.imencode('.png', output_img)
    response_img = buffer.tobytes()

    # Return the image as a response
    return Response(content=response_img, media_type="image/png")

# Bounding Box Inference Endpoint (Returning JSON)
@app.post("/bbox-json")
async def bbox_inference_json(image: UploadFile = File(...)) -> List[Dict]:
    # Read the image file
    image_bytes = await image.read()
    image_path = os.path.join(output_dir, "input_image.png")
    
    # Save the input image temporarily
    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    # Load the image to get its dimensions
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Run the YOLOv7 inference using the detect.py script
    yolo_dir = r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\YOLOV7\yolov7"
    os.chdir(yolo_dir)

    # Save the output to a results JSON file
    os.system(f'python detect.py --weights runs/train/yolov7-w6-custom/weights/best.pt --img 640 --conf 0.4 --source "{image_path}" --save-txt --save-conf')

    # Find the most recent 'exp' directory
    results_dir = max(glob.glob(os.path.join(yolo_dir, 'runs', 'detect', 'exp*', 'labels')), key=os.path.getmtime)

    # Read the results from the latest output directory
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]

    bboxes = []

    if result_files:
        with open(os.path.join(results_dir, result_files[0]), 'r') as file:
            lines = file.readlines()
            for line in lines:
                elements = line.strip().split()
                
                # Parsing the YOLO output
                class_id, confidence, x_center, y_center, width, height = map(float, elements[:6])
                
                # Convert from YOLO format to pixel coordinates
                x_min = max(int((x_center - width / 2) * img_width), 0)
                y_min = max(int((y_center - height / 2) * img_height), 0)
                x_max = min(int((x_center + width / 2) * img_width), img_width)
                y_max = min(int((y_center + height / 2) * img_height), img_height)
                
                class_name = class_id_to_name.get(int(class_id), "Unknown")  # Get class name from the dictionary
                bboxes.append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "class": class_name,
                    "Confidence_Score": confidence
                })

    return bboxes

# Bounding Box Inference Endpoint (Returning Image)
@app.post("/bbox-image")
async def bbox_inference_image(image: UploadFile = File(...)) -> FileResponse:
    # Read the image file
    image_bytes = await image.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Save the input image temporarily
    input_image_path = os.path.join(output_dir, "input_image.png")
    cv2.imwrite(input_image_path, img)

    # Run the YOLOv7 inference using the detect.py script
    yolo_dir = r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\YOLOV7\yolov7"
    os.chdir(yolo_dir)
    os.system(f'python detect.py --weights runs/train/yolov7-w6-custom/weights/best.pt --img 640 --conf 0.25 --source "{input_image_path}"')

    # Find the most recently created exp directory
    exp_dirs = sorted(glob.glob("runs/detect/exp*"), key=os.path.getmtime, reverse=True)
    latest_exp_dir = exp_dirs[0]

    # Get the path to the output image generated by detect.py
    output_image_path = os.path.join(yolo_dir, latest_exp_dir, "input_image.png")

    # Change back to the original directory
    os.chdir(r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind")

    # Return the output image
    return FileResponse(output_image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)