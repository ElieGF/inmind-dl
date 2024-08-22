import os
import numpy as np

# Define directories
bboxes_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\test\\Bounding Boxes"
output_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\test\\Bounding Boxes YOLO"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Image dimensions
image_width = 1280
image_height = 720

# Iterate through each numpy file in the bounding box directory
for bbox_file in os.listdir(bboxes_dir):
    if bbox_file.endswith('.npy'):
        input_path = os.path.join(bboxes_dir, bbox_file)
        output_path = os.path.join(output_dir, bbox_file.replace('.npy', '.txt'))

        # Load the numpy array
        bboxes = np.load(input_path)

        # Open the output file
        with open(output_path, 'w') as outfile:
            for bbox in bboxes:
                bbox_semantic_id = int(bbox[0])
                x_min = bbox[1]
                y_min = bbox[2]
                x_max = bbox[3]
                y_max = bbox[4]

                # Convert to YOLO format
                x_center = (x_min + x_max) / 2.0 / image_width
                y_center = (y_min + y_max) / 2.0 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height

                # Write to output file
                outfile.write(f"{bbox_semantic_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Conversion complete!")