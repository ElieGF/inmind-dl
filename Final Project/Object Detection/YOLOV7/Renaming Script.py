import os
import shutil

# Directories where images and YOLO files are located
images_dir = r"c:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\val\\Images"
yolo_files_dir = r"c:\\Users\\Elie_\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\val\\Bounding Boxes YOLO"

# List all image files
image_files = sorted(os.listdir(images_dir))

# Iterate over all YOLO files and copy/rename them
for image_file in image_files:
    # Extract the numeric part from the image filename
    numeric_part = image_file.split('_')[-1].split('.')[0]
    
    # Original YOLO filename
    original_yolo_file = f"bounding_box_2d_tight_{numeric_part}.txt"
    
    # Check if the YOLO file exists
    original_yolo_file_path = os.path.join(yolo_files_dir, original_yolo_file)
    if os.path.exists(original_yolo_file_path):
        # New YOLO filename to match the image filename
        new_yolo_file = f"{image_file.split('.')[0]}.txt"
        
        # Destination path for the copied and renamed YOLO file
        new_yolo_file_path = os.path.join(images_dir, new_yolo_file)
        
        # Copy and rename the YOLO file
        shutil.copy(original_yolo_file_path, new_yolo_file_path)
        print(f"Copied and renamed: {original_yolo_file} to {new_yolo_file}")
    else:
        print(f"YOLO file not found for image: {image_file}")

print("Copying and renaming complete!")
