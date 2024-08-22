import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to the original image and segmentation folders
images_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Images"
segmentation_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic Segmentation"

# Paths for the output train/validation/test folders
train_images_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\train_images"
train_masks_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\train_masks"
val_images_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\val_images"
val_masks_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\val_masks"
test_images_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\test_images"
test_masks_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\test_masks"

# Ensure output directories exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)

# List all image files and corresponding segmentation files
image_files = sorted(os.listdir(images_dir))
mask_files = sorted(os.listdir(segmentation_dir))

# Ensure that images and masks are aligned by their filenames
# Strip out the prefix and extensions
image_ids = [f.replace("rgb_", "").replace(".png", "") for f in image_files]
mask_ids = [f.replace("semantic_segmentation_", "").replace(".png", "") for f in mask_files]

# Check if the lists match (if not, raise an error)
if image_ids != mask_ids:
    raise ValueError("Image files and mask files are not aligned. Please check the filenames.")

# First, split off 5% for the test set
train_val_ids, test_ids = train_test_split(image_ids, test_size=0.05, random_state=42)

# Then, split the remaining 95% into 85% train and 10% val
train_ids, val_ids = train_test_split(train_val_ids, test_size=0.10526, random_state=42)  # 0.10526 * 95% ≈ 10%

# Copy the files to their respective directories
def copy_files(file_ids, images_dir, masks_dir, output_images_dir, output_masks_dir):
    for file_id in file_ids:
        # Copy image
        image_file = f"rgb_{file_id}.png"
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_images_dir, image_file))
        
        # Copy mask
        mask_file = f"semantic_segmentation_{file_id}.png"
        shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(output_masks_dir, mask_file))

# Copy train set
copy_files(train_ids, images_dir, segmentation_dir, train_images_dir, train_masks_dir)

# Copy validation set
copy_files(val_ids, images_dir, segmentation_dir, val_images_dir, val_masks_dir)

# Copy test set
copy_files(test_ids, images_dir, segmentation_dir, test_images_dir, test_masks_dir)

print("Data successfully split into training, validation, and testing sets.")
