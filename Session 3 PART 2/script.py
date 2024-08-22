import os
import shutil
import random

# Paths to the training images and masks
train_dir = 'C:\\Users\\Elie_\\Desktop\\train'
train_maskdir = 'C:\\Users\\Elie_\\Desktop\\train_masks'

# Paths for the validation images and masks
val_dir = 'C:\\Users\\Elie_\\Desktop\\val'
val_maskdir = 'C:\\Users\\Elie_\\Desktop\\val_masks'

# Create validation directories if they don't exist
os.makedirs(val_dir, exist_ok=True)
os.makedirs(val_maskdir, exist_ok=True)

# Get list of image and mask files
image_files = sorted(os.listdir(train_dir))
mask_files = sorted(os.listdir(train_maskdir))

# Ensure the number of images and masks are the same
assert len(image_files) == len(mask_files), "The number of images and masks must be the same."

# Determine number of validation samples (10% of the total)
num_val = len(image_files) // 10

# Randomly select indices for validation samples
val_indices = random.sample(range(len(image_files)), num_val)

# Move the selected files to the validation directories
for idx in val_indices:
    img_name = image_files[idx]
    mask_name = mask_files[idx]

    # Move the image
    src_img_path = os.path.join(train_dir, img_name)
    dest_img_path = os.path.join(val_dir, img_name)
    shutil.move(src_img_path, dest_img_path)

    # Move the mask
    src_mask_path = os.path.join(train_maskdir, mask_name)
    dest_mask_path = os.path.join(val_maskdir, mask_name)
    shutil.move(src_mask_path, dest_mask_path)

print(f'Moved {num_val} images and masks to validation directories.')
