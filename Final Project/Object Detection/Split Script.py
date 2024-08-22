import os
import random
import shutil

# Directories
images_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\Images"
bboxes_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\Bounding Boxes"
labels_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\Bounding Box Labels"

train_images_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\train\Images"
train_bboxes_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\train\Bounding Boxes"
train_labels_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\train\Bounding Box Labels"

val_images_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\val\Images"
val_bboxes_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\val\Bounding Boxes"
val_labels_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\val\Bounding Box Labels"

test_images_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\test\Images"
test_bboxes_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\test\Bounding Boxes"
test_labels_dir = r"c:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\dataset\test\Bounding Box Labels"

# Ensure the output directories exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_bboxes_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_bboxes_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_bboxes_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# List all files
image_files = sorted(os.listdir(images_dir))
bbox_files = sorted(os.listdir(bboxes_dir))
label_files = sorted(os.listdir(labels_dir))

# Extract the numeric part to ensure correct matching
image_ids = [f.split('_')[-1].split('.')[0] for f in image_files]
bbox_ids = [f.split('_')[-1].split('.')[0] for f in bbox_files]
label_ids = [f.split('_')[-1].split('.')[0] for f in label_files]

# Make sure that all the ids are matching
assert image_ids == bbox_ids == label_ids, "Mismatch in file IDs!"

# Shuffle and split
combined = list(zip(image_files, bbox_files, label_files))
random.shuffle(combined)

train_size = int(0.85 * len(combined))
val_size = int(0.10 * len(combined))

train_set = combined[:train_size]
val_set = combined[train_size:train_size + val_size]
test_set = combined[train_size + val_size:]

# Helper function to copy files
def copy_files(file_list, src_dir, dest_dir):
    for f in file_list:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, f))

# Copy train set
copy_files([f[0] for f in train_set], images_dir, train_images_dir)
copy_files([f[1] for f in train_set], bboxes_dir, train_bboxes_dir)
copy_files([f[2] for f in train_set], labels_dir, train_labels_dir)

# Copy validation set
copy_files([f[0] for f in val_set], images_dir, val_images_dir)
copy_files([f[1] for f in val_set], bboxes_dir, val_bboxes_dir)
copy_files([f[2] for f in val_set], labels_dir, val_labels_dir)

# Copy testing set
copy_files([f[0] for f in test_set], images_dir, test_images_dir)
copy_files([f[1] for f in test_set], bboxes_dir, test_bboxes_dir)
copy_files([f[2] for f in test_set], labels_dir, test_labels_dir)

print("Dataset split into training, validation, and testing sets successfully!")
