import os
import albumentations as A
import cv2

# Paths to images and labels
images_path = "C:\\Users\\Elie_\\Desktop\\data\\images"
labels_path = "C:\\Users\\Elie_\\Desktop\\data\\labels"
output_path = "C:\\Users\\Elie_\\Desktop\\data\\output"

# Get list of images and labels
image_files = sorted([f for f in os.listdir(images_path)])
label_files = sorted([f for f in os.listdir(labels_path)])

# Create output directory if it does not exist
os.makedirs(output_path, exist_ok=True)

# Define the augmentation transform
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=130, p=0.5),
    A.Blur(blur_limit=3, p=0.2),  # Adjusted blur limit to a more reasonable value
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.RGBShift(r_shift_limit=200, g_shift_limit=200, b_shift_limit=200, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.5),
], bbox_params=A.BboxParams(format='yolo'))

# Process each image and corresponding label
for img_file, lbl_file in zip(image_files, label_files):
    # Read image
    img_path = os.path.join(images_path, img_file)
    image = cv2.imread(img_path)

    # Open the label file
    lbl_path = os.path.join(labels_path, lbl_file)
    with open(lbl_path, 'r') as file:
        lines = file.readlines()

    # Initialize an empty list to store bounding boxes
    bboxes = []

    # Iterate through each line in the label file
    for line in lines:
        # Split the line by whitespace to get class and coordinates
        data = line.strip().split()
        if len(data) < 5:
            # Skip lines that don't have enough data
            continue
        # Extract the coordinates and convert them to floats
        # class x_center y_center width height
        bbox = [float(coord) for coord in data[1:]]
        bbox.append(data[0])
        # Append the bounding box to the list of bounding boxes
        bboxes.append(bbox)

    # Apply augmentations
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    # Convert the bounding boxes from relative to absolute coordinates
    img_height, img_width = transformed_image.shape[:2]

    # Draw bounding boxes on the augmented image
    for bbox in transformed_bboxes:
        x_center, y_center, width, height, class_id = bbox
        xmin = int((x_center - width / 2) * img_width)
        ymin = int((y_center - height / 2) * img_height)
        xmax = int((x_center + width / 2) * img_width)
        ymax = int((y_center + height / 2) * img_height)
        cv2.rectangle(transformed_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(transformed_image, class_id, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the augmented image with bounding boxes
    output_transformed_img_path = os.path.join(output_path, f'augmented_{img_file}')
    cv2.imwrite(output_transformed_img_path, transformed_image)

print("Processing complete!")