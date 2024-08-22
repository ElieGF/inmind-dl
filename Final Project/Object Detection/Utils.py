from Dataset import *
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import ImageDraw


def visualize_image_with_bboxes(image, targets, label_path):

    # Load the label map from the JSON file
    with open(label_path, "r") as f:
        label_map = {int(k): v["class"] for k, v in json.load(f).items()}

    # Convert the image tensor to a NumPy array and transpose it to HWC format for Pillow
    image_np = image.permute(1, 2, 0).numpy()

    # Convert the NumPy array back to a Pillow image
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

    # Draw bounding boxes on the image using ImageDraw from Pillow
    draw = ImageDraw.Draw(image_pil)
    for bbox, label in zip(targets['boxes'], targets['labels']):
        x_min, y_min, x_max, y_max = bbox

        # Ensure that the label is a scalar value
        if isinstance(label, torch.Tensor):
            label_scalar = label.item()
        else:
            label_scalar = label

        # Retrieve the class name from the label map using the scalar label ID
        label_name = label_map.get(label_scalar, str(label_scalar))
        
        # Draw the bounding box and the label
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        draw.text((x_min, y_min), label_name, fill="red")

    # Display the image using Pillow's show method
    image_pil.show()



def visualize_sample_images_with_labels(dataloader, num_images=5):
    
    images_shown = 0

    for images, targets, label_paths in dataloader:
        for i in range(len(images)):
            visualize_image_with_bboxes(images[i], targets[i], label_paths[i])
            images_shown += 1
            if images_shown >= num_images:
                return  # Exit once we've shown the desired number of images



def custom_collate_fn(batch):
    images = []
    targets = []
    label_paths = []
    
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
        label_paths.append(item[2])
    
    # Stack images into a batch
    images = torch.stack(images, dim=0)
    

    return images, targets, label_paths
