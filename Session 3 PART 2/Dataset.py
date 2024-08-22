import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))
        self.label_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = np.array(Image.open(img_path))  # Assuming images are already in RGB format

        # Load corresponding label
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        labels = np.load(label_path)  # Load the .npy file

        # Extract bounding box attributes
        boxes = []
        labels_list = []
        for label in labels:
            x_min, y_min, x_max, y_max = label[1], label[2], label[3], label[4]
            boxes.append([x_min, y_min, x_max, y_max])
            labels_list.append(label[0])  # bbox_semantic_id

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels_list = torch.tensor(labels_list, dtype=torch.int64)

        # Convert image to tensor (if no transformation applied)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # Convert HWC to CHW

        target = {'boxes': boxes, 'labels': labels_list}
        return image, target

# Example usage
dataset = CustomDataset(images_dir='path_to_images', labels_dir='path_to_npy_labels')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
