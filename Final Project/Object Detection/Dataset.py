import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, images_dir, bboxes_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.bboxes_dir = bboxes_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))
        self.bbox_files = sorted(os.listdir(bboxes_dir))
        self.label_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load corresponding bounding boxes
        bbox_path = os.path.join(self.bboxes_dir, self.bbox_files[idx])
        bboxes = np.load(bbox_path)

        # Load corresponding label file path
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        with open(label_path, 'r') as f:
            labels = json.load(f)

        boxes = []
        labels_list = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[1], bbox[2], bbox[3], bbox[4]
            boxes.append((x_min, y_min, x_max, y_max))  # Convert to tuple
            labels_list.append(int(bbox[0]))  # Convert to integer

        # Apply transformations if any
        if self.transform:
            # Apply transformation to image only, without bboxes
            transformed = self.transform(image=image)
            image = transformed['image']
            
        else:
            transform = transforms.ToTensor()
            image = transform(image)

        # Convert the boxes and labels back to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels_list = torch.tensor(labels_list, dtype=torch.int64)
        
        target = {'boxes': boxes, 'labels': labels_list}
        return image, target, label_path
