import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

        # Predefined mapping from RGBA to class indices
        self.rgb_to_class = {
            (0, 0, 0, 0): 0,  # Background
            (140, 255, 25, 255): 1,  # Rack
            (0, 0, 0, 255): 2,  # Unlabelled
            (140, 25, 255, 255): 3,  # Crate
            (255, 197, 25, 255): 4,  # Forklift
            (25, 255, 82, 255): 5,  # Iwhub
            (25, 82, 255, 255): 6,  # Dolly
            (255, 25, 197, 255): 7,  # Pallet
            (255, 111, 25, 255): 8,  # Railing
            (226, 255, 25, 255): 9,  # Floor
            (54, 255, 25, 255): 10,  # Stillage
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load corresponding mask
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        mask = np.array(Image.open(mask_path).convert("RGBA"))

        # Convert RGBA mask to class indices
        class_indices = self._mask_to_class_indices(mask)

        # Apply transformations if any
        if self.transform:
            transformed = self.transform(image=image, mask=class_indices)
            image = transformed['image']
            class_indices = transformed['mask']
        else:
            image = transforms.ToTensor()(image)
            class_indices = torch.tensor(class_indices, dtype=torch.int64)

        return image, class_indices

    def _mask_to_class_indices(self, mask):
        h, w, _ = mask.shape
        class_indices = np.zeros((h, w), dtype=np.int64)
        for rgb, class_idx in self.rgb_to_class.items():
            matches = np.all(mask == rgb, axis=-1)
            class_indices[matches] = class_idx
        return class_indices
