from Dataset import *
from Utils import *
from torch.utils.data import DataLoader
from torchvision import transforms
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A


def main():

    images_dir = "c:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Images"
    bboxes_dir = "c:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Bounding Boxes"
    labels_dir = "c:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Bounding Box Labels"

    # Initialize the custom dataset
    dataset = CustomDataset(images_dir=images_dir, bboxes_dir=bboxes_dir, labels_dir=labels_dir)

    # Initialize the DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)

    # Testing the visualization
    visualize_sample_images_with_labels(dataloader, num_images=5)


if __name__=='__main__':
    main()