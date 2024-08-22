import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Segmentation_Dataset import *
from DeepLabV3PLUS_model import *
from Utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time


# Set up parameters
test_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\test_images"
test_maskdir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\test_masks"

test_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])

# Initialize Model and Loss Function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepLabV3Plus(num_classes=11).to(device)
loss_fn = nn.CrossEntropyLoss()

# Load the model checkpoint from the last training session
checkpoint_path = r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\Segmentation\Checkpoints Resnet\deeplabv3_epoch_100.pth.tar"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["state_dict"])

# Create the testing dataset
test_dataset = SegmentationDataset(
    images_dir=test_dir,
    masks_dir=test_maskdir, 
    transform=test_transform,
)

# Get data loader for testing
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# Evaluation function
def evaluate_fn(test_loader, model, loss_fn, device, num_classes):
    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    correct_pixels = 0
    total_pixels = 0
    total_inference_time = 0.0
    num_batches = len(test_loader)
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device).float(), y.to(device).long()
            
            # Start timing the inference
            start_time = time.time()
            
            y_pred = model(x)
            
            # End timing the inference
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            
            # Calculate IoU for this batch
            predictions = torch.argmax(y_pred, dim=1)
            iou = calculate_iou(predictions.cpu(), y.cpu(), num_classes)
            test_iou += iou
            
            # Calculate accuracy for this batch
            correct_pixels += (predictions == y).sum().item()
            total_pixels += torch.numel(predictions)

    avg_test_loss = test_loss / num_batches
    avg_test_iou = test_iou / num_batches
    avg_test_accuracy = correct_pixels / total_pixels
    avg_inference_time = total_inference_time / num_batches  # Average inference time per batch
    
    return avg_test_loss, avg_test_iou, avg_test_accuracy, avg_inference_time


def main():
    print(f"Using device: {device}")
    
    # Evaluate the model on the testing dataset
    test_loss, test_iou, test_accuracy, avg_inference_time = evaluate_fn(test_loader, model, loss_fn, device, num_classes=11)
    print(f'Testing Loss: {test_loss:.4f}, Testing IoU: {test_iou:.4f}, Testing Accuracy: {test_accuracy:.4f}, Average Inference Time per Batch: {avg_inference_time:.4f} seconds')


if __name__ == "__main__":
    main()
