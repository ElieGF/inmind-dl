import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Segmentation_Dataset import *
from DeepLabV3PLUS_model import *
from Utils import *
from torch.utils.tensorboard import SummaryWriter

# Set up parameters
train_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\train_images"
train_maskdir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\train_masks"
val_dir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\val_images"
val_maskdir = "C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\dataset\\Semantic\\val_masks"
EPOCHS = 100

train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=130, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    ToTensorV2(),
])

val_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])

# Initialize Model, Optimizer, and Loss Function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepLabV3Plus(num_classes=11).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\Segmentation\tensorboard_logs_resnet")

# Set up the train function
def train_fn(train_loader, model, optimizer, loss_fn, device, num_classes):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device).float(), y.to(device).long()  # Convert images to float32 and labels to int64
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Calculate IoU for this batch
        predictions = torch.argmax(y_pred, dim=1)
        iou = calculate_iou(predictions.cpu(), y.cpu(), num_classes)
        running_iou += iou
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")

    avg_loss = running_loss / len(train_loader)
    avg_iou = running_iou / len(train_loader)
    return avg_loss, avg_iou


# Set up the validation function
def validate_fn(val_loader, model, loss_fn, device, num_classes):
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device).float(), y.to(device).long()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
            
            # Calculate IoU for this batch
            predictions = torch.argmax(y_pred, dim=1)
            iou = calculate_iou(predictions.cpu(), y.cpu(), num_classes)
            val_iou += iou

    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    return avg_val_loss, avg_val_iou


# Main function
def main():
    
    print(f"Using device: {device}")
    
    # Create the datasets
    train_dataset = SegmentationDataset(
        images_dir=train_dir,
        masks_dir=train_maskdir,  
        transform=train_transform,
    )
    
    val_dataset = SegmentationDataset(
        images_dir=val_dir,
        masks_dir=val_maskdir,  
        transform=val_transform,
    )

    # Get data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # Training loop
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        train_loss, train_iou = train_fn(train_loader, model, optimizer, loss_fn, device, num_classes=11)
        val_loss, val_iou = validate_fn(val_loader, model, loss_fn, device, num_classes=11)

        print(f'Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}')
        
        # Log the losses and IoU to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('IoU/Train', train_iou, epoch)
        writer.add_scalar('IoU/Validation', val_iou, epoch)
        
        # Ensure the checkpoints directory exists
        if not os.path.exists(r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\Segmentation\Checkpoints Resnet"):
            os.makedirs(r"C:\Users\Elie_\Desktop\VS CODE PROJECTS\Final Project Inmind\Segmentation\Checkpoints Resnet")

        # Save checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"C:\\Users\\Elie_\\Desktop\\VS CODE PROJECTS\\Final Project Inmind\\Segmentation\\Checkpoints Resnet HYPERPARAMETERS\\deeplabv3_epoch_{epoch+1}.pth.tar")

    writer.close()


if __name__ == "__main__":
    main()
