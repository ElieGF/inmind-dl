import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from utils import *
from Dataset import *
from model import *



# Set up parameters
train_dir = "C:\\Users\\Elie_\\Desktop\\train"
train_maskdir = "C:\\Users\\Elie_\\Desktop\\train_masks"
val_dir = "C:\\Users\\Elie_\\Desktop\\val"
val_maskdir = "C:\\Users\\Elie_\\Desktop\\val_masks"
EPOCHS = 2
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=130, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    ToTensorV2(),
])
val_transform = A.Compose([A.Resize(256, 256), ToTensorV2()])


# Initialize Model, Optimizer, and Loss Function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNET(in_channels=3, out_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()


# Set up the train function
def train_fn(train_loader, model, optimizer, loss_fn, device):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item()}")


# Main function
def main():
    print(f"Using device: {device}")
    # Get data loaders
    train_loader, val_loader = get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size=8,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=4,
        pin_memory=True,
    )

    # Training loop
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        train_fn(train_loader, model, optimizer, loss_fn, device)
        print(f'Epochs completed: {epoch+1}')


if __name__=="__main__":
    main()