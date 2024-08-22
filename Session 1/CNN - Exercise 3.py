from torch import nn
import torch


class LeNet5V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14

            # 2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5

            # 3
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),  # 1*1*120
            nn.Tanh()

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Fully Connected Layer 1: Input features = 120*1*1 = 120, Output features = 84
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            # Fully Connected Layer 2: Input features = 84, Output features = 10
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(self.feature(x))


model = LeNet5V1()
print(model)

# model.cuda()
from torchsummary import summary

summary(model, (1, 32, 32))
# Instantiate the LeNet5V1 model
model = LeNet5V1()

# Generate random input tensor of shape (batch_size, channels, height, width)
batch_size = 1
channels = 1
height = 32
width = 32
random_input = torch.rand(batch_size, channels, height, width)

# Feed the input through the model
with torch.no_grad():
    output = model(random_input)

# Print the output
print("Output Tensor:")
print(output)