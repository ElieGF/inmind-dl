import torch
import torch.nn as nn

class MiniBlock(nn.Module):
    def __init__(self,in_channels,num_filters,kernel_size,stride,padding):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, num_filters, bias=False, kernel_size=kernel_size, stride=stride,padding=padding)
        self.batchnorm = nn.BatchNorm2d(num_filters)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x=self.conv(x)
        x=self.batchnorm(x)
        x=self.leakyrelu(x)
        return x


class YOLOV1(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(

            MiniBlock(3,64,7,2,3),
            nn.MaxPool2d(2,2),

            MiniBlock(64,192,3,1,1),
            nn.MaxPool2d(2,2),

            MiniBlock(192,128,1,1,0),
            MiniBlock(128,256,3,1,1),
            MiniBlock(256,256,1,1,0),
            MiniBlock(256,512,3,1,1),
            nn.MaxPool2d(2,2),

            MiniBlock(512,256,1,1,0),
            MiniBlock(256,512,3,1,1),
            MiniBlock(512,256,1,1,0),
            MiniBlock(256,512,3,1,1),
            MiniBlock(512,256,1,1,0),
            MiniBlock(256,512,3,1,1),
            MiniBlock(512,256,1,1,0),
            MiniBlock(256,512,3,1,1),

            MiniBlock(512,512,1,1,0),
            MiniBlock(512,1024,3,1,1),
            nn.MaxPool2d(2,2),

            MiniBlock(1024,512,1,1,0),
            MiniBlock(512,1024,3,1,1),
            MiniBlock(1024,512,1,1,0),
            MiniBlock(512,1024,3,1,1),

            MiniBlock(1024,1024,3,1,1),
            MiniBlock(1024,1024,3,2,1),

            MiniBlock(1024,1024,3,1,1),
            MiniBlock(1024,1024,3,1,1)
        )

        self.flatten = nn.Flatten()

        self.con = nn.Sequential(nn.Linear(1024*7*7,4096),
                   nn.LeakyReLU(0.1),
                   nn.Linear(4096,30*7*7),
                   nn.LeakyReLU(0.1))

    def forward(self,x):
        x = self.seq(x)
        x = self.flatten(x)
        x = self.con(x)
        return x


model = YOLOV1()

input_tensor = torch.randn(1, 3, 448, 448)  # (batch_size, channels, height, width)

model.eval()
with torch.no_grad():
    output = model(input_tensor)

print('Output Tensor Shape:', output.shape)
