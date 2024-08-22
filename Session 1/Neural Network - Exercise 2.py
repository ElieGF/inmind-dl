import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 4)
        self.fc5 = nn.Linear(4, 2)

    def forward(self, x):
        # Leaky ReLU: Allows a small gradient when the unit is not active
        x = F.leaky_relu(self.fc1(x))
        # Tanh: Squashes the input to the range [-1, 1]
        x = torch.tanh(self.fc2(x))
        # Sigmoid: Squashes the input to the range [0, 1]
        x = torch.sigmoid(self.fc3(x))
        # ELU: Exponential Linear Unit, smooths negative values to avoid dead neurons
        x = F.elu(self.fc4(x))
        # Softmax: Converts the input to a probability distribution, summing to 1
        x = F.softmax(self.fc5(x), dim=1)
        return x

my_model = NN()

# Testing the model
print(my_model(torch.rand(10,4)).detach())   # Inputting 10 batches of 4 random floats to the model
                                             # We used .detach() to remove the grad_fn from the output

# Printing the model summary
summary(my_model, input_size=(4,))
