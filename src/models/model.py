import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolutional neural network
class CNN(nn.Module):
    """Convolutional neural network (2 convolutional layers, 2 fully connected layers)"""

    def __init__(self):
        """Initializes the neural network parameters"""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=3, padding=1
        )  # 28*28*1 -> 28*28*3
        self.batchnorm1 = nn.BatchNorm2d(3)
        self.pool = nn.MaxPool2d(2, 2)  # 28*28*3 -> 14*14*3
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, padding=0
        )  # 14*14*3 -> 10*10*6

        self.fc1 = nn.Linear(6 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        """Runs the forward pass

        Args:
            x (torch.Tensor): image input (1*28*28) (grey scale)
        Returns:
            x: (torch.Tensor): used for embeddings
            y: (torch.Tensor): used for predictions (10 classes)
        """
        x = self.pool(self.batchnorm1(F.relu(self.conv1(x))))  # 28*28*1 -> 14*14*3
        x = self.pool(F.relu(self.conv2(x)))  # 14*14*3 -> 5*5*6

        # flatten to fit into linear layers
        x = torch.flatten(x, start_dim=1)  # 5*5*6 -> 150
        x = F.relu(self.fc1(F.dropout(x, p=0.2)))  # 150 -> 128
        x = F.relu(self.fc2(F.dropout(x, p=0.2)))  # 128 -> 64
        x = self.fc3(x)  # 64 -> 10
        y = F.softmax(x, dim=1)  # preds

        return x, y
