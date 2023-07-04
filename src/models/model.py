import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 3, padding = 1 ) # 28*28*1 -> 28*28*3
        self.batchnorm1 = nn.BatchNorm2d(3)
        self.pool = nn.MaxPool2d(2, 2) # 28*28*3 -> 14*14*3
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, padding = 0) # 14*14*3 -> 10*10*6
        
        self.fc1 = nn.Linear(6*5*5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(self.batchnorm1(F.relu(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        
        # flatten
        x = torch.flatten(x, start_dim = 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(F.dropout(x, p = 0.2)))
        x = F.relu(self.fc2(F.dropout(x, p = 0.2)))
        x = self.fc3(x)
        y = F.softmax(x, dim = 1)
        
        return x, y