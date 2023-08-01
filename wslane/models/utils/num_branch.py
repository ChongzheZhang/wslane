import torch.nn as nn
import torch.nn.functional as F


class NumBranch(nn.Module):
    def __init__(self, nlane, channels):
        super().__init__()
        self.nlane = nlane
        self.channels = channels
        self.inter_channels = channels // 4
        self.conv1x1 = nn.Conv2d(channels, self.inter_channels, kernel_size=1)
        self.linear = nn.Linear(self.inter_channels, self.nlane)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        maxpool = nn.MaxPool2d(x.shape[2:])
        x = F.softmax(x, dim=1)
        x = maxpool(x)
        x = self.conv1x1(x)
        x = x.view(-1, self.inter_channels)
        x = self.linear(x)
        
        return x

    def loss(self, predictions, labels, weight):
        return self.ce_loss(predictions, labels) * weight
