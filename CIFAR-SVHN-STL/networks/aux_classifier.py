import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import SupConLoss


class AuxiliaryClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.fc = None  
        self.num_features = None  
        self.num_classes = num_classes

    def forward(self, x):
        if self.fc is None:
            self.num_features = torch.prod(torch.tensor(x.shape[1:])).item()
            self.fc = nn.Linear(self.num_features, self.num_classes)
        x = x.view(x.size(0), -1)
        return self.fc(x)