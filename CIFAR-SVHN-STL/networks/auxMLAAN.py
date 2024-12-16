import torch.nn as nn
import torch.nn.functional as F


class AuxClassifier(nn.Module):
    def __init__(self, inplanes,class_num=10, widen=1, feature_dim=128):
        super(AuxClassifier, self).__init__()
        assert inplanes in [16, 32, 64]
        self.feature_dim = feature_dim
        self.criterion = nn.CrossEntropyLoss()
        self.head = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes,class_num)

    def forward(self, x, target):
        features = self.head(x)
        features = features.view(features.size(0),-1)
        features = self.fc(features)
        loss = self.criterion(features, target)
        return loss