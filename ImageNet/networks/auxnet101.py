import torch.nn as nn
import torch.nn.functional as F

class AuxClassifier(nn.Module):
    def __init__(self, inplanes):
        super(AuxClassifier, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.fc_out_channels = 1000

        if inplanes == 256:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 1000)
            )

        elif inplanes == 512:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 1000)
            )

        elif inplanes == 1024:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(1024, 1000)
            )

        elif inplanes == 2048:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(2048, 1000)
            )

    def forward(self, x, target):

        features = self.head(x)

        loss = self.criterion(features,target)

        return loss