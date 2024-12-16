import torch.nn as nn
import torch.nn.functional as F

class AuxClassifier(nn.Module):
    def __init__(self, inplanes):
        super(AuxClassifier, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.fc_out_channels = 1000

        if inplanes == 64:
            self.head = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 1000)
            )

        elif inplanes == 128:
            self.head = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 1000)
            )

        elif inplanes == 256:
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
    def forward(self, x, target):

        features = self.head(x)

        loss = self.criterion(features,target)

        return loss