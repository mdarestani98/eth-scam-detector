import torch.nn as nn
from torch import Tensor


class FCN(nn.Module):
    def __init__(self, in_features: int, no_classes: int):
        super(FCN, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(in_features, 48, bias=False),
            nn.BatchNorm1d(48, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(48, 64, bias=False),
            nn.BatchNorm1d(64, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=0.1)
        self.layer2 = nn.Sequential(
            nn.Linear(64, 96, bias=False),
            nn.BatchNorm1d(96, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(96, 32, bias=False),
            nn.BatchNorm1d(32, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Linear(32, no_classes, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cls(x)
        return x
