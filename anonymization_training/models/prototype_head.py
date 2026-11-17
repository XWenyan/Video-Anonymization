import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):

    def __init__(self,
                 in_channels=2048,
                 hidden_dim=512,
                 prototype_dim=256):
        super().__init__()

        self.projector = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=1,
                bias=False),

            nn.BatchNorm2d(hidden_dim),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)

        )

        self.fc = nn.Linear(
            hidden_dim,
            prototype_dim)

    def forward(self, feature):

        x = self.projector(feature)

        x = torch.flatten(x,1)

        prototype = self.fc(x)

        prototype = F.normalize(
            prototype,
            dim=1)

        return prototype