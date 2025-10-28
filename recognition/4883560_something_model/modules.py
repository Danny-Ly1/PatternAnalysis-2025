import torch.nn as nn
import torch
import torch.nn.functional as F


from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained = True)
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return nn.functional.normalize(x, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = nn.functional.pairwise_distance(out1, out2)
        loss = torch.mean(
            label * torch.pow(dist, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss