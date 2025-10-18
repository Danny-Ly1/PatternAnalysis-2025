import torch.nn as nn
import torch
import torch.nn.functional as F


from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        base.fc = nn.Identity()  # remove classifier
        self.feature_extractor = base
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.embedding(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        loss = torch.mean(
            label * torch.pow(dist, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss