import torch.nn as nn
import torch
import torch.nn.functional as F


from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in base.parameters():
            param.requires_grad = False
        for param in base.layer3.parameters():
            param.requires_grad = True
        for param in base.layer4.parameters():
            param.requires_grad = True

        base.fc = nn.Identity()
        self.feature_extractor = base

        self.fc = nn.Sequential(
            nn.Linear(2048 * 2 + 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward_once(self, x):
        return self.feature_extractor(x)

    def forward(self, x1, x2):
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)
        cosine_sim = F.cosine_similarity(f1, f2).unsqueeze(1)
        combined = torch.cat([f1, f2, cosine_sim], dim=1)
        out = self.fc(combined)
        return out