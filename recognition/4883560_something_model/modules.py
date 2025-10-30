"""
Contains the source code that creates the Siamese network and the 
loss function (Contrastive) for the model.

Author: Danny Ly
"""

import torch.nn as nn
import torch
from torchvision import models

#---------------------Siamese Network------------------------#
class SiameseNetwork(nn.Module):
    """
    Creates a siamese network using a resnet-50 CNN as the backbone of the network and a modified head.
    """
    def __init__(self):
        """
        Initalises the model.
        """
        super().__init__()
        base = models.resnet50(pretrained = True)
        
        # Replace the final layer of the resnet
        base.fc = nn.Identity()
        self.backbone = base
        # Modify the head of the CNN specific to lower-dimensional embedding space
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64)
        )

    def forward_once(self, x):
        """
        Performs a forward pass through the network.

        Parameters:
            x: The input image tensor.
        
        Returns: 
            Normalized embedding vector for that image.

        """
        x = self.backbone(x)
        x = self.head(x)
        return nn.functional.normalize(x, p=2, dim=1)
    
    def forward(self, x1, x2):
        """
        Performs a forward pass through the network with two images.

        Parameters:
            x1: First image.
            x2: Second image.
        
        Returns:
            Embeddings for both images.
        """
        return self.forward_once(x1), self.forward_once(x2)


#---------------------Loss Function------------------------#
class ContrastiveLoss(nn.Module):
    """
    Loss function for training the model.
    """
    def __init__(self, margin=1.5):
        """
        Initialise the function.

        Parameters:
            margin: Distance threshold.
        """
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        """
        Calculates the loss for a pair of images.

        Parameters:
            out1: Embedding tensor from the first image.
            out2: Embedding tensor from the second image.
            label: Label for the pair of embeddings.

        Returns:
            The contrastive loss value.
        """
        dist = nn.functional.pairwise_distance(out1, out2)
        loss = torch.mean(
            label * torch.pow(dist, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss