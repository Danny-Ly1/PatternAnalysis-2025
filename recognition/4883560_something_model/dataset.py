import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


base_dir = "/home/groups/comp3710/ISIC2018"

train_img_dir = os.path.join(base_dir, "ISIC2018_Task1-2_Training_Input_x2")
train_mask_dir = os.path.join(base_dir, "ISIC2018_Task1_Training_GroundTruth_x2")
test_img_dir  = os.path.join(base_dir, "ISIC2018_Task1-2_Test_Input")

data = []
for filename in os.listdir(train_img_dir):
    if filename.endswith(".png"):
        img_path = os.path.join(train_img_dir, filename)
        mask_path = os.path.join(train_mask_dir, filename)
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path).convert("L"))
        label = 1 if np.any(mask > 0) else 0
        data.append((img_path, label))

df = pd.DataFrame(data, columns=["path", "target"])
print("Label distribution:\n", df["target"].value_counts())

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)

class SiameseISICDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = df["target"].unique()
        self.class_indices = {c: df[df["target"] == c].index.tolist() for c in self.classes}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img1_path, label1 = self.df.loc[idx, ["path", "target"]]
        should_get_same = random.randint(0, 1)

        if should_get_same:
            idx2 = random.choice(self.class_indices[label1])
        else:
            diff_label = 1 - label1
            idx2 = random.choice(self.class_indices[diff_label])

        img2_path, label2 = self.df.loc[idx2, ["path", "target"]]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor([1 if label1 == label2 else 0], dtype=torch.float32)
        return img1, img2, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = SiameseISICDataset(train_df, transform)
test_dataset  = SiameseISICDataset(test_df, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)