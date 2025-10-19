import os, random
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Paths
image_dir = "/home/Student/s4883560/project/train-image/image"
metadata_path = "/home/Student/s4883560/project/isic_metadata/train-metadata.csv"

# Load metadata
df = pd.read_csv(metadata_path)
df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

# Split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.05, stratify=train_df['target'], random_state=42)

print("Train dataset size:", len(train_df))
print("Validation dataset size:", len(val_df))
print("Test dataset size:", len(test_df))

# Dataset
class SiameseISICDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = df["target"].unique()
        self.class_indices = {c: self.df[self.df["target"] == c].index.tolist() for c in self.classes}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img1_path, label1 = self.df.loc[idx, ["image_path", "target"]]
        should_get_same = random.randint(0, 1)

        if should_get_same:
            idx2 = random.choice(self.class_indices[int(label1)])
        else:
            diff_label = 1 - label1
            idx2 = random.choice(self.class_indices[int(diff_label)])

        img2_path, label2 = self.df.loc[idx2, ["image_path", "target"]]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor([1 if label1 == label2 else 0], dtype=torch.float32)
        return img1, img2, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])