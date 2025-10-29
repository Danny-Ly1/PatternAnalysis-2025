import os, random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Paths
image_dir = "/home/Student/s4883560/project/train-image/image"
metadata_path = "/home/Student/s4883560/project/isic_metadata/train-metadata.csv"

# Load metadata
df = pd.read_csv(metadata_path)
df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

# Split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['target'], random_state=42)

print("Train dataset size:", len(train_df))
print("Validation dataset size;", len(val_df))
print("Test dataset size:", len(test_df))

# Dataset
class SiameseISICDataset(Dataset):
    def __init__(self, df, transform, pairs_per_epoch):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(df["target"].unique().tolist())
        self.class_indices = {c: self.df[self.df["target"] == c].index.tolist() for c in self.classes}
        self.pairs_per_epoch = pairs_per_epoch

        
        class_counts = self.df['target'].value_counts()
        self.class_weights = {c: 1.0 / count for c, count in class_counts.items()}
        total_weight = sum(self.class_weights.values())
        self.class_probabilities = [self.class_weights[c] / total_weight for c in self.classes]

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        label = random.randint(0, 1)
        
        if label == 1:
            c = random.choices(self.classes, weights=self.class_probabilities, k=1)[0]
            
            if len(self.class_indices[c]) < 2:
                c = random.choice(self.classes)
            
            i1, i2 = random.sample(self.class_indices[c], 2)
            
        else:
            c1, c2 = random.sample(self.classes, 2)
            i1 = random.choice(self.class_indices[c1])
            i2 = random.choice(self.class_indices[c2])

        img1 = Image.open(self.df.loc[i1, "image_path"]).convert("RGB")
        img2 = Image.open(self.df.loc[i2, "image_path"]).convert("RGB")

        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        return (img1, img2), torch.tensor(float(label))

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform
    
def get_data_loaders(batch_size = 32, pairs_per_epoch = 15000):
    train_transform, test_transform = get_transforms()

    train_dataset = SiameseISICDataset(train_df, train_transform, pairs_per_epoch)
    validation_dataset = SiameseISICDataset(val_df, test_transform, pairs_per_epoch)
    test_dataset = SiameseISICDataset(test_df, test_transform, pairs_per_epoch)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    return train_loader, validation_loader, test_loader