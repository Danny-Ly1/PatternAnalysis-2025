from dataset import *
from modules import *
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import roc_curve, auc
from torchvision import transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
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

train_loader = DataLoader(SiameseISICDataset(train_df, train_transform), batch_size=32, shuffle=True)
val_loader = DataLoader(SiameseISICDataset(val_df, test_transform), batch_size=32)
test_loader = DataLoader(SiameseISICDataset(test_df, test_transform), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss(margin=1.5)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

num_epochs = 20
best_val_acc = 0

def evaluate(model, loader):
    model.eval()
    total_loss, all_dist, all_labels = 0.0, [], []
    with torch.no_grad():
        for (img1, img2), labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)
            total_loss += loss.item() * labels.size(0)
            dist = nn.functional.pairwise_distance(emb1, emb2)
            all_dist.append(dist.cpu())
            all_labels.append(labels.cpu())

    all_dist = torch.cat(all_dist)
    all_labels = torch.cat(all_labels)

    fpr, tpr, thresh = roc_curve(all_labels, -all_dist)
    auc_score = auc(fpr, tpr)

    best_idx = (tpr - fpr).argmax()
    best_thresh = thresh[best_idx]
    preds = (all_dist < -best_thresh).float()
    acc = (preds == all_labels).float().mean().item() * 100
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, auc_score

for epoch in range(num_epochs):
    model.train()
    total_loss, all_dist, all_labels = 0, [], []
    for (img1, img2), labels in train_loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        dist = nn.functional.pairwise_distance(emb1, emb2)
        all_dist.append(dist.detach().cpu())
        all_labels.append(labels.cpu())

    # Compute train metrics
    all_dist_train = torch.cat(all_dist)
    all_labels_train = torch.cat(all_labels)
    fpr, tpr, thresh = roc_curve(all_labels_train, -all_dist_train)
    train_auc = auc(fpr, tpr)
    best_idx = (tpr - fpr).argmax()
    train_thresh = thresh[best_idx]
    train_preds = (all_dist < -train_thresh).float()
    train_acc = (train_preds == all_labels_train).float().mean().item() * 100
    train_loss = total_loss / len(train_loader.dataset)

    # Validation
    val_loss, val_acc, val_auc = evaluate(model, val_loader)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train AUC: {train_auc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}")