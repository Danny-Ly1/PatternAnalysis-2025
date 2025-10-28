from modules import *
from dataset import *
from sklearn.metrics import roc_curve

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss(margin=1.0)

test_loader = DataLoader(SiameseISICDataset(test_df, test_transform), batch_size=32)

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
    best_idx = (tpr - fpr).argmax()
    best_thresh = thresh[best_idx]
    preds = (all_dist < -best_thresh).float()
    acc = (preds == all_labels).float().mean().item() * 100
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, best_thresh

test_loss, test_acc, _ = evaluate(model, test_loader)
print(f"\n Final Test Results: Loss = {test_loss:.4f} | Accuracy = {test_acc:.2f}%")