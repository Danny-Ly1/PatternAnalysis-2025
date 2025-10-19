from dataset import *
from modules import *


train_dataset = SiameseISICDataset(train_df, transform)
val_dataset   = SiameseISICDataset(val_df, transform)
test_dataset  = SiameseISICDataset(test_df, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)

num_epochs = 20
best_acc = 0.0

# ==============================
#  Training Loop
# ==============================
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(img1, img2)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # Validation with threshold tuning
    model.eval()
    val_outputs, val_labels = [], []
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            outputs = model(img1, img2)
            val_outputs.append(outputs)
            val_labels.append(label)

    val_outputs = torch.cat(val_outputs)
    val_labels = torch.cat(val_labels)

    # Find best threshold
    thresholds = torch.linspace(0.0, 1.0, 101).to(device)
    best_thresh, best_acc_epoch = 0.5, 0.0
    for t in thresholds:
        preds = (torch.sigmoid(val_outputs) >= t).float()
        acc = (preds == val_labels).float().mean().item()
        if acc > best_acc_epoch:
            best_acc_epoch = acc
            best_thresh = t.item()

    # Save best model
    if best_acc_epoch*100 > best_acc:
        best_acc = best_acc_epoch*100
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | "
        f"Val Acc: {best_acc_epoch*100:.2f}% | Best Thresh: {best_thresh:.3f} | Best Acc: {best_acc:.2f}%")


# ==============================
#  Final Test Evaluation
# ==============================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_outputs, test_labels = [], []
with torch.no_grad():
    for img1, img2, label in test_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        outputs = model(img1, img2)
        test_outputs.append(outputs)
        test_labels.append(label)

test_outputs = torch.cat(test_outputs)
test_labels = torch.cat(test_labels)
preds = (torch.sigmoid(test_outputs) >= best_thresh).float()
test_acc = (preds == test_labels).float().mean().item() * 100
print(f"\n Final Test Accuracy: {test_acc:.2f}% | Threshold: {best_thresh:.3f}")