from dataset import *
from modules import *


train_dataset = SiameseISICDataset(train_df, transform)
test_dataset  = SiameseISICDataset(test_df, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = 20
best_acc = 0.0
best_thresh = 0.98

# ==============================
#  Training Loop
# ==============================
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for img1, img2, labels in train_loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(img1, img2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(outputs) >= best_thresh).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total * 100

    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) >= best_thresh).float()
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_loss /= test_total
    test_acc = test_correct / test_total * 100

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Step LR scheduler
    scheduler.step()