from dataset import get_data_loaders
from modules import SiameseNetwork, ContrastiveLoss
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


NUM_EPOCHS = 20
MODEL_SAVE_PATH = 'best_siamese_model.pth'
HISTORY_SAVE_PATH = 'training_history.npz'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader, criterion):
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
    
    return avg_loss, acc, auc_score, best_thresh

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, all_dist, all_labels = 0, [], []
    for (img1, img2), labels in loader:
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

    all_dist_train = torch.cat(all_dist)
    all_labels_train = torch.cat(all_labels)
    fpr, tpr, thresh = roc_curve(all_labels_train, -all_dist_train)
    train_auc = auc(fpr, tpr)
    best_idx = (tpr - fpr).argmax()
    train_thresh = thresh[best_idx]
    train_preds = (all_dist_train < -train_thresh).float()

    train_acc = (train_preds == all_labels_train).float().mean().item() * 100
    train_loss = total_loss / len(loader.dataset)

    return train_loss, train_acc, train_auc

def train_loop():
    train_loader, validation_loader, _ = get_data_loaders(batch_size=32, pairs_per_epoch=15000)
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_aucs, val_aucs = [], []
    best_val_auc = 0
    best_val_thresh = 0.0

    print(f"Starting Training for {NUM_EPOCHS} epochs.")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, train_auc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc, val_thresh = evaluate(model, validation_loader, criterion)

        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_aucs.append(train_auc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_thresh = val_thresh
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Model saved! New Best Val AUC: {best_val_auc:.4f}")

    print("\nTraining Complete.")

    np.savez(
        HISTORY_SAVE_PATH,
        train_losses=train_losses, val_losses=val_losses,
        train_accs=train_accs, val_accs=val_accs,
        train_aucs=train_aucs, val_aucs=val_aucs,
        best_val_thresh=best_val_thresh
    )
    print(f"Training history and best threshold saved to {HISTORY_SAVE_PATH}")

def plot_metrics(train_metrics, val_metrics, metric_name, title, filename):
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'b-o', label=f'Training {metric_name}', markersize=4, linewidth=2)
    plt.plot(epochs, val_metrics, 'r--o', label=f'Validation {metric_name}', markersize=4, linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Save the plot instead of showing it
    plt.savefig(filename)
    plt.close() # Close the figure to free up memory
    print(f"Plot saved to: {filename}")

def get_train_val_plot():
    history_data = np.load(HISTORY_SAVE_PATH)
    train_losses = history_data['train_losses'].item()
    val_losses = history_data['val_losses'].item()
    train_accs = history_data['train_accs'].item()
    val_accs = history_data['val_accs'].item()
    train_aucs = history_data['train_aucs'].item()
    val_aucs = history_data['val_aucs'].item()

    plot_metrics(train_losses, val_losses, 'Loss', 'Training and Validation Loss over Epochs', 'loss_over_epochs.png')
    plot_metrics(train_accs, val_accs, 'Accuracy (%)', 'Training and Validation Accuracy over Epochs', 'accuracy_over_epochs.png')
    plot_metrics(train_aucs, val_aucs, 'AUC Score', 'Training and Validation AUC over Epochs', 'auc_over_epochs.png')

if __name__ == '__main__':
    train_loop()
    get_train_val_plot()