from dataset import get_data_loaders
from modules import SiameseNetwork, ContrastiveLoss
from train import train_loop
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_LOAD_PATH = 'best_siamese_model.pth'
HISTORY_LOAD_PATH = 'training_history.npz'

def predict():
    print("-- Starting Evaluation --")
    history_data = np.load(HISTORY_LOAD_PATH)
    best_val_thresh = history_data['best_val_thresh'].item()
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    model.eval()
    criterion = ContrastiveLoss(margin=1.5)
    _, _, test_loader = get_data_loaders(batch_size=32, pairs_per_epoch=15000)
    
    total_loss, all_dist, all_labels = 0.0, [], []
    with torch.no_grad():
        for (img1, img2), labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)
            total_loss += loss.item() * labels.size(0)
            dist = nn.functional.pairwise_distance(emb1, emb2)
            all_dist.append(dist.cpu())
            all_labels.append(labels.cpu())

    all_dist = torch.cat(all_dist)
    all_labels = torch.cat(all_labels)
    test_probas = -all_dist.numpy()
    test_labels = all_labels.numpy()
    
    # E. Calculate Final Metrics (using the best_val_thresh)
    test_preds = (test_probas < -best_val_thresh).astype(int) # Classification using saved threshold
    test_acc = (test_preds == test_labels).mean() * 100
    test_auc = auc(roc_curve(test_labels, test_probas)[0:2])
    test_loss = total_loss / len(test_loader.dataset)
    
    print(f"\n✅ Final Test Results (Best Model): Loss = {test_loss:.4f} | Accuracy = {test_acc:.2f}% | AUC = {test_auc:.4f}")
    plot_test_metrics(test_labels, test_probas, best_val_thresh)




def plot_test_metrics(all_labels, all_probas, best_thresh):
    # --- 1. Confusion Matrix ---
    test_preds = (all_probas > best_thresh).astype(int) 
    cm = confusion_matrix(all_labels, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    plt.figure(figsize=(6, 6))
    disp.plot(ax=plt.gca(), cmap=plt.cm.Blues, values_format='d')
    plt.title('Test Confusion Matrix (Best Threshold)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Plot saved to: confusion_matrix.png")
    
    # --- 2. ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_probas)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Test Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()
    print("Plot saved to: roc_curve.png")


if __name__ == '__main__':
    train_loop()
    predict()