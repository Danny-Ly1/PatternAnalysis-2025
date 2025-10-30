"""
Contains methods to train and evaluate the model. Also includes the main training loop and 
methods to plot the results for each epoch.

Author: Danny Ly
"""
from dataset import get_data_loaders
from modules import SiameseNetwork, ContrastiveLoss
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Constants
NUM_EPOCHS = 20
MODEL_SAVE_PATH = 'best_siamese_model.pth'
HISTORY_SAVE_PATH = 'training_history.npz'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------Train and Evaluate Functions------------------------#
def evaluate(model, loader, criterion):
    """
    Evaluates the model's performance on a given dataset (used for validation and core use in testing).

    Parameters:
        model: The siamese model to evaluate.
        loader: Dataloader to feed into the evaluation.
        criterion: The loss function.
    
    Returns:
        Average loss, accuracy, area under curve score, best threshold.
    """
    # Set to evaluation mode
    model.eval()
    total_loss, all_dist, all_labels = 0.0, [], []
    # Disable gradient calculations
    with torch.no_grad():
        # Iterate through the data in the dataloader
        for (img1, img2), labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            # Pass the images into the model
            emb1, emb2 = model(img1, img2)
            # Obtain the loss
            loss = criterion(emb1, emb2, labels)
            total_loss += loss.item() * labels.size(0)
            # Calculate the distance between the embeddings
            dist = nn.functional.pairwise_distance(emb1, emb2)
            # Store info
            all_dist.append(dist.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all stored distance and labels
    all_dist = torch.cat(all_dist)
    all_labels = torch.cat(all_labels)

    # Calculate metrics
    fpr, tpr, thresh = roc_curve(all_labels, -all_dist)
    auc_score = auc(fpr, tpr)
    # Find the best threshold
    best_idx = (tpr - fpr).argmax()
    best_thresh = thresh[best_idx]
    # Calculate predictions using best threshold
    preds = (all_dist < -best_thresh).float()
    # Calculate accuracy
    acc = (preds == all_labels).float().mean().item() * 100
    # Calculate average loss per sample
    avg_loss = total_loss / len(loader.dataset)
    
    # Return results
    return avg_loss, acc, auc_score, best_thresh

def train(model, loader, optimizer, criterion):
    """
    Performs a single epoch of training using the model.

    Parameters:
        model: The siamese model to train.
        loader: Dataloader to feed into the training.
        optimizer: The optimizer.
        criterion: The loss function.
    
    Returns:
        Training loss, accuracy and auc.
    """
    # Set model to training mode
    model.train()
    total_loss, all_dist, all_labels = 0, [], []
    for (img1, img2), labels in loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        # Reset gradient to 0
        optimizer.zero_grad()
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels)
        # Backpropagation
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Calculate metrics
        total_loss += loss.item() * labels.size(0)
        dist = nn.functional.pairwise_distance(emb1, emb2)
        # Store metrics
        all_dist.append(dist.detach().cpu())
        all_labels.append(labels.cpu())

    # Concatenate all stored distance and labels
    all_dist_train = torch.cat(all_dist)
    all_labels_train = torch.cat(all_labels)

    # Calculate metrics
    fpr, tpr, thresh = roc_curve(all_labels_train, -all_dist_train)
    train_auc = auc(fpr, tpr)
    best_idx = (tpr - fpr).argmax()
    train_thresh = thresh[best_idx]
    train_preds = (all_dist_train < -train_thresh).float()

    train_acc = (train_preds == all_labels_train).float().mean().item() * 100
    train_loss = total_loss / len(loader.dataset)

    return train_loss, train_acc, train_auc

#---------------------Main Training Loop------------------------#
def train_loop():
    """
    Main loop function that manages all the training for the model.

    Returns:
        None: Prints when the training is complete, saves the metrics, then calls a method to plot the metrics from the training.
    """
    # Obtain training and validation dataloaders
    train_loader, validation_loader, _ = get_data_loaders(batch_size=32, pairs_per_epoch=15000)
    # Initialise model
    model = SiameseNetwork().to(device)
    # Initialise loss function
    criterion = ContrastiveLoss(margin=1.5)
    # Initialise optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_aucs, val_aucs = [], []
    best_val_auc = 0
    best_val_thresh = 0.0

    print(f"Starting Training for {NUM_EPOCHS} epochs.")

    # Main training loop
    for epoch in range(NUM_EPOCHS):
        # Train the model
        train_loss, train_acc, train_auc = train(model, train_loader, optimizer, criterion)
        # Validate the model
        val_loss, val_acc, val_auc, val_thresh = evaluate(model, validation_loader, criterion)

        # Update learning rate
        scheduler.step()

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_aucs.append(train_auc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)

        # Print current epoch results
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}")
        
        # Store best threshold and model when validation accuracy reaches new peak
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_thresh = val_thresh
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Model saved! New Best Val AUC: {best_val_auc:.4f}")

    print("\nTraining Complete.")

    # Save metrics
    np.savez(
        HISTORY_SAVE_PATH,
        train_losses=train_losses, val_losses=val_losses,
        train_accs=train_accs, val_accs=val_accs,
        train_aucs=train_aucs, val_aucs=val_aucs,
        best_val_thresh=best_val_thresh
    )
    print(f"Training history and best threshold saved to {HISTORY_SAVE_PATH}")

    # Plot the training metrics
    get_train_val_plot()


#---------------------Plotting Metric Functions------------------------#
def plot_metrics(train_metrics, val_metrics, metric_name, title, filename):
    """
    Generates a line graph of the metric across the epochs during training and validation.

    Parameters:
        train_metrics: Training data.
        val_metrics: Validation data.
        metric_name: Metric that is being measured.
        title: Title of the plot.
        filename: File name for the plot.
    
    Returns:
        None: A .png image of the plot with the given filename.
    """
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_metrics) + 1)
    # Plot data
    plt.plot(epochs, train_metrics, 'b-o', label=f'Training {metric_name}', markersize=4, linewidth=2)
    plt.plot(epochs, val_metrics, 'r--o', label=f'Validation {metric_name}', markersize=4, linewidth=2)

    # Label and save figure
    plt.title(title, fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to: {filename}")

def get_train_val_plot():
    '''
    Obtains the metrics and generates a plot for the loss, accuracy and AUC score for both training and validation across the epochs.

    Returns:
        None: Calls a method to plot the metrics.
    '''
    # Obtain the metrics
    history_data = np.load(HISTORY_SAVE_PATH)
    train_losses = history_data['train_losses']
    val_losses = history_data['val_losses']
    train_accs = history_data['train_accs']
    val_accs = history_data['val_accs']
    train_aucs = history_data['train_aucs']
    val_aucs = history_data['val_aucs']

    # Plots the metrics
    plot_metrics(train_losses, val_losses, 'Loss', 'Training and Validation Loss over Epochs', 'loss_over_epochs.png')
    plot_metrics(train_accs, val_accs, 'Accuracy (%)', 'Training and Validation Accuracy over Epochs', 'accuracy_over_epochs.png')
    plot_metrics(train_aucs, val_aucs, 'AUC Score', 'Training and Validation AUC over Epochs', 'auc_over_epochs.png')