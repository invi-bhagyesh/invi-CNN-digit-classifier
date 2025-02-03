"""
Training script for MNIST CNN model
Includes data loading, training loop, validation, and model saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import MNISTCNN
from tqdm import tqdm
import matplotlib.pylab as plt

'''
(ADAM)
batch => 128
- alpha = 0.001 => 0.9948 at 17 epochs and 0.9819 at 1 epochs
- alpha = 0.01 => 0.9762 at 10 epochs and 0.9450 at 1 epochs
- alpha = 0.1 => 0.1135 at 10 epochs and 0.1028 at 1 epochs
batch=> 256
- alpha = 0.001 => 0.9825 at 10 epochs and 0.9710 at 1 epochs
- similar for another batch sizes
'''

# Configuration
DEVICE = torch.device("cpu")
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "mnist_cnn.pth"


def get_dataloaders():
    """
    Create and return MNIST train, validation, and test dataloaders
    Splits training data into train (90%) and validation (10%)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # (Mean, Deviation)
    ])
    
    full_train_set = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Split train set into train (90%) and validation (10%)
    train_size = int(0.9 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])
    
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False),
    )


def train_model():
    """Main training pipeline"""
    # Initialize model, loss, optimizer
    model = MNISTCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    """
    (ADAM) => 0.9948 at 17 epochs
    (SGD with momentum) => 0.9858 at 17 epochs.
    (ADAM with weight decay) => 0.9903 at 14 epochs.

    """
    # Load data
    train_loader, val_loader = get_dataloaders()
    
    best_val_accuracy = 0.0
    iteration_losses = []  # To store loss at each iteration
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            iteration_losses.append(loss.item())  # Append the loss for each iteration
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc = correct / total
        
        print(f"Loss: {epoch_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved new best model with validation accuracy {best_val_accuracy:.4f}")
    
    # After training, plot loss vs iteration
    plt.plot(iteration_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Cost vs Iteration')
    plt.show()


if __name__ == "__main__":
    train_model()
    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")
