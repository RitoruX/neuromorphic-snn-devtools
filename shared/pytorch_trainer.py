import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class Trainer:
    """
    A class to encapsulate the training, validation, and evaluation loop for a PyTorch model.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, early_stopping):
        """
        Initializes the Trainer object.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            train_loader (torch.utils.data.DataLoader): The data loader for the training set.
            val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            criterion (torch.nn.Module): The loss function.
            early_stopping (EarlyStopping): An object to handle early stopping logic.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.early_stopping = early_stopping
        # Move model to the same device as its parameters, assuming parameters are on the correct device
        self.device = next(model.parameters()).device

        # History for learning curve
        self.train_losses = []
        self.val_losses = []

    def _train_epoch(self, epoch, log_interval):
        """Performs a single training epoch."""
        self.model.train()
        running_loss = 0.0
        for i, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)

            # Log progress every n batches
            if log_interval and (i + 1) % log_interval == 0:
                print(f'Train Epoch: {epoch} [{i * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * i / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def _validate_epoch(self):
        """Performs a single validation epoch."""
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        return epoch_loss

    def train(self, num_epochs, log_interval=None):
        """
        Runs the full training loop for a specified number of epochs.
        
        Args:
            num_epochs (int): The maximum number of epochs to train for.
            log_interval (int, optional): How often to log training progress. Defaults to None.
        """
        print("Starting training...")
        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(epoch, log_interval)
            val_loss = self._validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f'Epoch {epoch} Summary -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')

            # Early stopping check
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        
        print("Training finished.")
        # Load the best model weights found during training
        self.early_stopping.load_best_model(self.model)
        print("Loaded best model weights.")

    def evaluate(self, test_loader):
        """
        Evaluates the model on the test set and provides detailed metrics.

        Args:
            test_loader (torch.utils.data.DataLoader): The data loader for the test set.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item() * data.size(0)
                
                # Get predictions
                _, pred = torch.max(output, 1)
                correct += (pred == target).sum().item()

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print('\n--- Test Set Evaluation ---')
        print(f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
        
        # Classification Report
        print('\nClassification Report:')
        print(classification_report(all_targets, all_preds))

        # Confusion Matrix
        print('\nConfusion Matrix:')
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()


    def plot_learning_curve(self):
        """Plots the training and validation loss curves."""
        if not self.train_losses or not self.val_losses:
            print("No loss history to plot. Please run training first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='dodgerblue')
        plt.plot(self.val_losses, label='Validation Loss', color='darkorange')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()