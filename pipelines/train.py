import os

import torch
import torch.nn as nn
from pipelines.data import MNISTDataModule
from pipelines.model import SimpleCNN
from torch import optim
# import the library
import wandb



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training Module
class Trainer:
    def __init__(self, model, train_loader, val_loader,  loss,optimizer , epochs=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = loss
        self.optimizer = optimizer
        self.epochs = epochs

        
        wandb.watch(self.model, log="all", log_freq=100)

        if os.path.exists("mnist_cnn.pth"):
            self.model.load_state_dict(torch.load("mnist_cnn.pth"))
            print("Loaded pretrained weights.")

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

            self.validate(epoch+1)
            
            torch.save(self.model.state_dict(), "mnist_cnn.pth")

    def validate(self,epoch):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        wandb.log({"epoch": epoch + 1, "accuracy": accuracy})
