import os

import torch
import torch.nn as nn
import wandb.sklearn
from pipelines.data import MNISTDataModule
from pipelines.model import SimpleCNN
from torch import optim
# import the library
import wandb

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


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
            correct = 0
            total = 0
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                # print(f"len outputs {len(outputs)} outputs{outputs}")
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            train_loss = running_loss / len(self.train_loader)
            train_acc = 100 * correct / total
            
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc})

            self.validate(epoch, train_acc, train_loss)
            
            torch.save(self.model.state_dict(), "mnist_cnn.pth")

    def validate(self,epoch, train_loss,train_acc):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(self.train_loader)
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1},  Train_Loss: {train_loss:.4f} | Train_Accuracy: {train_acc:.2f}% | Val_Loss: {val_loss:.4f} | Val_Accuracy: {val_acc:.2f}%")
        wandb.log({"epoch": epoch + 1, "al_loss": val_loss, "val_accuracy": val_acc})

        precision = precision_score(y_true=all_labels, y_pred=all_preds, average='macro') 
        recall = recall_score(y_true=all_labels, y_pred=all_preds, average='macro')
        f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='macro')
        print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} ")
        wandb.log({"precision": precision})
        wandb.log({"recall": recall})
        wandb.log({"f1_score": f1})
        
         
        # 2. Confusion Matrix plotting using scikit-learn method
        class_names = ['0','1','2','3','4','5','6','7','8','9']
        wandb.log({"conf_matrix" : wandb.sklearn.plot_confusion_matrix(
                        y_true=all_labels, y_pred=all_preds,
                        labels=class_names)})

       

