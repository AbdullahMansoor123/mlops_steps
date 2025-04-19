import os

import torch
import torch.nn as nn
import wandb.sklearn
from pipelines.data import MNISTDataModule
from pipelines.model import SimpleCNN
from torch import optim
# import the library
import wandb

import hydra

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training Module
class Trainer:
    def __init__(self, model, train_loader, val_loader,  loss,optimizer, model_save_root , epochs=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.model_save_root = model_save_root
        self.best_val_acc = 0.0

        wandb.watch(self.model, log="all", log_freq=100)

        load_model_path = os.path.join(self.model_save_root, "models/new_mnist_cnn.pth")
        if os.path.exists(load_model_path):
            self.model.load_state_dict(torch.load(load_model_path))
            print("Loaded last trained weights.")
        else:
            pre_trained_model_path = os.path.join(self.model_save_root, "mnist_cnn.pth")
            self.model.load_state_dict(torch.load(pre_trained_model_path))
            print("loaded pretrained weights.") 

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

            _, val_acc, _, _, _ = self.validate(epoch, train_acc, train_loss)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.model_save_root, "models/new_mnist_cnn.pth"))
                print(f"Model saved with accuracy: {val_acc:.2f}%")

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

        return val_loss, val_acc, precision, recall, f1
    
