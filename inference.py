import torch
from pipelines.data import MNISTDataModule
from pipelines.model import SimpleCNN

# Inference Module
class Inference:
    def __init__(self, model, test_loader):
        self.model = model.to(device)
        self.test_loader = test_loader

    def run(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Test Accuracy: {100 * correct / total:.2f}%')


data_module = MNISTDataModule()
data_module.setup()
test_loader = data_module.test_loader
model = SimpleCNN(num_classes=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inference = Inference(model, test_loader)
inference.run()