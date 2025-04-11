from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Data Module
class MNISTDataModule:
    def __init__(self, batch_size=64, val_split=0.1, test_split=0.1):
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self):
        dataset = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=self.transform)

        train_len = int(len(dataset) * (1 - self.val_split - self.test_split))
        val_len = int(len(dataset) * self.val_split)
        test_len = len(dataset) - train_len - val_len
        train_data, val_data, _ = random_split(dataset, [train_len, val_len, test_len])

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


# for testing the DataModule class

# if __name__ == "__main__":
#     data_model = MNISTDataModule(batch_size=32, val_split=0.2, test_split=0.2)
#     data_model.setup()
#     batch = next(iter(data_model.train_loader))
#     print(len(batch))
#     print(batch)
