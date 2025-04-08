import torch
from pipelines.data import DataModule
from pipelines.model import load_model
from pipelines.train import train_model


def main():
    """
    Main function to execute the entire workflow.
    """
    dataset_name = "imdb"  # Hugging Face dataset name
    model_name = "bert-base-uncased"  # Pre-trained model name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    lr = 3e-5

    data_model = DataModule(dataset_name=dataset_name, model_name=model_name)
    train_loader, val_loader = data_model.train_loader, data_model.val_loader

    
    # Load pre-trained model
    model = load_model(model_name, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Train the model
    train_model(train_loader, val_loader, model, optimizer, device)
    
if __name__ == "__main__":
    main()