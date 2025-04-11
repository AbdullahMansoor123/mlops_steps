import torch
from datetime import datetime
import torch.nn as nn
from pipelines.data import MNISTDataModule
from pipelines.model import SimpleCNN
from pipelines.train import Trainer

import wandb

def main():
    """
    Main function to execute the entire workflow.
    """
    config = {
        "batch" : 32,
        "epochs" : 5,
        "val_split" : 0.2,
        "test_split" : 0.2,
        "num_classes" : 10,
        "lr" : 3e-5,
        "loss": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "model": SimpleCNN,
        "wandb_project": "mlops-steps"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = MNISTDataModule(batch_size=config["batch"], 
                                  val_split= config["val_split"], 
                                  test_split=config["test_split"])
    data_module.setup()
    train_loader, val_loader = data_module.train_loader, data_module.val_loader

    
    # Load pre-trained model
    model = SimpleCNN(config["num_classes"]).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    

    # Initialize wandb
    exp_name = f"mnist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project= config["wandb_project"], 
                name= exp_name,
                config= config)


    # Train the model
    training = Trainer(
                        model,
                        train_loader,
                        val_loader,
                        loss,
                        optimizer,
                        epochs=config["epochs"]
                    
                    )
    
    training.train()
    
if __name__ == "__main__":
    main()