import torch
from datetime import datetime
import torch.nn as nn
from pipelines.data import MNISTDataModule
from pipelines.model import SimpleCNN
from pipelines.train import Trainer

import wandb
import omegaconf

import hydra
from omegaconf.omegaconf import OmegaConf


@hydra.main(config_path="./configs", config_name="config")

def main(cfg):
    """
    Main function to execute the entire workflow.
    """



    config = {
            "batch" : 32,
            "epochs" : 3,
            "val_split" : 0.2,
            "test_split" : 0.2,
            "num_classes" : 10,
            "lr" : 3e-5,
            "loss": "CrossEntropyLoss",
            "optimizer": "AdamW",
            "model": SimpleCNN,
            "wandb_project": "mlops-steps"
    }
    # print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = MNISTDataModule(batch_size= cfg.processing.batch_size, 
                                  val_split= cfg.processing.val_split, 
                                  test_split= cfg.processing.test_split)
    data_module.setup()
    train_loader, val_loader = data_module.train_loader, data_module.val_loader
    
    # Load pre-trained model
    model = SimpleCNN(cfg.model.num_classes).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    # # Initialize wandb
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    exp_name = f"mnist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project= cfg.training.wandb_project, 
                name= exp_name)

    # Train the model
    training = Trainer(
                        model,
                        train_loader,
                        val_loader,
                        loss,
                        optimizer,
                        epochs=cfg.training.epochs
                    
                    )
    
    training.train()
    
if __name__ == "__main__":
    main()