import hydra
import torch
import os
from pipelines.model import SimpleCNN
from pipelines.data import MNISTDataModule
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="./configs", config_name="config",version_base=None,)
def convert_to_onnx(cfg):
    # Load model checkpoint

    root_dir = hydra.utils.get_original_cwd()
    model_path = os.path.join(root_dir, "models/new_mnist_cnn.pth")
    logger.info(f"Loading model weights from: {model_path}")

    # Model setup
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Load data
    data_module = MNISTDataModule()
    data_module.setup()
    sample_batch = next(iter(data_module.train_loader))
    sample_input = sample_batch[0][0].unsqueeze(0)  # Shape: [1, 1, 28, 28]

    # ONNX export path
    onnx_model_path = os.path.join(root_dir, "onnx/mnist_model.onnx")

    # Export to ONNX
    logger.info("Exporting model to ONNX format...")
    torch.onnx.export(
        model,
        sample_input,
        onnx_model_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    logger.info(f"Model successfully exported to ONNX at: {onnx_model_path}")

if __name__ == "__main__":
    convert_to_onnx()
