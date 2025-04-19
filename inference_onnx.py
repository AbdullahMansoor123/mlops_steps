import onnxruntime as ort
import numpy as np
from pipelines.data import MNISTDataModule

# Inference with ONNX model
class Inference_Onnx:
    def __init__(self, onnx_model_path, test_loader):
        self.session = ort.InferenceSession(onnx_model_path)
        self.test_loader = test_loader

        # Get input and output names from ONNX model
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self):
        correct = 0
        total = 0

        for images, labels in self.test_loader:
            # Convert to NumPy and float32 (required by ONNX Runtime)
            images_np = images.numpy().astype(np.float32)

            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: images_np})
            predictions = np.argmax(outputs[0], axis=1)

            # Accuracy calculation
            total += labels.size(0)
            correct += (predictions == labels.numpy()).sum().item()

        print(f'Test Accuracy (ONNX): {100 * correct / total:.2f}%')


data_module = MNISTDataModule()
data_module.setup()
test_loader = data_module.test_loader

onnx_model_path = "onnx/mnist_model.onnx"  # or your model path
inference = Inference_Onnx(onnx_model_path, test_loader)
inference.run()