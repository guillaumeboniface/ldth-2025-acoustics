from model import MelClassifier
import torch

torch_model = MelClassifier()
# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
dummy_mel = torch.randn(8, 1, 64, 601)
onnx_program = torch.onnx.export(torch_model, dummy_mel, "mel_classifier.onnx", input_names=["mel"], output_names=["output"], dynamic_axes={"mel": {0: "batch_size"}})