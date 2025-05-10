from model import TinyMelClassifier
import torch
from safetensors.torch import load_file

RUN_NAME = "stellar-vortex-13"
EPOCH = 100

torch_model = TinyMelClassifier()
torch_model.load_state_dict(load_file(f"model/{RUN_NAME}/model_{EPOCH}.safetensors"))
torch_model.eval()

# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
dummy_mel = torch.randn(8, 1, 220500)
onnx_program = torch.onnx.export(torch_model, dummy_mel, "tiny_mel_classifier.onnx", input_names=["mel"], output_names=["output"], dynamic_axes={"mel": {0: "batch_size"}})