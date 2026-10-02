from pathlib import Path

import torch
from transformers import AutoModel


model_path = Path("/opt/models/falcon1")
if not model_path.is_dir():
	raise SystemExit(f"Falcon-1 model directory not found: {model_path}")

model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
model.eval()

if not hasattr(model, "model") or not hasattr(model.model, "decoder"):
	raise SystemExit("Unexpected Falcon-1 model structure")

print("torch:", torch.__version__)
print("model:", type(model).__name__)
print("seq_length:", model.model.seq_length)
print("hidden_size:", model.model.config.hidden_size)
print("num_hidden_layers:", model.model.config.num_hidden_layers)
print("shared_patch_size:", model.model.config.shared_patch_size)
print("Falcon-1 import/load OK")
