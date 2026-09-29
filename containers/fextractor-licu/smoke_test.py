"""Dependency smoke test for the fextractor LiCu container."""

from __future__ import annotations

import json

import huggingface_hub
import light_curve
import onnxruntime as ort
import torch


def main() -> None:
	providers = ort.get_available_providers()

	payload = {
		"torch": torch.__version__,
		"cuda_runtime": torch.version.cuda,
		"cuda_available": torch.cuda.is_available(),
		"light_curve": light_curve.__version__,
		"huggingface_hub": huggingface_hub.__version__,
		"onnxruntime": ort.__version__,
		"onnxruntime_providers": providers,
	}

	print("LiCu container dependency smoke test OK")
	print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
	main()
