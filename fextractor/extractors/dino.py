"""DINOv2 representation extractor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..base import FeatureExtractor
from ..image_utils import to_uint8_rgb
from ..preprocessing import ImagePreprocessConfig, apply_image_preprocessing, read_image


class DINOv2FeatureExtractor(FeatureExtractor):
	"""Extract DINOv2 image representations using the official torch.hub models."""

	backend = "dinov2"
	modality = "image"

	def __init__(
		self,
		model_name: str = "dinov2_vits14",
		device: str = "cuda",
		imgsize: int = 224,
		preprocessing: ImagePreprocessConfig | None = None,
	) -> None:
		super().__init__()
		self.model_name = model_name
		self.requested_device = device
		self.device = device
		self.imgsize = imgsize
		self.preprocessing = preprocessing or ImagePreprocessConfig()
		self.model = None
		self.transform = None
		self.torch = None

	def load_model(self) -> None:
		"""Load DINOv2 and the model-specific torchvision transform."""
		try:
			import torch
			import torchvision.transforms as T
		except ImportError as exc:
			raise RuntimeError("DINOv2 backend requested, but PyTorch/torchvision are not installed. Install fextractor[dino].") from exc

		self.torch = torch
		if "cuda" in self.device and not torch.cuda.is_available():
			self.device = "cpu"

		self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
		self.model.to(self.device)
		self.model.eval()
		self.transform = T.Compose([
			T.Resize((self.imgsize, self.imgsize), interpolation=T.InterpolationMode.BICUBIC),
			T.ToTensor(),
			T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		])

	def prepare(self, source: str | Path):
		"""Read one image and apply domain and DINO-specific preprocessing."""
		data = read_image(source)
		data = apply_image_preprocessing(data, self.preprocessing)
		data = to_uint8_rgb(data)
		image = Image.fromarray(data, mode="RGB")
		return self.transform(image).unsqueeze(0)

	def extract(self, source: str | Path) -> np.ndarray:
		"""Return one DINOv2 representation vector."""
		self.ensure_loaded()
		image_tensor = self.prepare(source).to(self.device)
		with self.torch.no_grad():
			features = self.model(image_tensor)
		features = features.detach().cpu().numpy()
		if features.ndim > 1 and features.shape[0] == 1:
			features = features[0]
		return np.asarray(features, dtype=np.float32).reshape(-1)

	def metadata(self) -> dict:
		metadata = super().metadata()
		metadata.update({
			"model": self.model_name,
			"device": self.device,
			"imgsize": self.imgsize,
			"preprocessing": self.preprocessing.__dict__.copy(),
		})
		return metadata
