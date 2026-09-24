"""SigLIP representation extractor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..base import FeatureExtractor
from ..config import ExtractorConfig
from ..image_utils import ensure_channels, to_uint8_rgb
from ..preprocessing import ImagePreprocessConfig, apply_image_preprocessing, read_image


DEFAULT_MODEL = "google/siglip-so400m-patch14-384"


class SigLIPFeatureExtractor(FeatureExtractor):
	"""Extract image features from Hugging Face SigLIP models."""

	backend = "siglip"
	modality = "image"

	def __init__(
		self,
		model_name: str = DEFAULT_MODEL,
		device: str = "cuda",
		imgsize: int | None = None,
		reset_meanstd: bool = False,
		reset_rescale: bool = False,
		preprocessing: ImagePreprocessConfig | None = None,
	) -> None:
		super().__init__()
		self.model_name = model_name
		self.requested_device = device
		self.device = device
		self.imgsize = imgsize
		self.reset_meanstd = reset_meanstd
		self.reset_rescale = reset_rescale
		self.preprocessing = preprocessing or ImagePreprocessConfig()
		self.model = None
		self.processor = None
		self.torch = None

	def load_model(self) -> None:
		"""Load the SigLIP model and Hugging Face image processor."""
		try:
			import torch
			from transformers import AutoModel, AutoProcessor
		except ImportError as exc:
			raise RuntimeError("SigLIP backend requested, but PyTorch/Transformers are not installed. Install fextractor[siglip].") from exc

		self.torch = torch
		if "cuda" in self.device and not torch.cuda.is_available():
			self.device = "cpu"

		self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
		self.model.eval()
		self.processor = AutoProcessor.from_pretrained(self.model_name)
		image_processor = self.processor.image_processor

		if self.imgsize is not None:
			if isinstance(image_processor.size, dict):
				if "height" in image_processor.size:
					image_processor.size["height"] = self.imgsize
				if "width" in image_processor.size:
					image_processor.size["width"] = self.imgsize
				if "shortest_edge" in image_processor.size:
					image_processor.size["shortest_edge"] = self.imgsize

		if self.reset_meanstd:
			image_processor.image_mean = [0.0, 0.0, 0.0]
			image_processor.image_std = [1.0, 1.0, 1.0]

		if self.reset_rescale:
			image_processor.do_rescale = False
			image_processor.rescale_factor = 1.0

	def prepare(self, source: str | Path):
		"""Read one image and return processor-ready data."""
		data = read_image(source)
		data = apply_image_preprocessing(data, self.preprocessing)
		if self.reset_rescale:
			return ensure_channels(np.asarray(data, dtype=np.float32), 3)
		data = to_uint8_rgb(data)
		return Image.fromarray(data, mode="RGB")

	def extract(self, source: str | Path) -> np.ndarray:
		"""Return one SigLIP image representation vector."""
		self.ensure_loaded()
		image = self.prepare(source)
		inputs = self.processor(images=image, return_tensors="pt").to(self.device)
		with self.torch.no_grad():
			features = self.model.get_image_features(**inputs)
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
			"reset_meanstd": self.reset_meanstd,
			"reset_rescale": self.reset_rescale,
			"preprocessing": self.preprocessing.__dict__.copy(),
		})
		return metadata


def create(config: ExtractorConfig) -> SigLIPFeatureExtractor:
	"""Create a SigLIP extractor from a backend-neutral configuration."""
	return SigLIPFeatureExtractor(
		model_name=config.model or DEFAULT_MODEL,
		device=config.device,
		imgsize=config.imgsize,
		reset_meanstd=bool(config.get_option("reset_meanstd", False)),
		reset_rescale=bool(config.get_option("reset_rescale", False)),
		preprocessing=config.preprocessing,
	)
