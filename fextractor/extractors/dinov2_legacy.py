"""DINOv2 representation extractor."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ..base import FeatureExtractor
from ..config import ExtractorConfig
from ..image_utils import to_uint8_rgb
from ..preprocessing import ImagePreprocessConfig, apply_image_preprocessing, read_image

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "dinov2_vits14"


class DINOv2LegacyFeatureExtractor(FeatureExtractor):
	"""Extract DINOv2 representations using the original Facebook torch.hub implementation."""

	backend = "dinov2_legacy"
	modality = "image"

	def __init__(
		self,
		model_name: str = DEFAULT_MODEL,
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
		"""Legacy torch.hub DINOv2 representation extractor."""
		try:
			import torch
			import torchvision.transforms as T
		except ImportError as exc:
			raise RuntimeError(
				"DINOv2 legacy backend requested, but PyTorch/torchvision are not installed. "
				"Install fextractor[dino-legacy]."
			) from exc

		self.torch = torch

		logger.info(
			"Loading legacy DINOv2 model='%s' requested_device='%s'",
			self.model_name,
			self.requested_device,
		)

		if "cuda" in self.device and not torch.cuda.is_available():
			logger.warning(
				"CUDA requested for DINOv2 but no CUDA device is available; "
				"falling back to CPU"
			)
			self.device = "cpu"

		if "cuda" in self.device:
			logger.info(
				"Using CUDA device='%s' name='%s'",
				self.device,
				torch.cuda.get_device_name(0),
			)
		else:
			logger.info("Using device='%s'", self.device)

		self.model = torch.hub.load(
			"facebookresearch/dinov2",
			self.model_name,
		)

		self.model.to(self.device)
		self.model.eval()

		self.transform = T.Compose([
			T.Resize(
				(self.imgsize, self.imgsize),
				interpolation=T.InterpolationMode.BICUBIC,
			),
			T.ToTensor(),
			T.Normalize(
				mean=(0.485, 0.456, 0.406),
				std=(0.229, 0.224, 0.225),
			),
		])

		logger.info(
			"Legacy DINOv2 model loaded successfully: model='%s' device='%s' imgsize=%d",
			self.model_name,
			self.device,
			self.imgsize,
		)
		

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

		logger.debug(
			"Extracting DINOv2 representation from '%s'",
			source,
		)

		image_tensor = self.prepare(source).to(self.device)

		logger.debug(
			"DINOv2 input tensor shape=%s device='%s'",
			tuple(image_tensor.shape),
			self.device,
		)

		with self.torch.no_grad():
			features = self.model(image_tensor)

		features = features.detach().cpu().numpy()

		if features.ndim > 1 and features.shape[0] == 1:
			features = features[0]

		features = np.asarray(
			features,
			dtype=np.float32,
		).reshape(-1)

		logger.debug(
			"DINOv2 representation extracted: features=%d",
			features.size,
		)

		return features


	def metadata(self) -> dict:
		metadata = super().metadata()
		metadata.update({
			"model": self.model_name,
			"requested_device": self.requested_device,
			"device": self.device,
			"imgsize": self.imgsize,
			"preprocessing": self.preprocessing.__dict__.copy(),
		})
		return metadata


def create(config: ExtractorConfig) -> DINOv2LegacyFeatureExtractor:
	"""Create a legacy torch.hub DINOv2 extractor from configuration."""
	return DINOv2LegacyFeatureExtractor(
		model_name=config.model or DEFAULT_MODEL,
		device=config.device,
		imgsize=config.imgsize if config.imgsize is not None else 224,
		preprocessing=config.preprocessing,
	)
