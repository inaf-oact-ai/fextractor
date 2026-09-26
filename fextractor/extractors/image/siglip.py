"""SigLIP representation extractor."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ...base import FeatureExtractor
from ...config import ExtractorConfig
from ...image.utils import ensure_channels, to_uint8_rgb
from ...image.io import read_image
from ...preprocessing import ImagePreprocessConfig, apply_image_preprocessing

logger = logging.getLogger(__name__)

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
			raise RuntimeError(
				"SigLIP backend requested, but PyTorch/Transformers are not installed. "
				"Install fextractor[siglip]."
			) from exc

		self.torch = torch

		logger.info(
			"Loading SigLIP model='%s' requested_device='%s'",
			self.model_name,
			self.requested_device,
		)

		if "cuda" in self.device and not torch.cuda.is_available():
			logger.warning(
				"CUDA requested for SigLIP but no CUDA device is available; "
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

		self.model = AutoModel.from_pretrained(
			self.model_name,
		).to(self.device)

		self.model.eval()

		self.processor = AutoProcessor.from_pretrained(
			self.model_name,
		)

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
			logger.info("Resetting SigLIP processor mean/std")
			image_processor.image_mean = [0.0, 0.0, 0.0]
			image_processor.image_std = [1.0, 1.0, 1.0]

		if self.reset_rescale:
			logger.info("Disabling SigLIP processor rescaling")
			image_processor.do_rescale = False
			image_processor.rescale_factor = 1.0

		logger.info(
			"SigLIP model loaded successfully: model='%s' device='%s' imgsize=%s",
			self.model_name,
			self.device,
			self.imgsize,
		)


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

		logger.debug(
			"Extracting SigLIP representation from '%s'",
			source,
		)

		image = self.prepare(source)

		inputs = self.processor(
			images=image,
			return_tensors="pt",
		).to(self.device)

		logger.debug(
			"SigLIP processor output keys=%s device='%s'",
			tuple(inputs.keys()),
			self.device,
		)

		with self.torch.no_grad():
			features = self.model.get_image_features(**inputs)

		features = features.detach().cpu().numpy()

		if features.ndim > 1 and features.shape[0] == 1:
			features = features[0]

		features = np.asarray(
			features,
			dtype=np.float32,
		).reshape(-1)

		logger.debug(
			"SigLIP representation extracted: features=%d",
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
