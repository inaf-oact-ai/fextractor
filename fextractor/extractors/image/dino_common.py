"""Shared Hugging Face DINO representation extractor."""

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np
from PIL import Image

from ...base import FeatureExtractor
from ...image.utils import to_uint8_rgb
from ...image.io import read_image
from ...preprocessing import ImagePreprocessConfig, apply_image_preprocessing

logger = logging.getLogger(__name__)


class HuggingFaceDINOFeatureExtractor(FeatureExtractor):
	"""Base extractor for Hugging Face DINO-family image backbones."""

	modality = "image"

	def __init__(
		self,
		model_name: str,
		device: str = "cuda",
		preprocessing: ImagePreprocessConfig | None = None,
	) -> None:
		super().__init__()
		self.model_name = model_name
		self.requested_device = device
		self.device = device
		self.preprocessing = preprocessing or ImagePreprocessConfig()
		self.model = None
		self.processor = None
		self.torch = None

	def load_model(self) -> None:
		"""Load the Hugging Face DINO model and image processor."""
		try:
			import torch
			from transformers import AutoImageProcessor, AutoModel
		except ImportError as exc:
			raise RuntimeError(
				f"{self.backend} backend requested, but PyTorch/Transformers are not installed. "
				"Install the appropriate fextractor DINO dependency group."
			) from exc

		self.torch = torch

		logger.info(
			"Loading %s model='%s' requested_device='%s'",
			self.backend,
			self.model_name,
			self.requested_device,
		)

		if "cuda" in self.device and not torch.cuda.is_available():
			logger.warning(
				"CUDA requested for %s but no CUDA device is available; falling back to CPU",
				self.backend,
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

		self.processor = AutoImageProcessor.from_pretrained(
			self.model_name,
		)

		self.model = AutoModel.from_pretrained(
			self.model_name,
		).to(self.device)

		self.model.eval()

		logger.info(
			"%s model loaded successfully: model='%s' device='%s'",
			self.backend,
			self.model_name,
			self.device,
		)

	def prepare(self, source: str | Path) -> Image.Image:
		"""Read one image and apply scientific preprocessing before HF processing."""
		data = read_image(source)
		data = apply_image_preprocessing(data, self.preprocessing)
		data = to_uint8_rgb(data)
		return Image.fromarray(data, mode="RGB")

	def extract(self, source: str | Path) -> np.ndarray:
		"""Return one pooled DINO image representation vector."""
		self.ensure_loaded()

		logger.debug(
			"Extracting %s representation from '%s'",
			self.backend,
			source,
		)

		image = self.prepare(source)

		inputs = self.processor(
			images=image,
			return_tensors="pt",
		).to(self.device)

		logger.debug(
			"%s processor output keys=%s device='%s'",
			self.backend,
			tuple(inputs.keys()),
			self.device,
		)

		with self.torch.inference_mode():
			outputs = self.model(**inputs)

		if not hasattr(outputs, "pooler_output") or outputs.pooler_output is None:
			raise RuntimeError(
				f"{self.backend} model '{self.model_name}' did not return pooler_output"
			)

		features = outputs.pooler_output.detach().cpu().numpy()

		if features.ndim > 1 and features.shape[0] == 1:
			features = features[0]

		features = np.asarray(
			features,
			dtype=np.float32,
		).reshape(-1)

		logger.debug(
			"%s representation extracted: features=%d",
			self.backend,
			features.size,
		)

		return features

	def metadata(self) -> dict:
		"""Return model and preprocessing provenance."""
		metadata = super().metadata()

		metadata.update({
			"model": self.model_name,
			"requested_device": self.requested_device,
			"device": self.device,
			"feature_type": "pooler_output",
			"preprocessing": self.preprocessing.__dict__.copy(),
		})

		return metadata
