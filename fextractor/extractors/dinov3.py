"""Hugging Face DINOv3 representation extractor."""

from __future__ import annotations

from ..config import ExtractorConfig
from .dino_common import HuggingFaceDINOFeatureExtractor


DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


class DINOv3FeatureExtractor(HuggingFaceDINOFeatureExtractor):
	"""Extract DINOv3 image representations using Hugging Face Transformers."""

	backend = "dinov3"


def create(config: ExtractorConfig) -> DINOv3FeatureExtractor:
	"""Create a Hugging Face DINOv3 extractor from configuration."""
	return DINOv3FeatureExtractor(
		model_name=config.model or DEFAULT_MODEL,
		device=config.device,
		preprocessing=config.preprocessing,
	)
