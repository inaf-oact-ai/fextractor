"""Hugging Face DINOv2 representation extractor."""

from __future__ import annotations

from ...config import ExtractorConfig
from .dino_common import HuggingFaceDINOFeatureExtractor


DEFAULT_MODEL = "facebook/dinov2-small"


class DINOv2FeatureExtractor(HuggingFaceDINOFeatureExtractor):
	"""Extract DINOv2 image representations using Hugging Face Transformers."""

	backend = "dinov2"


def create(config: ExtractorConfig) -> DINOv2FeatureExtractor:
	"""Create a Hugging Face DINOv2 extractor from configuration."""
	return DINOv2FeatureExtractor(
		model_name=config.model or DEFAULT_MODEL,
		device=config.device,
		preprocessing=config.preprocessing,
	)
