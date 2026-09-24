"""Base interfaces for feature extractors."""

from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

class FeatureExtractor(ABC):
	"""Base class for pretrained feature/representation extractors."""

	backend = "base"
	modality = "unknown"

	def __init__(self) -> None:
		self._loaded = False

	@property
	def is_loaded(self) -> bool:
		"""Return whether the backend model has been loaded."""
		return self._loaded

	@abstractmethod
	def load_model(self) -> None:
		"""Load the model and any model-specific processor."""

	@abstractmethod
	def extract(self, source: str | Path) -> np.ndarray:
		"""Extract one feature vector from one source."""

	def ensure_loaded(self) -> None:
	"""Load the model on first use."""
	if not self._loaded:
		logger.info(
			"Loading extractor backend='%s' modality='%s'",
			self.backend,
			self.modality,
		)

		self.load_model()
		self._loaded = True

		logger.info(
			"Extractor backend='%s' loaded successfully",
			self.backend,
		)

	def extract_many(self, sources: Iterable[str | Path]) -> list[np.ndarray]:
		"""Extract representations from several sources."""
		return [self.extract(source) for source in sources]

	def metadata(self) -> dict:
		"""Return minimal extractor metadata suitable for output provenance."""
		return {
			"backend": self.backend,
			"modality": self.modality,
		}
