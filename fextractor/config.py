"""Configuration objects used to create feature extractors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractorConfig:
	"""Backend-neutral configuration for an extractor instance.

	Backend-specific settings should be placed in ``options`` rather than added
	to the CLI/factory every time a new backend is introduced.
	"""

	backend: str
	model: str | None = None
	model_weights: str | None = None
	device: str = "cuda"
	imgsize: int | None = None
	in_chans: int | None = None
	preprocessing: Any = None
	options: dict[str, Any] = field(default_factory=dict)

	def get_option(self, name: str, default: Any = None) -> Any:
		"""Return one backend-specific option."""
		return self.options.get(name, default)
