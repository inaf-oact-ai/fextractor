"""Extractor construction helpers."""

from __future__ import annotations

from .base import FeatureExtractor
from .config import ExtractorConfig
from .registry import get_backend_spec, get_extractor_class, get_extractor_factory


def create_extractor(config: ExtractorConfig | str, **kwargs) -> FeatureExtractor:
	"""Create an extractor.

	The preferred API accepts :class:`ExtractorConfig` and dispatches to the
	backend-local factory function. Passing a backend string plus keyword
	arguments is retained as a compatibility shortcut.
	"""
	if isinstance(config, str):
		return get_extractor_class(config)(**kwargs)

	spec = get_backend_spec(config.backend)
	factory = get_extractor_factory(spec.name)
	return factory(config)
