"""Extractor registry and factory."""

from __future__ import annotations

from typing import Type

from .base import FeatureExtractor


_BACKENDS = {
	"tensorflow": "fextractor.extractors.tensorflow:TensorFlowFeatureExtractor",
	"tf": "fextractor.extractors.tensorflow:TensorFlowFeatureExtractor",
	"dinov2": "fextractor.extractors.dino:DINOv2FeatureExtractor",
	"dino": "fextractor.extractors.dino:DINOv2FeatureExtractor",
	"siglip": "fextractor.extractors.siglip:SigLIPFeatureExtractor",
}


def list_backends() -> tuple[str, ...]:
	"""Return canonical backend names."""
	return ("tensorflow", "dinov2", "siglip")


def get_extractor_class(backend: str) -> Type[FeatureExtractor]:
	"""Resolve a backend name to its extractor class without importing all ML frameworks."""
	backend = backend.lower()
	try:
		target = _BACKENDS[backend]
	except KeyError as exc:
		raise KeyError(f"Unknown backend '{backend}'. Available: {', '.join(list_backends())}") from exc

	module_name, class_name = target.split(":", maxsplit=1)
	module = __import__(module_name, fromlist=[class_name])
	return getattr(module, class_name)


def create_extractor(backend: str, **kwargs) -> FeatureExtractor:
	"""Instantiate an extractor backend."""
	return get_extractor_class(backend)(**kwargs)
