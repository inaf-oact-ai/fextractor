"""Extractor backend registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from .base import FeatureExtractor


@dataclass(frozen=True)
class BackendSpec:
	"""Description of one extractor backend."""

	name: str
	modality: str
	class_path: str
	factory_path: str
	aliases: tuple[str, ...] = ()
	default_model: str | None = None
	dependency_group: str | None = None


_BACKEND_SPECS = (
	BackendSpec(
		name="tensorflow",
		modality="image",
		class_path="fextractor.extractors.tensorflow:TensorFlowFeatureExtractor",
		factory_path="fextractor.extractors.tensorflow:create",
		aliases=("tf",),
		dependency_group="tensorflow",
	),
	BackendSpec(
		name="dinov2",
		modality="image",
		class_path="fextractor.extractors.dino:DINOv2FeatureExtractor",
		factory_path="fextractor.extractors.dino:create",
		aliases=("dino",),
		default_model="dinov2_vits14",
		dependency_group="dino",
	),
	BackendSpec(
		name="siglip",
		modality="image",
		class_path="fextractor.extractors.siglip:SigLIPFeatureExtractor",
		factory_path="fextractor.extractors.siglip:create",
		default_model="google/siglip-so400m-patch14-384",
		dependency_group="siglip",
	),
)

_BACKENDS: dict[str, BackendSpec] = {}
for _spec in _BACKEND_SPECS:
	_BACKENDS[_spec.name] = _spec
	for _alias in _spec.aliases:
		_BACKENDS[_alias] = _spec


def _import_object(target: str):
	"""Import ``module:object`` lazily."""
	module_name, object_name = target.split(":", maxsplit=1)
	module = __import__(module_name, fromlist=[object_name])
	return getattr(module, object_name)


def list_backends(modality: str | None = None) -> tuple[str, ...]:
	"""Return canonical backend names, optionally filtered by modality."""
	names = []
	for spec in _BACKEND_SPECS:
		if modality is None or spec.modality == modality:
			names.append(spec.name)
	return tuple(names)


def get_backend_spec(backend: str) -> BackendSpec:
	"""Return the backend specification for a name or alias."""
	key = backend.lower()
	try:
		return _BACKENDS[key]
	except KeyError as exc:
		available = ", ".join(list_backends())
		raise KeyError(f"Unknown backend '{backend}'. Available: {available}") from exc


def get_extractor_class(backend: str) -> Type[FeatureExtractor]:
	"""Resolve a backend name to its extractor class lazily."""
	spec = get_backend_spec(backend)
	return _import_object(spec.class_path)


def get_extractor_factory(backend: str) -> Callable:
	"""Resolve a backend name to its configuration factory lazily."""
	spec = get_backend_spec(backend)
	return _import_object(spec.factory_path)
