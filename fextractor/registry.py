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
		class_path="fextractor.extractors.image.tensorflow:TensorFlowFeatureExtractor",
		factory_path="fextractor.extractors.image.tensorflow:create",
		aliases=("tf",),
		dependency_group="tensorflow",
	),
	BackendSpec(
		name="dinov2",
		modality="image",
		class_path="fextractor.extractors.image.dinov2:DINOv2FeatureExtractor",
		factory_path="fextractor.extractors.image.dinov2:create",
		aliases=("dino",),
		default_model="facebook/dinov2-small",
		dependency_group="dino",
	),
	BackendSpec(
		name="dinov3",
		modality="image",
		class_path="fextractor.extractors.image.dinov3:DINOv3FeatureExtractor",
		factory_path="fextractor.extractors.image.dinov3:create",
		default_model="facebook/dinov3-vits16-pretrain-lvd1689m",
		dependency_group="dino",
	),
	BackendSpec(
		name="dinov2_legacy",
		modality="image",
		class_path="fextractor.extractors.image.dinov2_legacy:DINOv2LegacyFeatureExtractor",
		factory_path="fextractor.extractors.image.dinov2_legacy:create",
		default_model="dinov2_vits14",
		dependency_group="dino-legacy",
	),
	BackendSpec(
		name="siglip",
		modality="image",
		class_path="fextractor.extractors.image.siglip:SigLIPFeatureExtractor",
		factory_path="fextractor.extractors.image.siglip:create",
		default_model="google/siglip-so400m-patch14-384",
		dependency_group="siglip",
	),
	BackendSpec(
		name="siglip2",
		modality="image",
		class_path="fextractor.extractors.image.siglip2:SigLIP2FeatureExtractor",
		factory_path="fextractor.extractors.image.siglip2:create",
		default_model="google/siglip2-so400m-patch14-384",
		dependency_group="siglip2",
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
	
def get_backend_modality(name: str) -> str:
	"""Return the modality associated with a registered backend."""
	return get_backend_spec(name).modality
	
def list_modalities() -> tuple[str, ...]:
	"""Return the modalities exposed by registered backends."""
	return tuple(
		dict.fromkeys(
			spec.modality
			for spec in _BACKEND_SPECS
		)
	)

