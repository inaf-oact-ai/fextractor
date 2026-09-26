"""Tests for registry/factory dispatch that do not load ML frameworks."""

from fextractor.config import ExtractorConfig
from fextractor.factory import create_extractor
from fextractor.registry import get_backend_spec, list_backends


def test_list_backends():
	expected = (
		"tensorflow",
		"dinov2",
		"dinov3",
		"dinov2_legacy",
		"siglip",
		"siglip2",
		"chronos2",
	)

	assert list_backends() == expected

	assert list_backends(
		modality="image",
	) == (
		"tensorflow",
		"dinov2",
		"dinov3",
		"dinov2_legacy",
		"siglip",
		"siglip2",
	)

	assert list_backends(
		modality="timeseries",
	) == (
		"chronos2",
	)


def test_backend_alias():
	spec = get_backend_spec("tf")
	assert spec.name == "tensorflow"
	assert spec.modality == "image"


def test_dino_factory_without_loading_model():
	config = ExtractorConfig(backend="dinov2", model="dinov2_vits14", device="cpu")
	extractor = create_extractor(config)
	assert extractor.backend == "dinov2"
	assert extractor.model_name == "dinov2_vits14"
	assert extractor.device == "cpu"
	assert not extractor.is_loaded


def test_siglip_backend_options():
	config = ExtractorConfig(
		backend="siglip",
		options={"reset_meanstd": True, "reset_rescale": True},
	)
	extractor = create_extractor(config)
	assert extractor.reset_meanstd is True
	assert extractor.reset_rescale is True
	assert not extractor.is_loaded


def test_legacy_factory_api():
	extractor = create_extractor("dinov2", model_name="dinov2_vits14", device="cpu")
	assert extractor.backend == "dinov2"
	assert extractor.device == "cpu"
	
def test_chronos_factory_without_loading_model():
	config = ExtractorConfig(
		backend="chronos2",
		device="cpu",
		options={
			"aggregation": "mean_std",
			"context_length": 2048,
			"batch_size": 16,
		},
	)

	extractor = create_extractor(
		config
	)

	assert extractor.backend == "chronos2"
	assert extractor.modality == "timeseries"

	assert extractor.model_name == (
		"amazon/chronos-2"
	)

	assert extractor.aggregation == (
		"mean_std"
	)

	assert extractor.context_length == 2048
	assert extractor.batch_size == 16

	assert extractor.device == "cpu"
	assert not extractor.is_loaded
