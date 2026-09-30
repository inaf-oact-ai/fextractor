"""Tests for factory dispatch that do not load ML frameworks."""

from fextractor.config import ExtractorConfig
from fextractor.factory import create_extractor


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
			"input_sample_policy": "completed",
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

	assert extractor.input_sample_policy == (
		"completed"
	)
	
	assert extractor.context_length == 2048
	assert extractor.batch_size == 16

	assert extractor.device == "cpu"
	assert not extractor.is_loaded
	
	
def test_licu_factory_without_loading_library():
	config = ExtractorConfig(
		backend="licu",
		device="cpu",
		options={
			"feature_set": "basic",
			"invalid_feature_policy": "error",
			"min_samples": 8,
			"input_sample_policy": "completed",
		},
	)

	extractor = create_extractor(
		config
	)

	assert extractor.backend == "licu"
	assert extractor.modality == "timeseries"
	assert extractor.feature_set == "basic"
	assert extractor.invalid_feature_policy == "error"
	assert extractor.min_samples == 8
	assert not extractor.is_loaded
	assert extractor.input_sample_policy == (
		"completed"
	)
	
def test_licu_embed_factory_without_loading_model():
	config = ExtractorConfig(
		backend="licu_embed",
		model="/opt/models/astromer2",
		device="cpu",
		options={
			"licu_embed_output": "sequence",
			"aggregation": "mean_std",
			"licu_embed_reduction": "end",
			"min_samples": 3,
			"input_sample_policy": "completed",
		},
	)

	extractor = create_extractor(
		config
	)

	assert extractor.backend == "licu_embed"
	assert extractor.modality == "timeseries"

	assert extractor.model_spec.name == (
		"astromer2"
	)

	assert extractor.output == "sequence"
	assert extractor.aggregation == "mean_std"
	assert extractor.reduction == "end"
	assert extractor.min_samples == 3

	assert extractor.device == "cpu"

	assert extractor.input_sample_policy == (
		"completed"
	)

	assert not extractor.is_loaded
	

