"""Unit tests for the single-channel LiCu ML embedding backend."""

from __future__ import annotations

import numpy as np
import pytest

from fextractor.extractors.timeseries.licu_embed import (
	LiCuEmbeddingFeatureExtractor,
	get_model_spec,
)
from fextractor.timeseries import TimeSeries


def test_model_spec_from_directory_name():
	spec = get_model_spec("/opt/models/astromer2")
	assert spec.name == "astromer2"
	assert spec.filename == "astromer2.onnx"
	assert spec.embedding_dim == 256


def test_model_spec_from_moment_file():
	spec = get_model_spec(
		"/opt/models/moment1-base/moment1-base.onnx"
	)
	assert spec.name == "moment1-base"
	assert spec.size == "base"
	assert spec.embedding_dim == 768


def test_astromer_requires_timestamps():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="astromer2",
		device="cpu",
	)
	extractor.model = lambda *args: np.zeros((1, 1, 1, 256), dtype=np.float32)

	series = TimeSeries(
		values=np.arange(8, dtype=np.float32),
	)

	with pytest.raises(ValueError, match="requires explicit timestamps"):
		extractor.extract_timeseries(series)


def test_multichannel_embeddings_are_concatenated():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="astromer2",
		device="cpu",
		aggregation="mean",
	)

	class FakeModel:
		def __call__(self, time, values):
			value = float(np.mean(values))
			return np.full((1, 1, 1, 256), value, dtype=np.float32)

	extractor.model = FakeModel()

	time = np.arange(8, dtype=np.float64)
	series = TimeSeries(
		values=np.column_stack((
			np.arange(8, dtype=np.float32),
			np.arange(8, dtype=np.float32) + 10,
		)),
		times=time,
		channel_names=("a", "b"),
	)

	features = extractor.extract_timeseries(series)
	assert features.shape == (512,)
	assert np.allclose(features[:256], 3.5)
	assert np.allclose(features[256:], 13.5)


def test_moment_ignores_timestamps_but_preserves_order_without_them():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="moment1-small",
		device="cpu",
		aggregation="mean",
	)

	class FakeModel:
		def __call__(self, values):
			assert np.array_equal(values, np.array([3.0, 1.0, 2.0]))
			return np.ones((1, 1, 1, 512), dtype=np.float32)

	extractor.model = FakeModel()
	series = TimeSeries(
		values=np.array([3.0, 1.0, 2.0], dtype=np.float32),
	)

	features = extractor.extract_timeseries(series)
	assert features.shape == (512,)


def test_sequence_output_uses_fextractor_aggregation():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="astromer2",
		device="cpu",
		output="sequence",
		aggregation="mean_std",
	)

	class FakeModel:
		def __call__(self, time, values):
			base = np.arange(256, dtype=np.float32)
			return np.stack((base, base + 2.0), axis=0)[None, None, ...]

	extractor.model = FakeModel()
	series = TimeSeries(
		values=np.arange(5, dtype=np.float32),
		times=np.arange(5, dtype=np.float64),
	)

	features = extractor.extract_timeseries(series)
	assert features.shape == (512,)
	assert np.allclose(features[:256], np.arange(256) + 1.0)
	assert np.allclose(features[256:], 1.0)
