"""Unit tests for the LiCu ML embedding backend."""

from __future__ import annotations

import numpy as np
import pytest

from fextractor.extractors.timeseries.licu_embed import (
	LiCuEmbeddingFeatureExtractor,
	get_model_spec,
)
from fextractor.timeseries import TimeSeries
from fextractor.preprocessing import (
	TimeSeriesPreprocessConfig,
)

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
	
	
def test_astra_clr_model_spec():
	spec = get_model_spec(
		"/opt/models/astra-clr"
	)

	assert spec.name == "astra-clr"
	assert spec.filename == "astra_clr.onnx"
	assert spec.embedding_dim == 512
	assert spec.default_output == "mean"
	assert spec.multiband is True
	assert spec.requires_errors is True
	assert spec.n_model_bands == 3


def test_atat_model_spec():
	spec = get_model_spec(
		"/opt/models/atat"
	)

	assert spec.name == "atat"
	assert spec.filename == "atat.onnx"
	assert spec.embedding_dim == 192
	assert spec.default_output == "token"
	assert spec.multiband is True
	assert spec.requires_errors is False
	assert spec.n_model_bands == 6


def test_atcat_model_spec_uses_f32_filename():
	spec = get_model_spec(
		"/opt/models/atcat"
	)

	assert spec.name == "atcat"
	assert spec.filename == "atcat_f32.onnx"
	assert spec.embedding_dim == 384
	assert spec.default_output == "last"
	assert spec.multiband is True
	assert spec.requires_errors is True
	
def test_atat_uses_token_default_output():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atat",
		device="cpu",
	)

	assert extractor.output == "token"
	assert (
		extractor.reduction
		== "non-overlapping-windows"
	)


def test_atcat_uses_last_default_output():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atcat",
		device="cpu",
	)

	assert extractor.output == "last"


def test_astra_clr_uses_beginning_reduction():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="astra-clr",
		device="cpu",
	)

	assert extractor.output == "mean"
	assert extractor.reduction == "beginning"
	
def test_atat_normalises_string_band_labels():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atat",
		device="cpu",
	)

	bands = extractor._normalise_multiband_labels(
		np.asarray([
			"u",
			"g",
			"r",
			"i",
			"z",
			"Y",
		])
	)

	assert bands.tolist() == [
		0,
		1,
		2,
		3,
		4,
		5,
	]
	
def test_atat_accepts_integer_band_labels():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atat",
		device="cpu",
	)

	bands = extractor._normalise_multiband_labels(
		np.asarray([
			0,
			1,
			5,
		])
	)

	assert bands.tolist() == [
		0,
		1,
		5,
	]
	
def test_atat_rejects_unknown_band():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atat",
		device="cpu",
	)

	with pytest.raises(
		ValueError,
		match="Unsupported band",
	):
		extractor._normalise_multiband_labels(
			np.asarray([
				"g",
				"unknown",
			])
		)
		
def test_atat_can_ignore_unknown_band():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atat",
		device="cpu",
		allow_extra_bands=True,
	)

	bands = extractor._normalise_multiband_labels(
		np.asarray([
			"g",
			"unknown",
			"r",
		])
	)

	assert bands.tolist() == [
		1,
		-1,
		2,
	]
	
def test_multiband_model_requires_bands():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atat",
		device="cpu",
	)

	extractor.model = object()

	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		times=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
	)

	with pytest.raises(
		ValueError,
		match="requires one band label",
	):
		extractor.extract_timeseries(
			series
		)
		
def test_multiband_model_requires_timestamps():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atat",
		device="cpu",
	)

	extractor.model = object()

	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		bands=np.asarray([
			"g",
			"r",
			"i",
		]),
	)

	with pytest.raises(
		ValueError,
		match="requires explicit timestamps",
	):
		extractor.extract_timeseries(
			series
		)
		
def test_multiband_model_requires_one_value_channel():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atat",
		device="cpu",
	)

	extractor.model = object()

	series = TimeSeries(
		values=np.asarray([
			[1.0, 10.0],
			[2.0, 20.0],
			[3.0, 30.0],
		]),
		times=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		bands=np.asarray([
			"g",
			"r",
			"i",
		]),
	)

	with pytest.raises(
		ValueError,
		match="exactly one value channel",
	):
		extractor.extract_timeseries(
			series
		)
		
def test_atcat_requires_errors():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atcat",
		device="cpu",
	)

	extractor.model = object()

	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		times=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		bands=np.asarray([
			"g",
			"r",
			"i",
		]),
	)

	with pytest.raises(
		ValueError,
		match="requires measurement errors",
	):
		extractor.extract_timeseries(
			series
		)
		
def test_astra_clr_requires_positive_errors():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="astra-clr",
		device="cpu",
	)

	extractor.model = object()

	series = TimeSeries(
		values=np.asarray([
			18.1,
			18.2,
			18.3,
		]),
		times=np.asarray([
			59000.0,
			59001.0,
			59002.0,
		]),
		errors=np.asarray([
			0.1,
			0.0,
			0.2,
		]),
		bands=np.asarray([
			"g",
			"r",
			"i",
		]),
	)

	with pytest.raises(
		ValueError,
		match="strictly positive",
	):
		extractor.extract_timeseries(
			series
		)
		
def test_atat_multiband_embedding():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atat",
		device="cpu",
		aggregation="mean",
	)

	class FakeModel:
		def __init__(self):
			self.calls = 0

		def __call__(
			self,
			time,
			values,
			bands,
		):
			self.calls += 1

			np.testing.assert_allclose(
				time,
				[
					1.0,
					2.0,
					3.0,
					4.0,
				],
			)

			np.testing.assert_allclose(
				values,
				[
					10.0,
					11.0,
					12.0,
					13.0,
				],
			)

			assert bands.tolist() == [
				1,
				2,
				1,
				3,
			]

			return np.ones(
				(
					1,
					1,
					1,
					192,
				),
				dtype=np.float32,
			)

	model = FakeModel()
	extractor.model = model

	series = TimeSeries(
		values=np.asarray([
			10.0,
			11.0,
			12.0,
			13.0,
		]),
		times=np.asarray([
			1.0,
			2.0,
			3.0,
			4.0,
		]),
		bands=np.asarray([
			"g",
			"r",
			"g",
			"i",
		]),
	)

	features = extractor.extract_timeseries(
		series
	)

	assert model.calls == 1
	assert features.shape == (
		192,
	)

	assert np.allclose(
		features,
		1.0,
	)
	
def test_atcat_multiband_embedding_passes_errors():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atcat",
		device="cpu",
		aggregation="mean",
	)

	class FakeModel:
		def __call__(
			self,
			time,
			values,
			errors,
			bands,
		):
			np.testing.assert_allclose(
				errors,
				[
					0.1,
					0.2,
					0.3,
				],
			)

			assert bands.tolist() == [
				1,
				2,
				3,
			]

			return np.full(
				(
					1,
					1,
					1,
					384,
				),
				2.0,
				dtype=np.float32,
			)

	extractor.model = FakeModel()

	series = TimeSeries(
		values=np.asarray([
			10.0,
			11.0,
			12.0,
		]),
		times=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		errors=np.asarray([
			0.1,
			0.2,
			0.3,
		]),
		bands=np.asarray([
			"g",
			"r",
			"i",
		]),
	)

	features = extractor.extract_timeseries(
		series
	)

	assert features.shape == (
		384,
	)

	assert np.allclose(
		features,
		2.0,
	)
	
def test_astra_clr_multiband_embedding_sorts_by_time():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="astra-clr",
		device="cpu",
		aggregation="mean",
	)

	class FakeModel:
		def __call__(
			self,
			time,
			values,
			errors,
			bands,
		):
			np.testing.assert_allclose(
				time,
				[
					59000.0,
					59001.0,
					59002.0,
				],
			)

			np.testing.assert_allclose(
				values,
				[
					18.0,
					18.1,
					18.2,
				],
			)

			np.testing.assert_allclose(
				errors,
				[
					0.1,
					0.2,
					0.3,
				],
			)

			assert bands.tolist() == [
				0,
				1,
				2,
			]

			return np.ones(
				(
					1,
					1,
					1,
					512,
				),
				dtype=np.float32,
			)

	extractor.model = FakeModel()

	series = TimeSeries(
		values=np.asarray([
			18.2,
			18.0,
			18.1,
		]),
		times=np.asarray([
			59002.0,
			59000.0,
			59001.0,
		]),
		errors=np.asarray([
			0.3,
			0.1,
			0.2,
		]),
		bands=np.asarray([
			"i",
			"g",
			"r",
		]),
	)

	features = extractor.extract_timeseries(
		series
	)

	assert features.shape == (
		512,
	)
	
def test_multiband_extra_band_is_filtered():
	extractor = LiCuEmbeddingFeatureExtractor(
		model_name="atat",
		device="cpu",
		allow_extra_bands=True,
		aggregation="mean",
	)

	class FakeModel:
		def __call__(
			self,
			time,
			values,
			bands,
		):
			np.testing.assert_allclose(
				time,
				[
					1.0,
					3.0,
				],
			)

			np.testing.assert_allclose(
				values,
				[
					10.0,
					12.0,
				],
			)

			assert bands.tolist() == [
				1,
				2,
			]

			return np.ones(
				(
					1,
					1,
					1,
					192,
				),
				dtype=np.float32,
			)

	extractor.model = FakeModel()

	series = TimeSeries(
		values=np.asarray([
			10.0,
			999.0,
			12.0,
		]),
		times=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		bands=np.asarray([
			"g",
			"unknown",
			"r",
		]),
	)

	features = extractor.extract_timeseries(
		series
	)

	assert features.shape == (
		192,
	)
	

def test_multiband_models_reject_regularization():
	with pytest.raises(
		ValueError,
		match="does not support.*regularization",
	):
		LiCuEmbeddingFeatureExtractor(
			model_name="atat",
			device="cpu",
			preprocessing=TimeSeriesPreprocessConfig(
				regularize=True,
				cadence=1.0,
			),
		)


def test_multiband_models_reject_value_transform():
	with pytest.raises(
		ValueError,
		match="value_transform='none'",
	):
		LiCuEmbeddingFeatureExtractor(
			model_name="atat",
			device="cpu",
			preprocessing=TimeSeriesPreprocessConfig(
				value_transform="standard",
			),
		)


def test_astra_clr_rejects_time_origin_transform():
	with pytest.raises(
		ValueError,
		match="absolute MJD",
	):
		LiCuEmbeddingFeatureExtractor(
			model_name="astra-clr",
			device="cpu",
			preprocessing=TimeSeriesPreprocessConfig(
				time_transform="origin",
			),
		)


def test_multiband_models_reject_wide_layout():
	with pytest.raises(
		ValueError,
		match="wide layout",
	):
		LiCuEmbeddingFeatureExtractor(
			model_name="atat",
			device="cpu",
			preprocessing=TimeSeriesPreprocessConfig(
				layout="wide",
			),
		)
