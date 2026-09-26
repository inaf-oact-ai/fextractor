import numpy as np
import pytest

from fextractor.extractors.timeseries.chronos2 import (
	Chronos2FeatureExtractor,
)
from fextractor.timeseries import TimeSeries


class FakeChronosPipeline:

	def __init__(
		self,
		embeddings,
	):
		self.embeddings = embeddings
		self.last_inputs = None
		self.last_batch_size = None
		self.last_context_length = None

	def embed(
		self,
		inputs,
		batch_size=256,
		context_length=None,
	):
		self.last_inputs = inputs
		self.last_batch_size = batch_size
		self.last_context_length = context_length

		return (
			[
				self.embeddings,
			],
			[
				(
					None,
					None,
				),
			],
		)


def test_chronos_univariate_input_conversion():
	extractor = Chronos2FeatureExtractor(
		device="cpu",
	)

	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		times=np.asarray([
			0.0,
			1.0,
			2.0,
		]),
	)

	values = extractor._to_chronos_input(
		series
	)

	assert values.shape == (
		3,
	)

	np.testing.assert_allclose(
		values,
		[
			1.0,
			2.0,
			3.0,
		],
	)


def test_chronos_multivariate_input_conversion():
	extractor = Chronos2FeatureExtractor(
		device="cpu",
	)

	series = TimeSeries(
		values=np.asarray([
			[
				1.0,
				10.0,
			],
			[
				2.0,
				20.0,
			],
			[
				3.0,
				30.0,
			],
		]),
		times=np.asarray([
			0.0,
			1.0,
			2.0,
		]),
	)

	values = extractor._to_chronos_input(
		series
	)

	assert values.shape == (
		2,
		3,
	)


def test_chronos_rejects_irregular_timestamps():
	extractor = Chronos2FeatureExtractor(
		device="cpu",
	)

	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		times=np.asarray([
			0.0,
			1.0,
			5.0,
		]),
	)

	with pytest.raises(
		ValueError,
		match="regularly sampled",
	):
		extractor._to_chronos_input(
			series
		)


def test_chronos_token_split():
	d_model = 4
	n_patches = 3

	embeddings = np.arange(
		1
		* (
			n_patches
			+ 2
		)
		* d_model,
		dtype=np.float32,
	).reshape(
		1,
		n_patches + 2,
		d_model,
	)

	extractor = Chronos2FeatureExtractor(
		device="cpu",
	)

	extractor.pipeline = FakeChronosPipeline(
		embeddings
	)

	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
			4.0,
		]),
	)

	representation = extractor.extract_tokens(
		series
	)

	assert representation.context_tokens.shape == (
		1,
		n_patches,
		d_model,
	)

	assert representation.special_tokens[
		"reg"
	].shape == (
		1,
		1,
		d_model,
	)

	assert representation.special_tokens[
		"future"
	].shape == (
		1,
		1,
		d_model,
	)


def test_chronos_mean_std_representation():
	embeddings = np.asarray(
		[
			[
				[
					1.0,
					2.0,
				],
				[
					3.0,
					4.0,
				],
				[
					5.0,
					6.0,
				],
				[
					100.0,
					100.0,
				],
				[
					200.0,
					200.0,
				],
			],
		],
		dtype=np.float32,
	)

	extractor = Chronos2FeatureExtractor(
		device="cpu",
		aggregation="mean_std",
	)

	extractor.pipeline = FakeChronosPipeline(
		embeddings
	)

	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
	)

	features = extractor.extract_timeseries(
		series
	)

	assert features.shape == (
		4,
	)

	np.testing.assert_allclose(
		features[:2],
		[
			3.0,
			4.0,
		],
	)
