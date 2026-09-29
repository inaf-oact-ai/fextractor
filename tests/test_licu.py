"""Unit tests for the light-curve-python (LiCu) backend."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("light_curve")

from fextractor.extractors.timeseries.licu import LiCuFeatureExtractor
from fextractor.timeseries import TimeSeries


def make_irregular_series() -> TimeSeries:
	time = np.array(
		[0.0, 0.8, 2.1, 3.0, 5.7, 8.4, 9.1, 12.8],
		dtype=np.float64,
	)

	values = np.column_stack((
		1.0 + 0.2 * np.sin(time),
		2.0 + 0.1 * np.cos(0.5 * time),
	)).astype(np.float32)

	return TimeSeries(
		values=values,
		times=time,
		channel_names=(
			"flux_ratio",
			"flare_history",
		),
	)


def test_licu_extracts_fixed_multichannel_schema():
	extractor = LiCuFeatureExtractor(
		feature_set="basic",
		min_samples=5,
	)

	features = extractor.extract(
		make_irregular_series()
	)

	assert features.ndim == 1
	assert np.all(
		np.isfinite(features)
	)
	assert len(features) == 2 * len(
		extractor.base_feature_names
	)
	assert len(
		extractor.feature_names
	) == len(features)
	assert extractor.feature_names[0].startswith(
		"flux_ratio."
	)
	assert extractor.feature_names[
		len(extractor.base_feature_names)
	].startswith(
		"flare_history."
	)


def test_licu_requires_timestamps():
	extractor = LiCuFeatureExtractor(
		feature_set="basic",
	)

	series = TimeSeries(
		values=np.arange(
			8,
			dtype=np.float32,
		),
	)

	with pytest.raises(
		ValueError,
		match="requires explicit timestamps",
	):
		extractor.extract(
			series
		)


def test_licu_filters_nonfinite_observations():
	extractor = LiCuFeatureExtractor(
		feature_set="basic",
		min_samples=5,
	)

	series = make_irregular_series()
	series.values[
		1,
		0,
	] = np.nan

	features = extractor.extract(
		series
	)

	assert np.all(
		np.isfinite(features)
	)

	metadata = extractor.metadata()
	assert metadata[
		"channels"
	][0][
		"n_valid"
	] == 7


def test_licu_rejects_too_few_valid_samples():
	extractor = LiCuFeatureExtractor(
		feature_set="basic",
		min_samples=5,
	)

	series = TimeSeries(
		values=np.array(
			[1.0, 2.0, 3.0, 4.0],
			dtype=np.float32,
		),
		times=np.array(
			[0.0, 1.0, 2.0, 3.0],
			dtype=np.float64,
		),
		channel_names=("flux",),
	)

	with pytest.raises(
		ValueError,
		match="at least 5 are required",
	):
		extractor.extract(
			series
		)


def test_licu_metadata_contains_named_features():
	extractor = LiCuFeatureExtractor(
		feature_set="basic",
	)

	extractor.extract(
		make_irregular_series()
	)

	metadata = extractor.metadata()

	assert metadata[
		"backend"
	] == "licu"
	assert metadata[
		"library"
	] == "light-curve"
	assert metadata[
		"feature_set"
	] == "basic"
	assert metadata[
		"n_features"
	] == len(
		extractor.feature_names
	)
	assert metadata[
		"feature_names"
	] == list(
		extractor.feature_names
	)
