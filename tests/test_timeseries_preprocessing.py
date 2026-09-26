import numpy as np

from fextractor.preprocessing import (
	TimeSeriesPreprocessConfig,
	apply_timeseries_preprocessing,
	get_profile,
)
from fextractor.timeseries import TimeSeries


def test_timeseries_default_profile():
	profile = get_profile(
		"default",
		modality="timeseries",
	)

	assert profile.name == "default"
	assert isinstance(
		profile.preprocessing,
		TimeSeriesPreprocessConfig,
	)


def test_lightcurve_profile():
	profile = get_profile(
		"lightcurve",
		modality="timeseries",
	)

	config = profile.preprocessing

	assert config.time_column == "time"
	assert config.value_columns == ("flux",)
	assert config.sort_time is True
	assert config.regularize is False
	assert config.missing_strategy == "nan"


def test_image_default_profile_still_works():
	profile = get_profile(
		"default",
	)

	assert profile.name == "default"


def test_sort_timeseries():
	series = TimeSeries(
		times=np.asarray([
			3.0,
			1.0,
			2.0,
		]),
		values=np.asarray([
			30.0,
			10.0,
			20.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		sort_time=True,
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.times,
		[
			1.0,
			2.0,
			3.0,
		],
	)

	np.testing.assert_allclose(
		output.values[:, 0],
		[
			10.0,
			20.0,
			30.0,
		],
	)


def test_sort_timeseries_preserves_errors():
	series = TimeSeries(
		times=np.asarray([
			3.0,
			1.0,
			2.0,
		]),
		values=np.asarray([
			30.0,
			10.0,
			20.0,
		]),
		errors=np.asarray([
			0.3,
			0.1,
			0.2,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		sort_time=True,
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.errors[:, 0],
		[
			0.1,
			0.2,
			0.3,
		],
	)


def test_sort_timeseries_preserves_observed_mask():
	series = TimeSeries(
		times=np.asarray([
			3.0,
			1.0,
			2.0,
		]),
		values=np.asarray([
			30.0,
			np.nan,
			20.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		sort_time=True,
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	assert output.observed_mask[:, 0].tolist() == [
		False,
		True,
		True,
	]


def test_irregular_series_is_unchanged_when_regularization_disabled():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			1.0,
			4.0,
		]),
		values=np.asarray([
			1.0,
			2.0,
			5.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		regularize=False,
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.times,
		[
			0.0,
			1.0,
			4.0,
		],
	)

	np.testing.assert_allclose(
		output.values[:, 0],
		[
			1.0,
			2.0,
			5.0,
		],
	)


def test_regularization_from_preprocessing_config():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			1.0,
			4.0,
		]),
		values=np.asarray([
			1.0,
			2.0,
			5.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		regularize=True,
		cadence=1.0,
		missing_strategy="nan",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	assert output.values.shape == (
		5,
		1,
	)

	np.testing.assert_allclose(
		output.times,
		[
			0.0,
			1.0,
			2.0,
			3.0,
			4.0,
		],
	)

	assert np.isnan(
		output.values[2, 0]
	)

	assert np.isnan(
		output.values[3, 0]
	)
