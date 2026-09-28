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
	
	
def test_time_transform_origin():
	series = TimeSeries(
		times=np.asarray([
			59000.0,
			59001.5,
			59004.0,
		]),
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		time_transform="origin",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.times,
		[
			0.0,
			1.5,
			4.0,
		],
	)

	np.testing.assert_allclose(
		output.values[:, 0],
		[
			1.0,
			2.0,
			3.0,
		],
	)

	assert (
		output.metadata["time_transform"]
		== "origin"
	)

	assert (
		output.metadata["time_transform_reference"]
		== 59000.0
	)
	
def test_value_transform_maxabs():
	series = TimeSeries(
		values=np.asarray([
			-2.0,
			4.0,
			8.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		value_transform="maxabs",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.values[:, 0],
		[
			-0.25,
			0.5,
			1.0,
		],
	)
	
def test_value_transform_minmax():
	series = TimeSeries(
		values=np.asarray([
			2.0,
			4.0,
			6.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		value_transform="minmax",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.values[:, 0],
		[
			0.0,
			0.5,
			1.0,
		],
	)
	
	
def test_value_transform_standard():
	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		value_transform="standard",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		np.mean(
			output.values[:, 0]
		),
		0.0,
		atol=1.0e-6,
	)

	np.testing.assert_allclose(
		np.std(
			output.values[:, 0]
		),
		1.0,
		atol=1.0e-6,
	)
	
def test_value_transform_asinh():
	series = TimeSeries(
		values=np.asarray([
			-2.0,
			0.0,
			2.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		value_transform="asinh",
		value_transform_scale=2.0,
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.values[:, 0],
		np.arcsinh([
			-1.0,
			0.0,
			1.0,
		]),
	)
	
def test_value_transform_maxabs_preserves_errors():
	series = TimeSeries(
		values=np.asarray([
			2.0,
			4.0,
		]),
		errors=np.asarray([
			0.2,
			0.4,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		value_transform="maxabs",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.values[:, 0],
		[
			0.5,
			1.0,
		],
	)

	np.testing.assert_allclose(
		output.errors[:, 0],
		[
			0.05,
			0.1,
		],
	)
	
def test_value_transform_asinh_preserves_errors():
	series = TimeSeries(
		values=np.asarray([
			0.0,
			2.0,
		]),
		errors=np.asarray([
			0.2,
			0.4,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		value_transform="asinh",
		value_transform_scale=2.0,
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.errors[:, 0],
		[
			0.2 / 2.0,
			0.4 / np.sqrt(8.0),
		],
	)
	
def test_value_transform_maxabs_is_channel_wise():
	series = TimeSeries(
		values=np.asarray([
			[1.0, 10.0],
			[2.0, 20.0],
			[4.0, 40.0],
		]),
	)

	config = TimeSeriesPreprocessConfig(
		value_transform="maxabs",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.values,
		[
			[0.25, 0.25],
			[0.50, 0.50],
			[1.00, 1.00],
		],
	)
	
	
def test_alignment_peak_max():
	series = TimeSeries(
		times=np.asarray([
			2.1,
			3.4,
			8.7,
			9.0,
			15.2,
		]),
		values=np.asarray([
			1.0,
			2.0,
			9.0,
			7.0,
			1.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		alignment="peak-max",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	assert (
		output.metadata["alignment"]
		== "peak-max"
	)

	assert (
		output.metadata["alignment_anchor_index_original"]
		== 2
	)

	assert (
		output.metadata["alignment_anchor_time_original"]
		== 8.7
	)
	
	
def test_alignment_peak_min():
	series = TimeSeries(
		times=np.asarray([
			10.0,
			20.0,
			30.0,
			40.0,
		]),
		values=np.asarray([
			18.2,
			17.3,
			15.1,
			16.4,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		alignment="peak-min",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	assert (
		output.metadata["alignment_anchor_index_original"]
		== 2
	)

	assert (
		output.metadata["alignment_anchor_time_original"]
		== 30.0
	)
	
	
def test_alignment_peak_abs():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			1.0,
			2.0,
		]),
		values=np.asarray([
			2.0,
			-8.0,
			5.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		alignment="peak-abs",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	assert (
		output.metadata["alignment_anchor_index_original"]
		== 1
	)
	
def test_alignment_anchor_preserves_original_time_with_time_transform():
	series = TimeSeries(
		times=np.asarray([
			59000.0,
			59010.0,
			59020.0,
			59030.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
			10.0,
			4.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		alignment="peak-max",
		time_transform="origin",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.times,
		[
			-20.0,
			-10.0,
			0.0,
			10.0,
		],
	)

	assert (
		output.metadata["alignment_anchor_index_original"]
		== 2
	)

	assert (
		output.metadata["alignment_anchor_time_original"]
		== 59020.0
	)
	
	assert (
		output.metadata["time_transform_reference"]
		== 59000.0
	)

	assert (
		output.metadata["alignment_anchor_time_original"]
		== 59020.0
	)

	assert (
		output.metadata["alignment_anchor_time_aligned"]
		== 0.0
	)
	
	
def test_alignment_shifts_physical_time():
	series = TimeSeries(
		times=np.asarray([
			59000.0,
			59010.0,
			59020.0,
			59030.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
			10.0,
			4.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		alignment="peak-max",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.times,
		[
			-20.0,
			-10.0,
			0.0,
			10.0,
		],
	)

	assert (
		output.metadata[
			"alignment_anchor_time_original"
		]
		== 59020.0
	)

	assert (
		output.metadata[
			"alignment_anchor_time_aligned"
		]
		== 0.0
	)
	
def test_alignment_window():
	series = TimeSeries(
		times=np.asarray([
			59000.0,
			59010.0,
			59020.0,
			59030.0,
			59050.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
			10.0,
			4.0,
			2.0,
		]),
		errors=np.asarray([
			0.1,
			0.2,
			0.3,
			0.2,
			0.1,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		alignment="peak-max",
		alignment_window_before=10.0,
		alignment_window_after=20.0,
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.times,
		[
			-10.0,
			0.0,
			10.0,
		],
	)

	np.testing.assert_allclose(
		output.values[:, 0],
		[
			3.0,
			10.0,
			4.0,
		],
	)

	np.testing.assert_allclose(
		output.errors[:, 0],
		[
			0.2,
			0.3,
			0.2,
		],
	)
	
def test_alignment_after_time_origin_transform():
	series = TimeSeries(
		times=np.asarray([
			59000.0,
			59010.0,
			59020.0,
			59030.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
			10.0,
			4.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		time_transform="origin",
		alignment="peak-max",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.times,
		[
			-20.0,
			-10.0,
			0.0,
			10.0,
		],
	)

	assert (
		output.metadata[
			"time_transform_reference"
		]
		== 59000.0
	)

	assert (
		output.metadata[
			"alignment_anchor_time_original"
		]
		== 59020.0
	)
	
		
	
