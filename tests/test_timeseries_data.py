import numpy as np
import pytest

from fextractor.timeseries import TimeSeries


def test_univariate_series_is_normalized_to_2d():
	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
	)

	assert series.values.shape == (
		3,
		1,
	)

	assert series.n_time == 3
	assert series.n_variates == 1
	assert series.is_univariate


def test_multivariate_series():
	series = TimeSeries(
		values=np.asarray([
			[1.0, 10.0],
			[2.0, 20.0],
			[3.0, 30.0],
		]),
		channel_names=(
			"flux_g",
			"flux_r",
		),
	)

	assert series.values.shape == (
		3,
		2,
	)

	assert series.n_time == 3
	assert series.n_variates == 2
	assert not series.is_univariate


def test_observed_mask_is_generated_from_finite_values():
	series = TimeSeries(
		values=np.asarray([
			1.0,
			np.nan,
			3.0,
		]),
	)

	assert series.observed_mask.shape == (
		3,
		1,
	)

	assert series.observed_mask[:, 0].tolist() == [
		True,
		False,
		True,
	]


def test_time_length_must_match_values():
	with pytest.raises(
		ValueError,
		match="Time coordinate length",
	):
		TimeSeries(
			values=np.asarray([
				1.0,
				2.0,
				3.0,
			]),
			times=np.asarray([
				0.0,
				1.0,
			]),
		)


def test_error_shape_must_match_values():
	with pytest.raises(
		ValueError,
		match="errors shape",
	):
		TimeSeries(
			values=np.asarray([
				1.0,
				2.0,
				3.0,
			]),
			errors=np.asarray([
				0.1,
				0.2,
			]),
		)


def test_channel_names_must_match_variates():
	with pytest.raises(
		ValueError,
		match="channel_names length",
	):
		TimeSeries(
			values=np.asarray([
				[1.0, 2.0],
				[3.0, 4.0],
			]),
			channel_names=(
				"only_one",
			),
		)


def test_copy_is_independent():
	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		times=np.asarray([
			10.0,
			20.0,
			30.0,
		]),
		metadata={
			"source": "example",
		},
	)

	copied = series.copy()

	copied.values[0, 0] = 100.0
	copied.times[0] = -1.0
	copied.metadata["source"] = "changed"

	assert series.values[0, 0] == 1.0
	assert series.times[0] == 10.0
	assert series.metadata["source"] == "example"
