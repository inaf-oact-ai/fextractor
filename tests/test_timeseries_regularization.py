import numpy as np
import pytest

from fextractor.timeseries import (
	TimeSeries,
	infer_cadence,
	is_regular_timeseries,
	regularize_timeseries,
)


def test_infer_cadence():
	times = np.asarray([
		0.0,
		1.0,
		2.0,
		3.0,
	])

	assert infer_cadence(
		times
	) == pytest.approx(
		1.0
	)


def test_regular_timeseries():
	times = np.asarray([
		0.0,
		1.0,
		2.0,
		3.0,
	])

	assert is_regular_timeseries(
		times
	)


def test_irregular_timeseries():
	times = np.asarray([
		0.0,
		1.0,
		2.0,
		10.0,
	])

	assert not is_regular_timeseries(
		times
	)


def test_regularize_with_nan_gaps():
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

	output = regularize_timeseries(
		series,
		cadence=1.0,
		missing_strategy="nan",
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

	assert output.values[0, 0] == pytest.approx(
		1.0
	)

	assert output.values[1, 0] == pytest.approx(
		2.0
	)

	assert np.isnan(
		output.values[2, 0]
	)

	assert np.isnan(
		output.values[3, 0]
	)

	assert output.values[4, 0] == pytest.approx(
		5.0
	)


def test_regularize_with_linear_interpolation():
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

	output = regularize_timeseries(
		series,
		cadence=1.0,
		missing_strategy="linear",
	)

	np.testing.assert_allclose(
		output.values[:, 0],
		[
			1.0,
			2.0,
			3.0,
			4.0,
			5.0,
		],
	)


def test_regularize_requires_times():
	series = TimeSeries(
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
	)

	with pytest.raises(
		ValueError,
		match="without timestamps",
	):
		regularize_timeseries(
			series
		)


def test_regularize_rejects_invalid_strategy():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			1.0,
			2.0,
		]),
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
	)

	with pytest.raises(
		ValueError,
		match="missing_strategy",
	):
		regularize_timeseries(
			series,
			missing_strategy="unknown",
		)
		
		
def test_regularization_averages_duplicate_grid_samples():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			0.1,
			1.0,
		]),
		values=np.asarray([
			10.0,
			14.0,
			20.0,
		]),
	)

	output = regularize_timeseries(
		series,
		cadence=1.0,
	)

	assert output.values[0, 0] == pytest.approx(
		12.0
	)

	assert output.values[1, 0] == pytest.approx(
		20.0
	)
