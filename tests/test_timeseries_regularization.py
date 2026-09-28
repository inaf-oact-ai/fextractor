import numpy as np
import pytest

from fextractor.timeseries import (
	TimeSeries,
	infer_cadence,
	is_regular_timeseries,
	regularize_timeseries,
)

from fextractor.preprocessing import (
	TimeSeriesPreprocessConfig,
	apply_timeseries_preprocessing,
	get_profile,
)

from fextractor.timeseries.regularization import (
	make_regular_grid,
)

from fextractor.timeseries.gp import (
	regularize_timeseries_gp,
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

	assert output.observed_mask[:, 0].tolist() == [
		True,
		True,
		False,
		False,
		True,
	]

	assert output.interpolated_mask[:, 0].tolist() == [
		False,
		False,
		True,
		True,
		False,
	]


def test_regularize_with_pchip_interpolation():
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
		missing_strategy="pchip",
	)

	assert np.all(
		np.isfinite(
			output.values[
				1:5,
				0,
			]
		)
	)

	assert output.interpolated_mask[:, 0].tolist() == [
		False,
		False,
		True,
		True,
		False,
	]
	
def test_regularize_with_akima_interpolation():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			1.0,
			3.0,
			5.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
			2.0,
			4.0,
		]),
	)

	output = regularize_timeseries(
		series,
		cadence=1.0,
		missing_strategy="akima",
	)

	assert np.isfinite(
		output.values[2, 0]
	)

	assert np.isfinite(
		output.values[4, 0]
	)

	assert output.interpolated_mask[2, 0]
	assert output.interpolated_mask[4, 0]
	
	
def test_regularize_with_cubic_interpolation():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			1.0,
			3.0,
			5.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
			2.0,
			4.0,
		]),
	)

	output = regularize_timeseries(
		series,
		cadence=1.0,
		missing_strategy="cubic",
	)

	assert np.isfinite(
		output.values[2, 0]
	)

	assert np.isfinite(
		output.values[4, 0]
	)

	assert output.interpolated_mask[2, 0]
	assert output.interpolated_mask[4, 0]
	
def test_interpolation_does_not_extrapolate():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			2.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
		]),
	)

	output = regularize_timeseries(
		series,
		cadence=1.0,
		grid_start=-2.0,
		grid_stop=4.0,
		missing_strategy="pchip",
	)

	assert np.isnan(
		output.values[0, 0]
	)

	assert np.isnan(
		output.values[1, 0]
	)

	assert np.isnan(
		output.values[-1, 0]
	)

	assert np.isnan(
		output.values[-2, 0]
	)

	assert output.interpolated_mask[:, 0].tolist() == [
		False,
		False,
		False,
		True,
		False,
		False,
		False,
	]	

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
	
	
def test_regularization_explicit_grid_bounds():
	series = TimeSeries(
		times=np.asarray([
			-2.0,
			0.0,
			2.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
			5.0,
		]),
	)

	output = regularize_timeseries(
		series,
		cadence=1.0,
		grid_start=-4.0,
		grid_stop=4.0,
	)

	np.testing.assert_allclose(
		output.times,
		[
			-4.0,
			-3.0,
			-2.0,
			-1.0,
			0.0,
			1.0,
			2.0,
			3.0,
			4.0,
		],
	)

	assert output.values.shape == (
		9,
		1,
	)

	assert np.isnan(
		output.values[0, 0]
	)

	assert np.isnan(
		output.values[-1, 0]
	)

	assert (
		output.observed_mask[:, 0].tolist()
		== [
			False,
			False,
			True,
			False,
			True,
			False,
			True,
			False,
			False,
		]
	)	
	
	
def test_alignment_regularization_produces_fixed_anchor_bin():
	series = TimeSeries(
		times=np.asarray([
			59015.0,
			59018.0,
			59020.0,
			59024.0,
		]),
		values=np.asarray([
			1.0,
			4.0,
			10.0,
			3.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		alignment="peak-max",
		alignment_window_before=5.0,
		alignment_window_after=5.0,
		regularize=True,
		cadence=1.0,
		missing_strategy="nan",
	)

	output = apply_timeseries_preprocessing(
		series,
		config,
	)

	np.testing.assert_allclose(
		output.times,
		np.arange(
			-5.0,
			6.0,
			1.0,
		),
	)

	assert (
		output.metadata[
			"alignment_anchor_index_aligned"
		]
		== 5
	)

	assert (
		output.values.shape
		== (
			11,
			1,
		)
	)

	assert (
		output.values[5, 0]
		== 10.0
	)
	
def test_alignment_window_after_must_match_cadence():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			1.0,
			2.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
			2.0,
		]),
	)

	config = TimeSeriesPreprocessConfig(
		alignment="peak-max",
		alignment_window_before=1.0,
		alignment_window_after=1.5,
		regularize=True,
		cadence=1.0,
	)

	with pytest.raises(
		ValueError,
		match="alignment_window_after",
	):
		apply_timeseries_preprocessing(
			series,
			config,
		)
		
def test_regularization_does_not_bin_samples_outside_grid():
	series = TimeSeries(
		times=np.asarray([
			-5.4,
			-5.0,
			0.0,
			5.0,
			5.4,
		]),
		values=np.asarray([
			100.0,
			1.0,
			2.0,
			3.0,
			200.0,
		]),
	)

	output = regularize_timeseries(
		series,
		cadence=1.0,
		grid_start=-5.0,
		grid_stop=5.0,
	)

	assert output.values[0, 0] == pytest.approx(
		1.0
	)

	assert output.values[-1, 0] == pytest.approx(
		3.0
	)
	
	
def test_linear_interpolation_preserves_observed_mask():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			2.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
		]),
	)

	output = regularize_timeseries(
		series,
		cadence=1.0,
		missing_strategy="linear",
	)

	assert output.observed_mask[:, 0].tolist() == [
		True,
		False,
		True,
	]

	assert output.interpolated_mask[:, 0].tolist() == [
		False,
		True,
		False,
	]

	assert output.values[1, 0] == pytest.approx(
		2.0
	)
	
def test_make_regular_grid():
	grid = make_regular_grid(
		start=-2.0,
		stop=3.0,
		cadence=1.0,
	)

	np.testing.assert_allclose(
		grid,
		[
			-2.0,
			-1.0,
			0.0,
			1.0,
			2.0,
			3.0,
		],
	)
	
def test_make_regular_grid_requires_integer_number_of_intervals():
	with pytest.raises(
		ValueError,
		match="integer multiple",
	):
		make_regular_grid(
			start=-2.0,
			stop=3.0,
			cadence=2.0,
		)
		
def test_regularization_method_defaults_to_bin():
	config = TimeSeriesPreprocessConfig()

	assert (
		config.regularization_method
		== "bin"
	)
	
def test_regularization_records_method():
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

	output = regularize_timeseries(
		series,
		cadence=1.0,
	)

	assert (
		output.metadata[
			"regularization_method"
		]
		== "bin"
	)
	
	
def test_gp_regularization_basic():
	times = np.asarray([
		0.0,
		1.5,
		3.2,
		5.0,
	])

	values = np.asarray([
		0.0,
		1.0,
		0.5,
		0.0,
	])

	errors = np.asarray([
		0.1,
		0.1,
		0.1,
		0.1,
	])

	series = TimeSeries(
		times=times,
		values=values,
		errors=errors,
	)

	grid = make_regular_grid(
		start=0.0,
		stop=5.0,
		cadence=1.0,
	)

	output = regularize_timeseries_gp(
		series,
		grid=grid,
		rho=1.0,
	)

	np.testing.assert_allclose(
		output.times,
		grid,
	)

	assert output.values.shape == (
		grid.size,
		1,
	)

	assert output.errors.shape == (
		grid.size,
		1,
	)

	assert np.all(
		np.isfinite(
			output.values[:, 0]
		)
	)

	assert np.all(
		np.isfinite(
			output.errors[:, 0]
		)
	)

	assert (
		output.metadata[
			"regularization_method"
		]
		== "gp"
	)
	
def test_gp_regularization_preserves_aligned_anchor_bin():
	series = TimeSeries(
		times=np.asarray([
			-4.0,
			-1.0,
			0.0,
			2.0,
			5.0,
		]),
		values=np.asarray([
			1.0,
			4.0,
			10.0,
			5.0,
			1.0,
		]),
		errors=np.asarray([
			0.2,
			0.2,
			0.2,
			0.2,
			0.2,
		]),
		metadata={
			"alignment_anchor_time_aligned": 0.0,
		},
	)

	grid = make_regular_grid(
		start=-5.0,
		stop=5.0,
		cadence=1.0,
	)

	output = regularize_timeseries_gp(
		series,
		grid=grid,
		rho=2.0,
	)

	assert output.times[5] == pytest.approx(
		0.0
	)
	

def test_gp_regularization_marks_predictions():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			2.0,
			4.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
			1.0,
		]),
		errors=np.asarray([
			0.1,
			0.1,
			0.1,
		]),
	)

	grid = make_regular_grid(
		start=0.0,
		stop=4.0,
		cadence=1.0,
	)

	output = regularize_timeseries_gp(
		series,
		grid=grid,
		rho=1.0,
	)

	assert output.observed_mask[:, 0].tolist() == [
		False,
		False,
		False,
		False,
		False,
	]

	assert output.interpolated_mask[:, 0].tolist() == [
		False,
		False,
		False,
		False,
		False,
	]

	assert output.predicted_mask[:, 0].tolist() == [
		True,
		True,
		True,
		True,
		True,
	]
	
	
def test_gp_regularization_requires_enough_points():
	series = TimeSeries(
		times=np.asarray([
			0.0,
		]),
		values=np.asarray([
			1.0,
		]),
		errors=np.asarray([
			0.1,
		]),
	)

	grid = np.asarray([
		0.0,
		1.0,
	])

	output = regularize_timeseries_gp(
		series,
		grid=grid,
	)

	assert np.all(
		np.isnan(
			output.values[:, 0]
		)
	)
	
def test_gp_regularization_rejects_invalid_errors():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			1.0,
			2.0,
		]),
		values=np.asarray([
			1.0,
			2.0,
			1.0,
		]),
		errors=np.asarray([
			0.1,
			np.nan,
			0.1,
		]),
	)

	grid = np.asarray([
		0.0,
		1.0,
		2.0,
	])

	with pytest.raises(
		ValueError,
		match="finite positive errors",
	):
		regularize_timeseries_gp(
			series,
			grid=grid,
		)
		

