from pathlib import Path

import numpy as np
import pytest

from fextractor.timeseries import TimeSeries
from fextractor.timeseries.plotting import (
	make_index_coordinate,
	plot_timeseries_diagnostic,
	recover_physical_times,
)


def test_recover_physical_times_from_origin_transform():
	series = TimeSeries(
		times=np.asarray([
			0.0,
			10.0,
			20.0,
		]),
		values=np.asarray([
			1.0,
			2.0,
			3.0,
		]),
		metadata={
			"time_transform": "origin",
			"time_transform_reference": 59000.0,
		},
	)

	np.testing.assert_allclose(
		recover_physical_times(series),
		[
			59000.0,
			59010.0,
			59020.0,
		],
	)


def test_recover_physical_times_from_alignment():
	series = TimeSeries(
		times=np.asarray([
			-10.0,
			0.0,
			10.0,
		]),
		values=np.asarray([
			1.0,
			3.0,
			2.0,
		]),
		metadata={
			"alignment_anchor_time_original": 59020.0,
			"alignment_anchor_time_aligned": 0.0,
			"alignment_anchor_index_aligned": 1,
		},
	)

	np.testing.assert_allclose(
		recover_physical_times(series),
		[
			59010.0,
			59020.0,
			59030.0,
		],
	)

	index, label = make_index_coordinate(
		series
	)

	np.testing.assert_allclose(
		index,
		[
			-1.0,
			0.0,
			1.0,
		],
	)
	assert label == "Bin index relative to anchor"


def test_plot_timeseries_diagnostic(tmp_path: Path):
	pytest.importorskip(
		"matplotlib"
	)

	input_series = TimeSeries(
		times=np.asarray([
			59000.0,
			59010.0,
			59020.0,
			59030.0,
		]),
		values=np.asarray([
			[1.0, 0.0],
			[2.0, 1.0],
			[np.nan, np.nan],
			[4.0, 0.0],
		]),
		channel_names=(
			"flux",
			"history",
		),
	)

	processed_series = TimeSeries(
		times=np.asarray([
			59000.0,
			59010.0,
			59020.0,
			59030.0,
		]),
		values=np.asarray([
			[1.0, 0.0],
			[2.0, 1.0],
			[3.0, 0.5],
			[4.0, 0.0],
		]),
		observed_mask=np.asarray([
			[True, True],
			[True, True],
			[False, False],
			[True, True],
		]),
		interpolated_mask=np.asarray([
			[False, False],
			[False, False],
			[True, True],
			[False, False],
		]),
		channel_names=(
			"flux",
			"history",
		),
	)

	output_path = tmp_path / "diagnostic.png"

	result = plot_timeseries_diagnostic(
		input_series,
		processed_series,
		output_path,
		mode="both",
	)

	assert result == output_path
	assert output_path.exists()
	assert output_path.stat().st_size > 0
