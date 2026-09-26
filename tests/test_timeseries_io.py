import numpy as np
import pytest

from fextractor.timeseries import read_timeseries


def test_read_npy_timeseries(
	tmp_path,
):
	path = tmp_path / "series.npy"

	np.save(
		path,
		np.asarray([
			1.0,
			2.0,
			3.0,
		]),
	)

	series = read_timeseries(
		path
	)

	assert series.values.shape == (
		3,
		1,
	)

	assert series.times is None


def test_read_npz_timeseries(
	tmp_path,
):
	path = tmp_path / "series.npz"

	np.savez(
		path,
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
		errors=np.asarray([
			0.1,
			0.2,
			0.3,
		]),
	)

	series = read_timeseries(
		path
	)

	assert series.values.shape == (
		3,
		1,
	)

	assert series.times.shape == (
		3,
	)

	assert series.errors.shape == (
		3,
		1,
	)


def test_read_csv_timeseries(
	tmp_path,
):
	path = tmp_path / "series.csv"

	path.write_text(
		"time,flux,flux_err\n"
		"1.0,10.0,0.1\n"
		"2.0,11.0,0.2\n"
		"3.0,12.0,0.3\n",
		encoding="utf-8",
	)

	series = read_timeseries(
		path,
		time_column="time",
		value_columns=("flux",),
		error_columns=("flux_err",),
	)

	assert series.values.shape == (
		3,
		1,
	)

	assert series.channel_names == (
		"flux",
	)

	np.testing.assert_allclose(
		series.times,
		[
			1.0,
			2.0,
			3.0,
		],
	)

	np.testing.assert_allclose(
		series.values[:, 0],
		[
			10.0,
			11.0,
			12.0,
		],
	)


def test_read_multivariate_csv(
	tmp_path,
):
	path = tmp_path / "series.csv"

	path.write_text(
		"time,g,r\n"
		"1.0,10.0,20.0\n"
		"2.0,11.0,21.0\n",
		encoding="utf-8",
	)

	series = read_timeseries(
		path,
		time_column="time",
		value_columns=(
			"g",
			"r",
		),
	)

	assert series.values.shape == (
		2,
		2,
	)

	assert series.channel_names == (
		"g",
		"r",
	)


def test_error_columns_must_match_values(
	tmp_path,
):
	path = tmp_path / "series.csv"

	path.write_text(
		"time,g,r,g_err\n"
		"1.0,10.0,20.0,0.1\n",
		encoding="utf-8",
	)

	with pytest.raises(
		ValueError,
		match="error_columns",
	):
		read_timeseries(
			path,
			time_column="time",
			value_columns=(
				"g",
				"r",
			),
			error_columns=(
				"g_err",
			),
		)
