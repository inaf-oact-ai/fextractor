import numpy as np
import pytest

from fextractor.timeseries.io import (
	read_timeseries,
	read_timeseries_record,
)


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
	
def test_read_wide_timeseries(
	tmp_path,
):
	path = tmp_path / "wide.csv"

	path.write_text(
		"t0,r1,r2,r3,h1,h2,h3,label\n"
		"2026-01-01 00:00:00,"
		"1.0,2.0,3.0,"
		"0,1,1,"
		"X\n",
		encoding="utf-8",
	)

	series = read_timeseries(
		path,
		layout="wide",
		time_column="t0",
		value_prefixes=(
			"r",
			"h",
		),
		channel_names=(
			"flux_ratio",
			"flare_history",
		),
		label_column="label",
	)

	assert series.values.shape == (
		3,
		2,
	)

	assert series.channel_names == (
		"flux_ratio",
		"flare_history",
	)

	np.testing.assert_allclose(
		series.values[:, 0],
		[
			1.0,
			2.0,
			3.0,
		],
	)

	np.testing.assert_allclose(
		series.values[:, 1],
		[
			0.0,
			1.0,
			1.0,
		],
	)

	assert series.metadata[
		"time_origin"
	] == "2026-01-01 00:00:00"

	assert series.metadata[
		"label"
	] == "X"
	
	
def test_wide_channels_must_have_same_length(
	tmp_path,
):
	path = tmp_path / "wide.csv"

	path.write_text(
		"r1,r2,r3,h1,h2\n"
		"1,2,3,0,1\n",
		encoding="utf-8",
	)

	with pytest.raises(
		ValueError,
		match="different lengths",
	):
		read_timeseries(
			path,
			layout="wide",
			value_prefixes=(
				"r",
				"h",
			),
		)
		
		
def test_read_timeseries_record():
	record = {
		"satellite": "goes08",
		"label": "X",
		"n_points": 3,
		"t_start": 1000.0,
		"dt": 60.0,
		"flux": [
			1.0,
			2.0,
			3.0,
		],
		"history": [
			0.0,
			1.0,
			1.0,
		],
	}

	series = read_timeseries_record(
		record,
		value_keys=(
			"flux",
			"history",
		),
		channel_names=(
			"flux_ratio",
			"flare_history",
		),
		time_start_key="t_start",
		cadence_key="dt",
	)

	assert series.values.shape == (
		3,
		2,
	)

	assert series.channel_names == (
		"flux_ratio",
		"flare_history",
	)

	np.testing.assert_allclose(
		series.times,
		[
			1000.0,
			1060.0,
			1120.0,
		],
	)

	assert series.metadata[
		"label"
	] == "X"

	assert series.metadata[
		"satellite"
	] == "goes08"
	
	
def test_read_timeseries_record_rejects_different_lengths():
	record = {
		"a": [
			1.0,
			2.0,
			3.0,
		],
		"b": [
			4.0,
			5.0,
		],
	}

	with pytest.raises(
		ValueError,
		match="different lengths",
	):
		read_timeseries_record(
			record,
			value_keys=(
				"a",
				"b",
			),
		)
		
def test_read_multiband_csv(
	tmp_path,
):
	path = tmp_path / "multiband.csv"

	path.write_text(
		"time,flux,flux_err,band\n"
		"1.0,10.0,0.1,g\n"
		"2.0,11.0,0.2,r\n"
		"3.0,12.0,0.3,i\n",
		encoding="utf-8",
	)

	series = read_timeseries(
		path,
		time_column="time",
		value_columns=(
			"flux",
		),
		error_columns=(
			"flux_err",
		),
		band_column="band",
	)

	assert series.values.shape == (
		3,
		1,
	)

	assert series.errors.shape == (
		3,
		1,
	)

	assert series.bands.tolist() == [
		"g",
		"r",
		"i",
	]

	np.testing.assert_allclose(
		series.times,
		[
			1.0,
			2.0,
			3.0,
		],
	)
	
def test_band_column_is_not_inferred_as_value(
	tmp_path,
):
	path = tmp_path / "multiband.csv"

	path.write_text(
		"time,flux,band\n"
		"1.0,10.0,g\n"
		"2.0,11.0,r\n",
		encoding="utf-8",
	)

	series = read_timeseries(
		path,
		time_column="time",
		band_column="band",
	)

	assert series.values.shape == (
		2,
		1,
	)

	assert series.channel_names == (
		"flux",
	)

	assert series.bands.tolist() == [
		"g",
		"r",
	]
	
def test_read_timeseries_record_with_bands():
	record = {
		"time": [
			59000.0,
			59001.0,
			59002.0,
		],
		"flux": [
			10.0,
			11.0,
			12.0,
		],
		"flux_err": [
			0.1,
			0.2,
			0.3,
		],
		"band": [
			"g",
			"r",
			"i",
		],
	}

	series = read_timeseries_record(
		record,
		value_keys=(
			"flux",
		),
		error_keys=(
			"flux_err",
		),
		time_key="time",
		band_key="band",
	)

	assert series.values.shape == (
		3,
		1,
	)

	assert series.errors.shape == (
		3,
		1,
	)

	assert series.bands.tolist() == [
		"g",
		"r",
		"i",
	]
	
def test_read_timeseries_record_rejects_bad_band_length():
	record = {
		"time": [
			1.0,
			2.0,
			3.0,
		],
		"flux": [
			10.0,
			11.0,
			12.0,
		],
		"band": [
			"g",
			"r",
		],
	}

	with pytest.raises(
		ValueError,
		match="Band field.*length",
	):
		read_timeseries_record(
			record,
			value_keys=(
				"flux",
			),
			time_key="time",
			band_key="band",
		)
		
def test_wide_layout_rejects_band_column(
	tmp_path,
):
	path = tmp_path / "wide.csv"

	path.write_text(
		"r1,r2,r3\n"
		"1,2,3\n",
		encoding="utf-8",
	)

	with pytest.raises(
		ValueError,
		match="band_column.*wide",
	):
		read_timeseries(
			path,
			layout="wide",
			value_prefixes=(
				"r",
			),
			band_column="band",
		)
		
def test_read_npz_timeseries_with_bands(
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
		bands=np.asarray([
			"g",
			"r",
			"i",
		]),
	)

	series = read_timeseries(
		path
	)

	assert series.bands.tolist() == [
		"g",
		"r",
		"i",
	]
	

