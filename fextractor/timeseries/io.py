"""Time-series file readers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
from typing import Any

import numpy as np
from astropy.table import Table

from .data import TimeSeries


SUPPORTED_TIMESERIES_EXTENSIONS = {
	".csv",
	".ecsv",
	".fits",
	".fit",
	".fts",
	".npy",
	".npz",
}


def _column_to_array(
	table: Table,
	name: str,
	dtype,
) -> np.ndarray:
	"""Convert one table column into a NumPy array."""

	if name not in table.colnames:
		raise KeyError(
			f"Column '{name}' not found. Available columns: "
			f"{', '.join(table.colnames)}"
		)

	column = table[name]

	if hasattr(column, "filled"):
		column = column.filled(np.nan)

	return np.asarray(
		column,
		dtype=dtype,
	)


def _read_table(
	path: Path,
) -> Table:
	"""Read a supported tabular time-series file."""

	ext = path.suffix.lower()

	if ext == ".csv":
		return Table.read(
			path,
			format="ascii.csv",
		)

	if ext == ".ecsv":
		return Table.read(
			path,
			format="ascii.ecsv",
		)

	return Table.read(path)


def _select_default_value_columns(
	table: Table,
	time_column: str | None,
	error_columns: Sequence[str] | None,
) -> list[str]:
	"""Infer numeric value columns when none are explicitly supplied."""

	excluded = set()

	if time_column is not None:
		excluded.add(time_column)

	if error_columns is not None:
		excluded.update(error_columns)

	candidates = []

	for name in table.colnames:
		if name in excluded:
			continue

		try:
			array = _column_to_array(
				table,
				name,
				np.float64,
			)
		except (
			TypeError,
			ValueError,
		):
			continue

		if array.ndim == 1:
			candidates.append(name)

	if not candidates:
		raise ValueError(
			"Could not infer a numeric value column. "
			"Specify value_columns explicitly."
		)

	return candidates

def _indexed_columns(
	table: Table,
	prefix: str,
) -> list[str]:
	"""Return prefix+integer columns ordered by their integer suffix."""

	columns = []

	for name in table.colnames:
		if not name.startswith(
			prefix
		):
			continue

		suffix = name[
			len(prefix):
		]

		if not suffix.isdigit():
			continue

		columns.append(
			(
				int(suffix),
				name,
			)
		)

	columns.sort(
		key=lambda item: item[0]
	)

	return [
		name
		for _, name in columns
	]
	
def _read_wide_timeseries(
	table: Table,
	path: Path,
	time_column: str | None = None,
	value_prefixes: Sequence[str] | None = None,
	channel_names: Sequence[str] | None = None,
	label_column: str | None = None,
	metadata_columns: Sequence[str] | None = None,
) -> TimeSeries:
	"""Read one wide/windowed time-series record."""

	if len(table) != 1:
		raise ValueError(
			"Wide time-series layout currently expects exactly "
			f"one row per input file, found {len(table)}"
		)

	if not value_prefixes:
		raise ValueError(
			"value_prefixes must be specified for wide layout"
		)

	value_prefixes = tuple(
		value_prefixes
	)

	if channel_names is None:
		channel_names = value_prefixes

	else:
		channel_names = tuple(
			channel_names
		)

		if len(channel_names) != len(value_prefixes):
			raise ValueError(
				"channel_names must contain one name per "
				"value prefix"
			)

	row = table[0]

	channel_columns = []

	for prefix in value_prefixes:
		columns = _indexed_columns(
			table,
			prefix,
		)

		if not columns:
			raise ValueError(
				f"No wide time-series columns found "
				f"with prefix '{prefix}'"
			)

		channel_columns.append(
			columns
		)

	lengths = {
		len(columns)
		for columns in channel_columns
	}

	if len(lengths) != 1:
		raise ValueError(
			"Wide time-series channels have different lengths: "
			f"{sorted(lengths)}"
		)

	n_time = lengths.pop()

	values = np.empty(
		(
			n_time,
			len(channel_columns),
		),
		dtype=np.float32,
	)

	for channel_index, columns in enumerate(
		channel_columns
	):
		values[
			:,
			channel_index,
		] = np.asarray(
			[
				row[name]
				for name in columns
			],
			dtype=np.float32,
		)

	metadata = {
		"source": str(path),
		"layout": "wide",
		"value_prefixes": list(
			value_prefixes
		),
	}

	if time_column is not None:
		if time_column not in table.colnames:
			raise KeyError(
				f"Column '{time_column}' not found"
			)

		metadata[
			"time_origin"
		] = str(
			row[time_column]
		)

	if label_column is not None:
		if label_column not in table.colnames:
			raise KeyError(
				f"Column '{label_column}' not found"
			)

		metadata[
			"label"
		] = str(
			row[label_column]
		)

	if metadata_columns is not None:
		for name in metadata_columns:
			if name not in table.colnames:
				raise KeyError(
					f"Metadata column '{name}' not found"
				)

			metadata[
				name
			] = str(
				row[name]
			)

	return TimeSeries(
		values=values,
		times=None,
		channel_names=tuple(
			channel_names
		),
		metadata=metadata,
	)	
	
def read_timeseries_record(
	record: dict[str, Any],
	value_keys: Sequence[str],
	channel_names: Sequence[str] | None = None,
	time_key: str | None = None,
	time_start_key: str | None = None,
	cadence_key: str | None = None,
) -> TimeSeries:
	"""Build a TimeSeries from arrays stored directly in a mapping."""

	if not value_keys:
		raise ValueError(
			"value_keys must contain at least one time-series field"
		)

	value_keys = tuple(
		value_keys
	)

	if channel_names is None:
		channel_names = value_keys

	else:
		channel_names = tuple(
			channel_names
		)

		if len(channel_names) != len(value_keys):
			raise ValueError(
				"channel_names must contain one name per value key"
			)

	arrays = []

	for key in value_keys:
		if key not in record:
			raise KeyError(
				f"Time-series field '{key}' not found in record"
			)

		array = np.asarray(
			record[key],
			dtype=np.float32,
		)

		if array.ndim != 1:
			raise ValueError(
				f"Time-series field '{key}' must be one-dimensional, "
				f"got shape {array.shape}"
			)

		arrays.append(
			array
		)

	lengths = {
		len(array)
		for array in arrays
	}

	if len(lengths) != 1:
		raise ValueError(
			"Inline time-series fields have different lengths: "
			f"{sorted(lengths)}"
		)

	n_time = lengths.pop()

	values = np.column_stack(
		arrays
	)

	times = None

	if time_key is not None:
		if time_key not in record:
			raise KeyError(
				f"Time field '{time_key}' not found in record"
			)

		times = np.asarray(
			record[time_key],
			dtype=np.float64,
		)

		if times.ndim != 1:
			raise ValueError(
				f"Time field '{time_key}' must be one-dimensional"
			)

		if len(times) != n_time:
			raise ValueError(
				f"Time field '{time_key}' has length {len(times)}, "
				f"expected {n_time}"
			)

	elif (
		time_start_key is not None
		or cadence_key is not None
	):
		if (
			time_start_key is None
			or cadence_key is None
		):
			raise ValueError(
				"time_start_key and cadence_key must be supplied together"
			)

		if time_start_key not in record:
			raise KeyError(
				f"Time-start field '{time_start_key}' not found in record"
			)

		if cadence_key not in record:
			raise KeyError(
				f"Cadence field '{cadence_key}' not found in record"
			)

		time_start = float(
			record[time_start_key]
		)

		cadence = float(
			record[cadence_key]
		)

		if cadence <= 0:
			raise ValueError(
				f"Cadence must be positive, got {cadence}"
			)

		times = (
			time_start
			+ np.arange(
				n_time,
				dtype=np.float64,
			)
			* cadence
		)

	metadata = {
		"source_type": "inline_record",
	}

	excluded = set(
		value_keys
	)

	if time_key is not None:
		excluded.add(
			time_key
		)

	for key, value in record.items():
		if key in excluded:
			continue

		if isinstance(
			value,
			(
				str,
				int,
				float,
				bool,
				type(None),
			),
		):
			metadata[
				key
			] = value

	return TimeSeries(
		values=values,
		times=times,
		channel_names=tuple(
			channel_names
		),
		metadata=metadata,
	)	
	
def read_timeseries(
	filename: str | Path,
	time_column: str | None = None,
	value_columns: Sequence[str] | None = None,
	error_columns: Sequence[str] | None = None,
	layout: str = "long",
	value_prefixes: Sequence[str] | None = None,
	channel_names: Sequence[str] | None = None,
	label_column: str | None = None,
	metadata_columns: Sequence[str] | None = None,
) -> TimeSeries:
	"""Read one time series from a supported file."""

	path = Path(filename)
	ext = path.suffix.lower()

	if ext == ".npy":
		values = np.load(path)

		return TimeSeries(
			values=values,
			metadata={
				"source": str(path),
			},
		)

	if ext == ".npz":
		payload = np.load(path)

		if "values" not in payload:
			raise KeyError(
				f"NPZ file '{path}' must contain a 'values' array"
			)

		values = payload["values"]

		if "times" in payload:
			times = payload["times"]

		elif "time" in payload:
			times = payload["time"]

		else:
			times = None

		errors = (
			payload["errors"]
			if "errors" in payload
			else None
		)

		return TimeSeries(
			values=values,
			times=times,
			errors=errors,
			metadata={
				"source": str(path),
			},
		)

	if ext not in SUPPORTED_TIMESERIES_EXTENSIONS:
		raise ValueError(
			f"Unsupported time-series extension '{ext}' "
			f"for '{path}'"
		)

	table = _read_table(path)
	
	if layout == "wide":
		return _read_wide_timeseries(
			table=table,
			path=path,
			time_column=time_column,
			value_prefixes=value_prefixes,
			channel_names=channel_names,
			label_column=label_column,
			metadata_columns=metadata_columns,
		)

	if layout != "long":
		raise ValueError(
			f"Unsupported time-series layout '{layout}'. "
			"Supported values: long, wide"
		)
		

	if value_columns is None:
		value_columns = _select_default_value_columns(
			table,
			time_column=time_column,
			error_columns=error_columns,
		)

	value_columns = list(
		value_columns
	)

	values = np.column_stack([
		_column_to_array(
			table,
			name,
			np.float32,
		)
		for name in value_columns
	])

	times = None

	if time_column is not None:
		times = _column_to_array(
			table,
			time_column,
			np.float64,
		)

	errors = None

	if error_columns is not None:
		error_columns = list(
			error_columns
		)

		if len(error_columns) != len(value_columns):
			raise ValueError(
				"error_columns must contain one column "
				"per value column"
			)

		errors = np.column_stack([
			_column_to_array(
				table,
				name,
				np.float32,
			)
			for name in error_columns
		])

	return TimeSeries(
		values=values,
		times=times,
		errors=errors,
		channel_names=tuple(
			value_columns
		),
		metadata={
			"source": str(path),
			"time_column": time_column,
			"value_columns": value_columns,
			"error_columns": error_columns,
		},
	)
