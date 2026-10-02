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
	band_column: str | None = None,
) -> list[str]:
	"""Infer numeric value columns when none are explicitly supplied."""

	excluded = set()

	if time_column is not None:
		excluded.add(time_column)

	if error_columns is not None:
		excluded.update(error_columns)
		
	if band_column is not None:
		excluded.add(
			band_column
		)

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
	

####################################
##   TIME-SERIES - WIDE FORMAT
####################################
# - Example CSV for irregularly-sampled time series
# t0,t1,t2,..., var1_0,var1_1,..., var2_0,var2_1,..., var1_err_0,var1_err_1,...,var2_err_0,var2_err_1,...,

# - Example CSV for regularly-sampled time series
# t0,cadence, var1_0,var1_1,..., var2_0,var2_1,..., var1_err_0,var1_err_1,...,var2_err_0,var2_err_1,...,


def _read_wide_timeseries(
	table: Table,
	path: Path,
	time_column: str | None = None,
	time_prefix: str | None = None,
	time_start_column: str | None = None,
	cadence_column: str | None = None,
	value_prefixes: Sequence[str] | None = None,
	error_prefixes: Sequence[str] | None = None,
	channel_names: Sequence[str] | None = None,
	label_column: str | None = None,
	metadata_columns: Sequence[str] | None = None,
) -> TimeSeries:

	"""Read one wide/windowed time-series record."""

	# - Validate options
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
	
	if error_prefixes is not None:
		error_prefixes = tuple(
			error_prefixes
		)

		if len(error_prefixes) != len(value_prefixes):
			raise ValueError(
				"error_prefixes must contain one prefix "
				"per value prefix"
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

	# - Fill channel columns
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

	# - Fill error channel columns
	error_channel_columns = None

	if error_prefixes is not None:
		error_channel_columns = []

		for prefix in error_prefixes:
			columns = _indexed_columns(
				table,
				prefix,
			)

			if not columns:
				raise ValueError(
					f"No wide error columns found "
					f"with prefix '{prefix}'"
				)

			error_channel_columns.append(
				columns
			)

	# - Validate length
	lengths = {
		len(columns)
		for columns in channel_columns
	}

	if len(lengths) != 1:
		raise ValueError(
			"Wide time-series channels have different lengths: "
			f"{sorted(lengths)}"
		)
	
	# - Fill time columns
	n_time = lengths.pop()
	times = None

	if time_prefix is not None:
		time_columns = _indexed_columns(
			table,
			time_prefix,
		)

		if not time_columns:
			raise ValueError(
				f"No wide time columns found "
				f"with prefix '{time_prefix}'"
			)

		if len(time_columns) != n_time:
			raise ValueError(
				"Wide time-column length does not match "
				f"value-channel length: "
				f"{len(time_columns)} != {n_time}"
			)

		times = np.asarray(
			[
				row[name]
				for name in time_columns
			],
			dtype=np.float64,
		)

	elif (
		time_start_column is not None
		or cadence_column is not None
	):
		if (
			time_start_column is None
			or cadence_column is None
		):
			raise ValueError(
				"time_start_column and cadence_column "
				"must be supplied together"
			)

		if time_start_column not in table.colnames:
			raise KeyError(
				f"Column '{time_start_column}' not found"
			)

		if cadence_column not in table.colnames:
			raise KeyError(
				f"Column '{cadence_column}' not found"
			)

		time_start = float(
			row[time_start_column]
		)

		cadence = float(
			row[cadence_column]
		)

		if not np.isfinite(
			time_start
		):
			raise ValueError(
				"Wide time start must be finite"
			)

		if (
			not np.isfinite(cadence)
			or cadence <= 0
		):
			raise ValueError(
				f"Wide cadence must be positive and finite, "
				f"got {cadence}"
			)

		times = (
			time_start
			+ np.arange(
				n_time,
				dtype=np.float64,
			)
			* cadence
		)

	# - Validate error length	
	if error_channel_columns is not None:
		error_lengths = {
			len(columns)
			for columns in error_channel_columns
		}

		if len(error_lengths) != 1:
			raise ValueError(
				"Wide error channels have different lengths: "
				f"{sorted(error_lengths)}"
			)

		error_length = error_lengths.pop()

		if error_length != n_time:
			raise ValueError(
				"Wide error-channel length does not match "
				f"value-channel length: {error_length} != {n_time}"
			)	

	# - Fill values
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


	# - Fill errors
	errors = None
	
	if error_channel_columns is not None:
		errors = np.empty(
			(
				n_time,
				len(error_channel_columns),
			),
			dtype=np.float32,
		)

		for channel_index, columns in enumerate(
			error_channel_columns
		):
			errors[
				:,
				channel_index,
			] = np.asarray(
				[
					row[name]
					for name in columns
				],
				dtype=np.float32,
			)

	# - Fill metadata
	metadata = {
		"source": str(path),
		"layout": "wide",
		"value_prefixes": list(
			value_prefixes
		),
	}

	# Legacy scalar time metadata. This does not define the
	# per-sample time coordinate; use time_prefix or
	# time_start_column + cadence_column for that.
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
		times=times,
		errors=errors,
		channel_names=tuple(
			channel_names
		),
		metadata=metadata,
	)
	
def read_timeseries_record(
	record: dict[str, Any],
	value_keys: Sequence[str],
	error_keys: Sequence[str] | None = None,
	channel_names: Sequence[str] | None = None,
	time_key: str | None = None,
	time_start_key: str | None = None,
	cadence_key: str | None = None,
	band_key: str | None = None,
) -> TimeSeries:
	"""Build a TimeSeries from arrays stored directly in a mapping."""

	# - Check value keys
	if not value_keys:
		raise ValueError(
			"value_keys must contain at least one time-series field"
		)

	value_keys = tuple(
		value_keys
	)
	
	# - Check error keys
	if error_keys is not None:
		error_keys = tuple(
			error_keys
		)

		if len(error_keys) != len(value_keys):
			raise ValueError(
				"error_keys must contain one field "
				"per value field"
			)
	
	# - Check channel names
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

	# - Fill values
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
	
	# - Fill errors
	errors = None

	if error_keys is not None:
		error_arrays = []

		for key in error_keys:
			if key not in record:
				raise KeyError(
					f"Time-series error field '{key}' "
					"not found in record"
				)

			array = np.asarray(
				record[key],
				dtype=np.float32,
			)

			if array.ndim != 1:
				raise ValueError(
					f"Time-series error field '{key}' "
					"must be one-dimensional, "
					f"got shape {array.shape}"
				)

			if len(array) != n_time:
				raise ValueError(
					f"Time-series error field '{key}' "
					f"has length {len(array)}, "
					f"expected {n_time}"
				)

			error_arrays.append(
				array
			)

		errors = np.column_stack(
			error_arrays
		)

	# - Fill times
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

	# - Fill bands
	bands = None

	if band_key is not None:
		if band_key not in record:
			raise KeyError(
				f"Band field '{band_key}' not found in record"
			)

		bands = np.asarray(
			record[
				band_key
			]
		)

		if bands.ndim != 1:
			raise ValueError(
				f"Band field '{band_key}' must be one-dimensional"
			)

		if len(bands) != n_time:
			raise ValueError(
				f"Band field '{band_key}' has length {len(bands)}, "
				f"expected {n_time}"
			)
		
			
	metadata = {
		"source_type": "inline_record",
	}

	excluded = set(
		value_keys
	)

	if error_keys is not None:
		excluded.update(
			error_keys
		)

	if time_key is not None:
		excluded.add(
			time_key
		)

	if time_start_key is not None:
		excluded.add(
			time_start_key
		)

	if cadence_key is not None:
		excluded.add(
			cadence_key
		)

	if band_key is not None:
		excluded.add(
			band_key
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
		errors=errors,
		bands=bands,
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
	band_column: str | None = None,
	layout: str = "long",
	value_prefixes: Sequence[str] | None = None,
	error_prefixes: Sequence[str] | None = None,
	time_prefix: str | None = None,
	time_start_column: str | None = None,
	cadence_column: str | None = None,
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

		if "bands" in payload:
			bands = payload[
				"bands"
			]

		elif "band" in payload:
			bands = payload[
				"band"
			]

		else:
			bands = None

		return TimeSeries(
			values=values,
			times=times,
			errors=errors,
			bands=bands,
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
		if band_column is not None:
			raise ValueError(
				"band_column is not supported with wide time-series layout. "
				"LiCu multiband models require long-layout observation data."
			)

		return _read_wide_timeseries(
			table=table,
			path=path,
			time_column=time_column,
			time_prefix=time_prefix,
			time_start_column=time_start_column,
			cadence_column=cadence_column,
			value_prefixes=value_prefixes,
			error_prefixes=error_prefixes,
			channel_names=channel_names,
			label_column=label_column,
			metadata_columns=metadata_columns,
		)

	if layout != "long":
		raise ValueError(
			f"Unsupported time-series layout '{layout}'. "
			"Supported values: long, wide"
		)
		
	# - Fill value_columns
	if value_columns is None:
		value_columns = _select_default_value_columns(
			table,
			time_column=time_column,
			error_columns=error_columns,
			band_column=band_column,
		)

	value_columns = list(
		value_columns
	)

	# - Fill values
	values = np.column_stack([
		_column_to_array(
			table,
			name,
			np.float32,
		)
		for name in value_columns
	])

	# - Fill times
	times = None

	if time_column is not None:
		times = _column_to_array(
			table,
			time_column,
			np.float64,
		)

	# - Fill errors
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


	# - Fill bands
	bands = None

	if band_column is not None:
		if band_column not in table.colnames:
			raise KeyError(
				f"Band column '{band_column}' not found. "
				f"Available columns: {', '.join(table.colnames)}"
			)

		bands = np.asarray(
			table[
				band_column
			]
		)

		if bands.ndim != 1:
			raise ValueError(
				f"Band column '{band_column}' must be one-dimensional"
			)

		if len(bands) != values.shape[0]:
			raise ValueError(
				f"Band column '{band_column}' has length {len(bands)}, "
				f"expected {values.shape[0]}"
			)

	return TimeSeries(
		values=values,
		times=times,
		errors=errors,
		bands=bands,
		channel_names=tuple(
			value_columns
		),
		metadata={
			"source": str(path),
			"time_column": time_column,
			"value_columns": value_columns,
			"error_columns": error_columns,
			"band_column": band_column,
		},
	)
