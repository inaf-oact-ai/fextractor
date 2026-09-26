"""Time-series file readers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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


def read_timeseries(
	filename: str | Path,
	time_column: str | None = None,
	value_columns: Sequence[str] | None = None,
	error_columns: Sequence[str] | None = None,
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
