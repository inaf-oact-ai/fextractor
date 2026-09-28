"""High-level extraction runners."""

from __future__ import annotations

import logging
import time

from copy import deepcopy
from pathlib import Path
from typing import Any

from .base import FeatureExtractor
from .extractors.timeseries.base import TimeSeriesFeatureExtractor
from .timeseries.io import read_timeseries_record

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
	"""Error raised when a datalist entry cannot be processed."""

def _get_entry_source(
	item: dict[str, Any],
	extractor: FeatureExtractor,
	filepath_index: int = 0,
):
	"""Resolve the extraction source for one datalist entry."""

	if (
		"filepath" in item
		or "filepaths" in item
	):
		return _get_entry_filepath(
			item,
			filepath_index=filepath_index,
		)

	if isinstance(
		extractor,
		TimeSeriesFeatureExtractor,
	):
		preprocessing = extractor.preprocessing

		if preprocessing.value_columns is None:
			raise ValueError(
				"Inline time-series records require value_columns"
			)

		return read_timeseries_record(
			record=item,
			value_keys=preprocessing.value_columns,
			error_keys=preprocessing.error_columns,
			channel_names=preprocessing.channel_names,
			time_key=preprocessing.time_column,
			time_start_key=preprocessing.time_start_key,
			cadence_key=preprocessing.cadence_key,
		)

	raise KeyError(
		"Datalist entry must contain 'filepath' or 'filepaths'"
	)
	
def _get_entry_filepath(
	item: dict[str, Any],
	filepath_index: int = 0,
) -> Path:
	"""Resolve a source filepath from a datalist entry."""

	if "filepath" in item:
		return Path(
			item["filepath"]
		)

	if "filepaths" in item:
		filepaths = item[
			"filepaths"
		]

		if not isinstance(
			filepaths,
			(
				list,
				tuple,
			),
		):
			raise TypeError(
				"'filepaths' must be a list or tuple"
			)

		if not filepaths:
			raise ValueError(
				"'filepaths' cannot be empty"
			)

		try:
			return Path(
				filepaths[
					filepath_index
				]
			)

		except IndexError as exc:
			raise IndexError(
				f"filepath_index={filepath_index} is out of range "
				f"for {len(filepaths)} filepaths"
			) from exc

	raise KeyError(
		"Datalist entry must contain either "
		"'filepath' or 'filepaths'"
	)

def extract_datalist(
	datalist: list[dict[str, Any]],
	extractor: FeatureExtractor,
	nmax: int = -1,
	filepath_index: int = 0,
	feature_key: str = "feats",
	copy_data: bool = False,
	skip_errors: bool = False,
) -> list[dict[str, Any]]:
	"""
		Extract features from filepath-based datalist entries.
		Entries may contain either ``filepath`` for a single source or
		``filepaths`` for multiple associated sources.
	"""

	output = deepcopy(datalist) if copy_data else datalist

	total_entries = len(output)

	if nmax >= 0:
		entries_to_process = min(nmax, total_entries)
	else:
		entries_to_process = total_entries

	logger.info(
		"Starting datalist extraction: entries=%d backend='%s' feature_key='%s'",
		entries_to_process,
		extractor.backend,
		feature_key,
	)

	start_time = time.perf_counter()
	processed = 0
	skipped = 0

	for index, item in enumerate(output):
		if nmax >= 0 and index >= nmax:
			break

		try:
			logger.debug("Processing datalist entry=%d", index)
			
			source = _get_entry_source(
				item,
				extractor,
				filepath_index=filepath_index,
			)

			features = extractor.extract(source)

			item[feature_key] = [float(value) for value in features]

			processed += 1

			logger.debug(
				"Completed entry=%d features=%d",
				index,
				len(features),
			)

		except Exception as exc:
			if skip_errors:
				skipped += 1
				item["fextractor_error"] = str(exc)

				logger.warning(
					"Skipping datalist entry=%d error='%s'",
					index,
					exc,
				)

				continue

			raise ExtractionError(
				f"Failed to process datalist entry {index}: {exc}"
			) from exc

	elapsed = time.perf_counter() - start_time

	logger.info(
		"Datalist extraction completed: processed=%d skipped=%d elapsed=%.3fs",
		processed,
		skipped,
		elapsed,
	)

	return output
