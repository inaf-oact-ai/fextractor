"""High-level extraction runners."""

from __future__ import annotations

import logging
import time

from copy import deepcopy
from pathlib import Path
from typing import Any

from .base import FeatureExtractor


logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
	"""Error raised when a datalist entry cannot be processed."""


def extract_datalist(
	datalist: list[dict[str, Any]],
	extractor: FeatureExtractor,
	nmax: int = -1,
	filepath_index: int = 0,
	feature_key: str = "feats",
	copy_data: bool = False,
	skip_errors: bool = False,
) -> list[dict[str, Any]]:
	"""Extract features for the standard ``filepaths`` datalist format."""
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
			filepaths = item["filepaths"]
			filename = Path(filepaths[filepath_index])

			logger.debug(
				"Processing entry=%d file='%s'",
				index,
				filename,
			)

			features = extractor.extract(filename)
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
