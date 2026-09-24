"""High-level extraction runners."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .base import FeatureExtractor


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

	for index, item in enumerate(output):
		if nmax >= 0 and index >= nmax:
			break

		try:
			filepaths = item["filepaths"]
			filename = Path(filepaths[filepath_index])
			features = extractor.extract(filename)
			item[feature_key] = [float(value) for value in features]
		except Exception as exc:
			if skip_errors:
				item["fextractor_error"] = str(exc)
				continue
			raise ExtractionError(f"Failed to process datalist entry {index}: {exc}") from exc

	return output
