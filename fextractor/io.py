"""Datalist input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_DATALIST_KEY = "data"

IMAGE_EXTENSIONS = {
	".fits",
	".fit",
	".fts",
	".png",
	".jpg",
	".jpeg",
	".tif",
	".tiff",
}


def detect_input_type(filename: str | Path) -> str:
	"""Detect whether an input file is an image or a JSON datalist."""
	path = Path(filename)

	name = path.name.lower()

	if name.endswith(".json"):
		return "datalist"

	if name.endswith(".fits.gz"):
		return "image"

	if path.suffix.lower() in IMAGE_EXTENSIONS:
		return "image"

	raise ValueError(
		f"Unsupported input file type '{filename}'. "
		"Expected a JSON datalist or supported image file."
	)


def read_datalist(filename: str | Path, key: str = DEFAULT_DATALIST_KEY) -> list[dict[str, Any]]:
	"""Read the standard JSON datalist format.

	If the top-level object contains ``key``, that list is returned. A bare
	JSON list is also accepted.
	"""
	with Path(filename).open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if isinstance(payload, dict):
		if key not in payload:
			raise KeyError(f"Datalist key '{key}' not found in '{filename}'")
		payload = payload[key]

	if not isinstance(payload, list):
		raise TypeError("Datalist JSON must contain a list of entries")

	return payload


def save_datalist_json(
	datalist: list[dict[str, Any]],
	filename: str | Path,
	key: str = DEFAULT_DATALIST_KEY,
	metadata: dict[str, Any] | None = None,
) -> None:
	"""Save the standard wrapped JSON datalist format."""
	payload: dict[str, Any] = {key: datalist}
	if metadata is not None:
		payload["fextractor"] = metadata

	with Path(filename).open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, allow_nan=False)
		handle.write("\n")


def save_feature_vector(features, filename: str | Path, metadata: dict[str, Any] | None = None) -> None:
	"""Save one feature vector as JSON."""
	payload = {
		"feats": [float(value) for value in features],
	}
	if metadata is not None:
		payload["fextractor"] = metadata

	with Path(filename).open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, allow_nan=False)
		handle.write("\n")
