"""Datalist input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_DATALIST_KEY = "data"


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
