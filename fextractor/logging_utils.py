"""Logging helpers for fextractor."""

from __future__ import annotations

import logging
import sys


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
	"""Configure application-wide logging."""
	log_level = getattr(logging, level.upper(), None)

	if not isinstance(log_level, int):
		raise ValueError(f"Invalid log level '{level}'")

	logging.basicConfig(
		level=log_level,
		format=DEFAULT_LOG_FORMAT,
		datefmt=DEFAULT_DATE_FORMAT,
		stream=sys.stderr,
		force=True,
	)
