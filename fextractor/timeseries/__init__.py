"""Time-series data structures and utilities."""

from .data import TimeSeries
from .io import (
	SUPPORTED_TIMESERIES_EXTENSIONS,
	read_timeseries,
)

__all__ = [
	"SUPPORTED_TIMESERIES_EXTENSIONS",
	"TimeSeries",
	"read_timeseries",
]
