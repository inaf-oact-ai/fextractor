"""Time-series data structures and utilities."""

from .data import TimeSeries
from .io import (
	SUPPORTED_TIMESERIES_EXTENSIONS,
	read_timeseries,
)
from .regularization import (
	infer_cadence,
	is_regular_timeseries,
	regularize_timeseries,
)

from .aggregation import (
	SUPPORTED_AGGREGATIONS,
	TokenRepresentation,
	aggregate_token_representation,
)

__all__ = [
	"SUPPORTED_TIMESERIES_EXTENSIONS",
	"TimeSeries",
	"infer_cadence",
	"is_regular_timeseries",
	"read_timeseries",
	"regularize_timeseries",
	"SUPPORTED_AGGREGATIONS",
	"TokenRepresentation",
	"aggregate_token_representation",
]
