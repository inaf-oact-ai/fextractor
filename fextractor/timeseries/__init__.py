"""Time-series data structures and utilities."""

from .data import TimeSeries
from .io import (
	SUPPORTED_TIMESERIES_EXTENSIONS,
	read_timeseries,
)

from .regularization import (
	infer_cadence,
	is_regular_timeseries,
	make_regular_grid,
	regularize_timeseries,
)

from .gp import (
	regularize_timeseries_gp,
)

from .aggregation import (
	SUPPORTED_AGGREGATIONS,
	TokenRepresentation,
	aggregate_token_representation,
)

from .plotting import (
	SUPPORTED_TIMESERIES_PLOT_MODES,
	make_index_coordinate,
	plot_timeseries_diagnostic,
	recover_physical_times,
)

__all__ = [
	"SUPPORTED_TIMESERIES_EXTENSIONS",
	"TimeSeries",
	"infer_cadence",
	"is_regular_timeseries",
	"read_timeseries",
	"make_regular_grid",
	"regularize_timeseries",
	"regularize_timeseries_gp",
	"SUPPORTED_AGGREGATIONS",
	"TokenRepresentation",
	"aggregate_token_representation",
	"SUPPORTED_TIMESERIES_PLOT_MODES",
	"make_index_coordinate",
	"plot_timeseries_diagnostic",
	"recover_physical_times",
]


