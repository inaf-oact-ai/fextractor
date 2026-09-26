"""Domain-level time-series preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..timeseries import (
	TimeSeries,
	regularize_timeseries,
)


@dataclass(frozen=True)
class TimeSeriesPreprocessConfig:
	"""Domain-level time-series preprocessing options."""

	time_column: str | None = None
	value_columns: tuple[str, ...] | None = None
	error_columns: tuple[str, ...] | None = None

	sort_time: bool = True

	regularize: bool = False
	cadence: float | None = None
	missing_strategy: str = "nan"


@dataclass(frozen=True)
class TimeSeriesPreprocessProfile:
	"""Named time-series preprocessing profile."""

	name: str
	preprocessing: TimeSeriesPreprocessConfig


_PROFILES = {
	"default": TimeSeriesPreprocessProfile(
		name="default",
		preprocessing=TimeSeriesPreprocessConfig(),
	),

	"lightcurve": TimeSeriesPreprocessProfile(
		name="lightcurve",
		preprocessing=TimeSeriesPreprocessConfig(
			time_column="time",
			value_columns=("flux",),
			error_columns=None,
			sort_time=True,
			regularize=False,
			cadence=None,
			missing_strategy="nan",
		),
	),
}


def list_profiles() -> tuple[str, ...]:
	"""Return available time-series preprocessing profiles."""

	return tuple(
		sorted(_PROFILES)
	)


def get_profile(
	name: str,
) -> TimeSeriesPreprocessProfile:
	"""Return a named time-series preprocessing profile."""

	try:
		return _PROFILES[name]

	except KeyError as exc:
		raise KeyError(
			f"Unknown time-series preprocessing profile '{name}'. "
			f"Available: {', '.join(list_profiles())}"
		) from exc


def apply_timeseries_preprocessing(
	series: TimeSeries,
	config: TimeSeriesPreprocessConfig,
) -> TimeSeries:
	"""Apply domain-level preprocessing to one time series."""

	output = series.copy()

	if (
		config.sort_time
		and output.times is not None
	):
		order = np.argsort(
			output.times
		)

		output.times = output.times[
			order
		]

		output.values = output.values[
			order
		]

		output.observed_mask = (
			output.observed_mask[
				order
			]
		)

		if output.errors is not None:
			output.errors = (
				output.errors[
					order
				]
			)

	if config.regularize:
		output = regularize_timeseries(
			output,
			cadence=config.cadence,
			missing_strategy=config.missing_strategy,
		)

	return output


def override_config(
	config: TimeSeriesPreprocessConfig,
	**changes,
) -> TimeSeriesPreprocessConfig:
	"""Return a frozen preprocessing configuration with overrides."""

	return replace(
		config,
		**changes,
	)
