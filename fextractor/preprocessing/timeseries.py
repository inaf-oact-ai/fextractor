"""Domain-level time-series preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..timeseries import (
	TimeSeries,
	make_regular_grid,
	regularize_timeseries,
	regularize_timeseries_gp,
)

import logging

logger = logging.getLogger(__name__)

SUPPORTED_TIME_TRANSFORMS = (
	"none",
	"origin",
)

SUPPORTED_VALUE_TRANSFORMS = (
	"none",
	"maxabs",
	"minmax",
	"standard",
	"asinh",
)

SUPPORTED_ALIGNMENTS = (
	"none",
	"peak-max",
	"peak-min",
	"peak-abs",
)

SUPPORTED_REGULARIZATION_METHODS = (
	"bin",
	"gp",
)

@dataclass(frozen=True)
class TimeSeriesPreprocessConfig:
	"""Domain-level time-series preprocessing options."""

	layout: str = "long"
	
	time_column: str | None = None
	value_columns: tuple[str, ...] | None = None
	error_columns: tuple[str, ...] | None = None
	band_column: str | None = None

	value_prefixes: tuple[str, ...] | None = None
	channel_names: tuple[str, ...] | None = None
	
	error_prefixes: tuple[str, ...] | None = None
	time_prefix: str | None = None
	time_start_column: str | None = None
	cadence_column: str | None = None
	
	label_column: str | None = None
	metadata_columns: tuple[str, ...] | None = None
	
	sort_time: bool = True
	
	time_transform: str = "none"
	value_transform: str = "none"
	value_transform_scale: float | None = None
	
	alignment: str = "none"
	alignment_window_before: float | None = None
	alignment_window_after: float | None = None
	
	regularize: bool = False
	regularization_method: str = "bin"

	gp_sigma: float | None = None
	gp_rho: float | None = None
	gp_jitter: float | None = None
	
	cadence: float | None = None
	missing_strategy: str = "nan"
	bin_aggregation: str = "mean"
	
	time_start_key: str | None = None
	cadence_key: str | None = None
	band_key: str | None = None

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



def _transform_time(
	series: TimeSeries,
	mode: str,
) -> TimeSeries:
	"""Apply a transform to the time coordinate."""

	if mode not in SUPPORTED_TIME_TRANSFORMS:
		raise ValueError(
			f"Unsupported time_transform '{mode}'. "
			f"Supported values: {', '.join(SUPPORTED_TIME_TRANSFORMS)}"
		)

	output = series.copy()

	if mode == "none":
		return output

	if output.times is None:
		raise ValueError(
			f"Cannot apply time_transform='{mode}' "
			"to a time series without timestamps"
		)

	if output.times.size == 0:
		raise ValueError(
			"Cannot transform an empty time coordinate"
		)

	if mode == "origin":
		reference = float(
			output.times[0]
		)

		output.times = (
			output.times
			- reference
		)

		output.metadata.update({
			"time_transform": mode,
			"time_transform_reference": reference,
		})

	return output

def _transform_values(
	series: TimeSeries,
	mode: str,
	scale: float | None = None,
) -> TimeSeries:
	"""Apply a channel-wise transform to time-series values."""

	if mode not in SUPPORTED_VALUE_TRANSFORMS:
		raise ValueError(
			f"Unsupported value_transform '{mode}'. "
			f"Supported values: {', '.join(SUPPORTED_VALUE_TRANSFORMS)}"
		)

	output = series.copy()

	if mode == "none":
		return output

	values = output.values.copy()

	for channel in range(
		output.n_variates
	):
		valid = (
			output.observed_mask[:, channel]
			& np.isfinite(
				values[:, channel]
			)
		)

		if not np.any(valid):
			continue

		x = values[
			valid,
			channel,
		]

		# =====================
		# ==   MAXABS
		# =====================
		if mode == "maxabs":
			denominator = float(
				np.max(
					np.abs(x)
				)
			)

			if denominator > 0:
				values[
					valid,
					channel,
				] = (
					x
					/ denominator
				)

				if output.errors is not None:
					output.errors[
						valid,
						channel,
					] = (
						output.errors[
							valid,
							channel,
						]
						/ denominator
					)		
				
		# ============================
		# ==   MINMAX
		# ============================
		elif mode == "minmax":
			x_min = float(
				np.min(x)
			)

			x_max = float(
				np.max(x)
			)

			denominator = (
				x_max
				- x_min
			)

			if denominator > 0:
				values[
					valid,
					channel,
				] = (
					(x - x_min)
					/ denominator
				)

				if output.errors is not None:
					output.errors[
						valid,
						channel,
					] = (
						output.errors[
							valid,
							channel,
						]
						/ denominator
					)

		# ============================
		# ==   STANDARDIZATION
		# ============================
		elif mode == "standard":
			mean = float(
				np.mean(x)
			)

			std = float(
				np.std(x)
			)

			if std > 0:
				values[
					valid,
					channel,
				] = (
					(x - mean)
					/ std
				)

				if output.errors is not None:
					output.errors[
						valid,
						channel,
					] = (
						output.errors[
							valid,
							channel,
						]
						/ std
					)

		# ============================
		# ==   ASINH
		# ============================
		elif mode == "asinh":
			transform_scale = (
				1.0
				if scale is None
				else float(scale)
			)

			if (
				not np.isfinite(
					transform_scale
				)
				or transform_scale <= 0
			):
				raise ValueError(
					"value_transform_scale must be "
					"finite and positive for asinh"
				)

			values[
				valid,
				channel,
			] = np.arcsinh(
				x
				/ transform_scale
			)

			if output.errors is not None:
				output.errors[
					valid,
					channel,
				] = (
					output.errors[
						valid,
						channel,
					]
					/ np.sqrt(
						x ** 2
						+ transform_scale ** 2
					)
				)

	output.values = values

	output.metadata.update({
		"value_transform": mode,
	})

	if mode == "asinh":
		output.metadata[
			"value_transform_scale"
		] = (
			1.0
			if scale is None
			else float(scale)
		)

	return output


def _find_alignment_anchor(
	series: TimeSeries,
	mode: str,
) -> tuple[int, float | None]:
	"""Return the time-index and physical time of an alignment anchor."""

	if mode not in SUPPORTED_ALIGNMENTS:
		raise ValueError(
			f"Unsupported alignment '{mode}'. "
			f"Supported values: {', '.join(SUPPORTED_ALIGNMENTS)}"
		)

	if mode == "none":
		raise ValueError(
			"Alignment anchor is undefined for alignment='none'"
		)

	valid = (
		series.observed_mask
		& np.isfinite(series.values)
	)

	if not np.any(valid):
		raise ValueError(
			"Cannot determine alignment anchor: "
			"no finite observed values"
		)

	values = np.where(
		valid,
		series.values,
		np.nan,
	)

	if mode == "peak-max":
		score = np.nanmax(
			values,
			axis=1,
		)

		anchor_index = int(
			np.nanargmax(score)
		)

	elif mode == "peak-min":
		score = np.nanmin(
			values,
			axis=1,
		)

		anchor_index = int(
			np.nanargmin(score)
		)

	elif mode == "peak-abs":
		score = np.nanmax(
			np.abs(values),
			axis=1,
		)

		anchor_index = int(
			np.nanargmax(score)
		)

	else:
		raise RuntimeError(
			f"Unexpected alignment mode '{mode}'"
		)

	anchor_time = None

	if series.times is not None:
		anchor_time = float(
			series.times[anchor_index]
		)

	return (
		anchor_index,
		anchor_time,
	)
	

def _detect_alignment(
	series: TimeSeries,
	mode: str,
) -> TimeSeries:
	"""Detect and record the alignment anchor."""

	output = series.copy()

	if mode == "none":
		return output

	(
		anchor_index,
		anchor_time,
	) = _find_alignment_anchor(
		output,
		mode,
	)

	output.metadata.update({
		"alignment": mode,
		"alignment_anchor_index_original": anchor_index,
		"alignment_anchor_time_original": anchor_time,
	})

	return output

def _apply_alignment(
	series: TimeSeries,
	mode: str,
	window_before: float | None = None,
	window_after: float | None = None,
) -> TimeSeries:
	"""Shift a time series to its detected anchor and optionally crop it."""

	output = series.copy()

	if mode == "none":
		return output

	if output.times is None:
		raise ValueError(
			f"Cannot apply alignment='{mode}' "
			"to a time series without timestamps"
		)

	if (
		"alignment_anchor_index_original"
		not in output.metadata
	):
		raise ValueError(
			"Alignment anchor metadata is missing"
		)

	anchor_index = int(
		output.metadata[
			"alignment_anchor_index_original"
		]
	)

	if (
		anchor_index < 0
		or anchor_index >= output.n_time
	):
		raise ValueError(
			f"Invalid alignment anchor index {anchor_index}"
		)

	anchor_coordinate = float(
		output.times[anchor_index]
	)

	output.times = (
		output.times
		- anchor_coordinate
	)

	if window_before is not None:
		window_before = float(
			window_before
		)

		if (
			not np.isfinite(window_before)
			or window_before < 0
		):
			raise ValueError(
				"alignment_window_before must be "
				"finite and non-negative"
			)

	if window_after is not None:
		window_after = float(
			window_after
		)

		if (
			not np.isfinite(window_after)
			or window_after < 0
		):
			raise ValueError(
				"alignment_window_after must be "
				"finite and non-negative"
			)

	n_left_truncated = 0
	n_right_truncated = 0

	if window_before is not None:
		n_left_truncated = int(
			np.sum(
				output.times < -window_before
			)
		)

	if window_after is not None:
		n_right_truncated = int(
			np.sum(
				output.times > window_after
			)
		)

	if n_left_truncated > 0:
		logger.warning(
			"Alignment window truncated %d sample(s) before the anchor",
			n_left_truncated,
		)

	if n_right_truncated > 0:
		logger.warning(
			"Alignment window truncated %d sample(s) after the anchor",
			n_right_truncated,
		)

	keep = np.ones(
		output.n_time,
		dtype=bool,
	)

	if window_before is not None:
		keep &= (
			output.times
			>= -window_before
		)

	if window_after is not None:
		keep &= (
			output.times
			<= window_after
		)

	output.times = output.times[
		keep
	]

	output.values = output.values[
		keep
	]

	output.observed_mask = (
		output.observed_mask[
			keep
		]
	)
	
	output.interpolated_mask = (
		output.interpolated_mask[
			keep
		]
	)
	
	output.predicted_mask = (
		output.predicted_mask[
			keep
		]
	)

	if output.errors is not None:
		output.errors = (
			output.errors[
				keep
			]
		)

	if output.bands is not None:
		output.bands = (
			output.bands[
				keep
			]
		)

	output.metadata.update({
		"alignment_anchor_time_aligned": 0.0,
		"alignment_window_before": window_before,
		"alignment_window_after": window_after,
		"alignment_truncated_left": n_left_truncated,
		"alignment_truncated_right": n_right_truncated,
	})

	return output
	
	
def apply_timeseries_preprocessing(
	series: TimeSeries,
	config: TimeSeriesPreprocessConfig,
) -> TimeSeries:
	"""Apply domain-level preprocessing to one time series."""

	output = series.copy()

	logger.info(
		"Starting time-series preprocessing: "
		"n_time=%d n_variates=%d "
		"time_transform='%s' value_transform='%s' "
		"alignment='%s' regularize=%s "
		"regularization_method='%s'",
		output.n_time,
		output.n_variates,
		config.time_transform,
		config.value_transform,
		config.alignment,
		config.regularize,
		config.regularization_method,
	)	

	# - Sort entries
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
		
		output.interpolated_mask = (
			output.interpolated_mask[
				order
			]
		)
		
		output.predicted_mask = (
			output.predicted_mask[
				order
			]
		)

		if output.errors is not None:
			output.errors = (
				output.errors[
					order
				]
			)

		if output.bands is not None:
			output.bands = (
				output.bands[
					order
				]
			)

	# - Detect alignment anchor
	output = _detect_alignment(
		output,
		mode=config.alignment,
	)

	# - Transform time
	output = _transform_time(
		output,
		mode=config.time_transform,
	)

	# - Transform values
	output = _transform_values(
		output,
		mode=config.value_transform,
		scale=config.value_transform_scale,
	)

	# - Apply physical-time alignment/windowing
	output = _apply_alignment(
		output,
		mode=config.alignment,
		window_before=config.alignment_window_before,
		window_after=config.alignment_window_after,
	)

	# - Regularize
	if (
		config.regularize
		and config.regularization_method
		not in SUPPORTED_REGULARIZATION_METHODS
	):
		raise ValueError(
			f"Unsupported regularization_method "
			f"'{config.regularization_method}'. "
			f"Supported values: "
			f"{', '.join(SUPPORTED_REGULARIZATION_METHODS)}"
		)
		
	if (
		config.regularize
		and config.alignment != "none"
		and config.cadence is not None
	):
		if (
			config.alignment_window_before is not None
			and not np.isclose(
				config.alignment_window_before
				/ config.cadence,
				round(
					config.alignment_window_before
					/ config.cadence
				),
			)
		):
			raise ValueError(
				"alignment_window_before must be an integer "
				"multiple of cadence so the alignment anchor "
				"falls on an exact regular-grid bin"
			)

		if (
			config.alignment_window_after is not None
			and not np.isclose(
				config.alignment_window_after
				/ config.cadence,
				round(
					config.alignment_window_after
					/ config.cadence
				),
			)
		):
			raise ValueError(
				"alignment_window_after must be an integer "
				"multiple of cadence so the alignment anchor "
				"falls on an exact regular-grid bin"
			)
			
	if not config.regularize:
		logger.info(
			"Time-series regularization disabled"
		)			
			
	# - Finally regularize
	if config.regularize:
		grid_start = None
		grid_stop = None

		if (
			config.alignment != "none"
			and config.alignment_window_before is not None
		):
			grid_start = (
				-config.alignment_window_before
			)

		if (
			config.alignment != "none"
			and config.alignment_window_after is not None
		):
			grid_stop = (
				config.alignment_window_after
			)

		# - Bin regularization mode
		if config.regularization_method == "bin":
			logger.info(
				"Applying bin regularization: "
				"cadence=%s missing_strategy='%s' "
				"bin_aggregation='%s' grid_start=%s grid_stop=%s",
				config.cadence,
				config.missing_strategy,
				config.bin_aggregation,
				grid_start,
				grid_stop,
			)
					
			output = regularize_timeseries(
				output,
				cadence=config.cadence,
				missing_strategy=config.missing_strategy,
				grid_start=grid_start,
				grid_stop=grid_stop,
				bin_aggregation=config.bin_aggregation,
			)

		# - Gaussian Process regularization mode
		elif config.regularization_method == "gp":
			if config.cadence is None:
				raise ValueError(
					"GP regularization currently requires "
					"an explicit cadence"
				)

			if grid_start is None:
				grid_start = float(
					output.times.min()
				)

			if grid_stop is None:
				grid_stop = float(
					output.times.max()
				)

			grid = make_regular_grid(
				start=grid_start,
				stop=grid_stop,
				cadence=config.cadence,
			)

			logger.info(
				"Applying GP regularization: "
				"cadence=%s grid_start=%s grid_stop=%s "
				"grid_size=%d sigma=%s rho=%s jitter=%s",
				config.cadence,
				grid_start,
				grid_stop,
				grid.size,
				config.gp_sigma,
				config.gp_rho,
				config.gp_jitter,
			)
			
			output = regularize_timeseries_gp(
				output,
				grid=grid,
				sigma=config.gp_sigma,
				rho=config.gp_rho,
				jitter=config.gp_jitter,
			)
	
	
	logger.info(
		"Time-series preprocessing completed: "
		"n_time=%d n_variates=%d "
		"observed=%d interpolated=%d predicted=%d",
		output.n_time,
		output.n_variates,
		int(np.sum(output.observed_mask)),
		int(np.sum(output.interpolated_mask)),
		int(np.sum(output.predicted_mask)),
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
