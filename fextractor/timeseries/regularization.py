"""Utilities for time-series sampling and regularization."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import (
	Akima1DInterpolator,
	CubicSpline,
	PchipInterpolator,
)

from .data import TimeSeries

SUPPORTED_BIN_AGGREGATIONS = (
	"mean",
	"inverse-variance",
)

SUPPORTED_MISSING_STRATEGIES = (
	"nan",
	"linear",
	"pchip",
	"akima",
	"cubic",
)


def infer_cadence(
	times: np.ndarray,
) -> float:
	"""Infer a representative cadence from increasing timestamps."""

	times = np.asarray(
		times,
		dtype=np.float64,
	).reshape(-1)

	if times.size < 2:
		raise ValueError(
			"At least two timestamps are required to infer cadence"
		)

	delta = np.diff(
		times
	)

	delta = delta[
		np.isfinite(delta)
		& (delta > 0)
	]

	if delta.size == 0:
		raise ValueError(
			"Could not infer cadence from timestamps"
		)

	return float(
		np.median(delta)
	)


def is_regular_timeseries(
	times: np.ndarray,
	rtol: float = 1.0e-3,
	atol: float = 0.0,
) -> bool:
	"""Return whether timestamps are approximately regularly spaced."""

	times = np.asarray(
		times,
		dtype=np.float64,
	).reshape(-1)

	if times.size < 3:
		return True

	delta = np.diff(
		times
	)

	if np.any(
		~np.isfinite(delta)
	):
		return False

	if np.any(
		delta <= 0
	):
		return False

	reference = float(
		np.median(delta)
	)

	return bool(
		np.allclose(
			delta,
			reference,
			rtol=rtol,
			atol=atol,
		)
	)


def _fill_missing(
	grid: np.ndarray,
	values: np.ndarray,
	observed_mask: np.ndarray,
	errors: np.ndarray | None = None,
	method: str = "linear",
) -> tuple[
	np.ndarray,
	np.ndarray | None,
	np.ndarray,
]:
	"""Fill internal gaps using the requested interpolation method.

	Only gaps between the first and last observed samples of each channel
	are filled. No extrapolation is performed.

	For linear interpolation, measurement uncertainties are propagated
	analytically assuming independent errors on the two bracketing
	observations.

	For higher-order interpolation methods, interpolated uncertainties are
	left as NaN.
	"""

	if method not in SUPPORTED_MISSING_STRATEGIES:
		raise ValueError(
			f"Unsupported interpolation method '{method}'. "
			f"Supported values: "
			f"{', '.join(SUPPORTED_MISSING_STRATEGIES)}"
		)

	if method == "nan":
		return (
			values.copy(),
			None if errors is None else errors.copy(),
			np.zeros(
				values.shape,
				dtype=bool,
			),
		)

	grid = np.asarray(
		grid,
		dtype=np.float64,
	).reshape(-1)

	if grid.shape[0] != values.shape[0]:
		raise ValueError(
			"Grid length does not match values: "
			f"{grid.shape[0]} != {values.shape[0]}"
		)

	output = values.copy()

	output_errors = (
		None
		if errors is None
		else errors.copy()
	)

	interpolated_mask = np.zeros(
		values.shape,
		dtype=bool,
	)

	for channel in range(
		values.shape[1]
	):
		valid = (
			observed_mask[:, channel]
			& np.isfinite(
				output[:, channel]
			)
		)

		n_valid = int(
			np.sum(valid)
		)

		if n_valid < 2:
			continue

		valid_indices = np.flatnonzero(
			valid
		)

		first = int(
			valid_indices[0]
		)

		last = int(
			valid_indices[-1]
		)

		fill_region = np.zeros(
			values.shape[0],
			dtype=bool,
		)

		fill_region[
			first:last + 1
		] = (
			~valid[
				first:last + 1
			]
		)

		if not np.any(
			fill_region
		):
			continue

		x_obs = grid[
			valid
		]

		y_obs = output[
			valid,
			channel,
		]

		x_fill = grid[
			fill_region
		]

		if method == "linear":
			y_fill = np.interp(
				x_fill,
				x_obs,
				y_obs,
			)

		elif method == "pchip":
			interpolator = PchipInterpolator(
				x_obs,
				y_obs,
				extrapolate=False,
			)

			y_fill = interpolator(
				x_fill
			)

		elif method == "akima":
			if n_valid < 3:
				continue

			interpolator = Akima1DInterpolator(
				x_obs,
				y_obs,
				extrapolate=False,
			)

			y_fill = interpolator(
				x_fill
			)

		elif method == "cubic":
			if n_valid < 3:
				continue

			interpolator = CubicSpline(
				x_obs,
				y_obs,
				extrapolate=False,
			)

			y_fill = interpolator(
				x_fill
			)

		else:
			raise RuntimeError(
				f"Unhandled interpolation method '{method}'"
			)

		finite_fill = np.isfinite(
			y_fill
		)

		target_indices = np.flatnonzero(
			fill_region
		)

		target_indices = target_indices[
			finite_fill
		]

		output[
			target_indices,
			channel,
		] = y_fill[
			finite_fill
		]

		interpolated_mask[
			target_indices,
			channel,
		] = True

		# - Propagate errors only for linear interpolation
		if (
			method == "linear"
			and output_errors is not None
		):
			for target_index in target_indices:
				left_candidates = valid_indices[
					valid_indices < target_index
				]

				right_candidates = valid_indices[
					valid_indices > target_index
				]

				if (
					left_candidates.size == 0
					or right_candidates.size == 0
				):
					continue

				left = int(
					left_candidates[-1]
				)

				right = int(
					right_candidates[0]
				)

				error_left = output_errors[
					left,
					channel,
				]

				error_right = output_errors[
					right,
					channel,
				]

				if (
					not np.isfinite(error_left)
					or not np.isfinite(error_right)
				):
					continue

				x_left = grid[
					left
				]

				x_right = grid[
					right
				]

				x_target = grid[
					target_index
				]

				alpha = (
					(x_target - x_left)
					/ (x_right - x_left)
				)

				output_errors[
					target_index,
					channel,
				] = np.sqrt(
					(1.0 - alpha) ** 2
					* error_left ** 2
					+ alpha ** 2
					* error_right ** 2
				)

	return (
		output,
		output_errors,
		interpolated_mask,
	)

def make_regular_grid(
	start: float,
	stop: float,
	cadence: float,
) -> np.ndarray:
	"""Construct a regular temporal grid."""

	start = float(
		start
	)

	stop = float(
		stop
	)

	cadence = float(
		cadence
	)

	if not np.isfinite(start):
		raise ValueError(
			f"Grid start must be finite, got {start}"
		)

	if not np.isfinite(stop):
		raise ValueError(
			f"Grid stop must be finite, got {stop}"
		)

	if not np.isfinite(cadence):
		raise ValueError(
			f"Cadence must be finite, got {cadence}"
		)

	if cadence <= 0:
		raise ValueError(
			f"Cadence must be positive, got {cadence}"
		)

	if stop < start:
		raise ValueError(
			f"Grid stop ({stop}) must be greater than "
			f"or equal to grid start ({start})"
		)

	span = (
		stop
		- start
	)

	n_intervals_float = (
		span
		/ cadence
	)

	n_intervals = int(
		np.round(
			n_intervals_float
		)
	)

	if not np.isclose(
		n_intervals_float,
		n_intervals,
		rtol=1.0e-9,
		atol=1.0e-12,
	):
		raise ValueError(
			"Regularization grid span must be an integer "
			"multiple of cadence: "
			f"start={start}, stop={stop}, cadence={cadence}"
		)

	n_grid = (
		n_intervals
		+ 1
	)

	return (
		start
		+ np.arange(
			n_grid,
			dtype=np.float64,
		)
		* cadence
	)

def regularize_timeseries(
	series: TimeSeries,
	cadence: float | None = None,
	missing_strategy: str = "nan",
	grid_start: float | None = None,
	grid_stop: float | None = None,
	bin_aggregation: str = "mean",
) -> TimeSeries:
	"""Project an irregular time series onto a regular temporal grid.

	Multiple observations assigned to the same grid point are aggregated
	according to ``bin_aggregation``.

	Missing internal grid points can optionally be interpolated using
	the configured interpolation strategy. Extrapolation is not performed.
	"""

	if bin_aggregation not in SUPPORTED_BIN_AGGREGATIONS:
		raise ValueError(
			f"Unsupported bin_aggregation '{bin_aggregation}'. "
			f"Supported values: "
			f"{', '.join(SUPPORTED_BIN_AGGREGATIONS)}"
		)

	if series.times is None:
		raise ValueError(
			"Cannot regularize a time series without timestamps"
		)

	if cadence is None:
		cadence = infer_cadence(
			series.times
		)

	if not np.isfinite(
		cadence
	):
		raise ValueError(
			f"Cadence must be finite, got {cadence}"
		)

	if cadence <= 0:
		raise ValueError(
			f"Cadence must be positive, got {cadence}"
		)

	times = np.asarray(
		series.times,
		dtype=np.float64,
	)

	if np.any(
		~np.isfinite(times)
	):
		raise ValueError(
			"Timestamps contain non-finite values"
		)

	# - Resolve regularization bounds
	if grid_start is None:
		start = float(
			times.min()
		)

	else:
		start = float(
			grid_start
		)

	if grid_stop is None:
		stop = float(
			times.max()
		)

	else:
		stop = float(
			grid_stop
		)

	if not np.isfinite(start):
		raise ValueError(
			f"Grid start must be finite, got {start}"
		)

	if not np.isfinite(stop):
		raise ValueError(
			f"Grid stop must be finite, got {stop}"
		)

	if stop < start:
		raise ValueError(
			f"Grid stop ({stop}) must be greater than "
			f"or equal to grid start ({start})"
		)

	# - Construct regular grid
	grid = make_regular_grid(
		start=start,
		stop=stop,
		cadence=cadence,
	)

	n_grid = (
		grid.size
	)

	n_variates = (
		series.n_variates
	)

	# - Common bin accumulators
	sums = np.zeros(
		(
			n_grid,
			n_variates,
		),
		dtype=np.float64,
	)

	counts = np.zeros(
		(
			n_grid,
			n_variates,
		),
		dtype=np.int64,
	)

	# - Error accumulators for arithmetic mean
	error_variance_sums = None
	error_counts = None

	if series.errors is not None:
		error_variance_sums = np.zeros(
			(
				n_grid,
				n_variates,
			),
			dtype=np.float64,
		)

		error_counts = np.zeros(
			(
				n_grid,
				n_variates,
			),
			dtype=np.int64,
		)

	# - Accumulators for inverse-variance aggregation
	weighted_sums = None
	weight_sums = None

	if bin_aggregation == "inverse-variance":
		if series.errors is None:
			raise ValueError(
				"bin_aggregation='inverse-variance' "
				"requires measurement errors"
			)

		weighted_sums = np.zeros(
			(
				n_grid,
				n_variates,
			),
			dtype=np.float64,
		)

		weight_sums = np.zeros(
			(
				n_grid,
				n_variates,
			),
			dtype=np.float64,
		)

	# - Assign samples to nearest grid point
	indices = np.rint(
		(times - start)
		/ cadence
	).astype(
		np.int64
	)

	valid_index = (
		(indices >= 0)
		& (indices < n_grid)
		& (times >= start)
		& (times <= stop)
	)

	for source_index in np.flatnonzero(
		valid_index
	):
		grid_index = indices[
			source_index
		]

		for channel in range(
			n_variates
		):
			if not series.observed_mask[
				source_index,
				channel,
			]:
				continue

			value = series.values[
				source_index,
				channel,
			]

			if not np.isfinite(
				value
			):
				continue

			# - Always count the observed sample
			sums[
				grid_index,
				channel,
			] += value

			counts[
				grid_index,
				channel,
			] += 1

			# - Handle errors / weighted aggregation
			if series.errors is not None:
				error = series.errors[
					source_index,
					channel,
				]

				if bin_aggregation == "mean":
					if np.isfinite(
						error
					):
						error_variance_sums[
							grid_index,
							channel,
						] += (
							error ** 2
						)

						error_counts[
							grid_index,
							channel,
						] += 1

				elif bin_aggregation == "inverse-variance":
					if (
						not np.isfinite(error)
						or error <= 0
					):
						raise ValueError(
							"inverse-variance aggregation requires "
							"finite positive errors for all "
							"observed values"
						)

					weight = (
						1.0
						/ error ** 2
					)

					weighted_sums[
						grid_index,
						channel,
					] += (
						weight
						* value
					)

					weight_sums[
						grid_index,
						channel,
					] += weight

	# - Allocate output arrays
	values = np.full(
		(
			n_grid,
			n_variates,
		),
		np.nan,
		dtype=np.float32,
	)

	observed_mask = (
		counts > 0
	)

	interpolated_mask = np.zeros(
		(
			n_grid,
			n_variates,
		),
		dtype=bool,
	)

	errors = None

	if series.errors is not None:
		errors = np.full(
			(
				n_grid,
				n_variates,
			),
			np.nan,
			dtype=np.float32,
		)

	# - Aggregate values and uncertainties
	if bin_aggregation == "mean":
		values[
			observed_mask
		] = (
			sums[
				observed_mask
			]
			/ counts[
				observed_mask
			]
		).astype(
			np.float32
		)

		if errors is not None:
			valid_errors = (
				(error_counts > 0)
				& (error_counts == counts)
			)

			errors[
				valid_errors
			] = (
				np.sqrt(
					error_variance_sums[
						valid_errors
					]
				)
				/ counts[
					valid_errors
				]
			).astype(
				np.float32
			)

	elif bin_aggregation == "inverse-variance":
		valid_weights = (
			weight_sums > 0
		)

		values[
			valid_weights
		] = (
			weighted_sums[
				valid_weights
			]
			/ weight_sums[
				valid_weights
			]
		).astype(
			np.float32
		)

		errors[
			valid_weights
		] = (
			1.0
			/ np.sqrt(
				weight_sums[
					valid_weights
				]
			)
		).astype(
			np.float32
		)

	# - Handle missing grid cells
	if missing_strategy not in SUPPORTED_MISSING_STRATEGIES:
		raise ValueError(
			f"Unsupported missing_strategy '{missing_strategy}'. "
			f"Supported values: "
			f"{', '.join(SUPPORTED_MISSING_STRATEGIES)}"
		)

	if missing_strategy != "nan":
		(
			values,
			errors,
			interpolated_mask,
		) = _fill_missing(
			grid,
			values,
			observed_mask,
			errors=errors,
			method=missing_strategy,
		)

	# - Fill metadata
	metadata = (
		series.metadata.copy()
	)

	metadata.update({
		"regularized": True,
		"regularization_method": "bin",
		"cadence": cadence,
		"missing_strategy": missing_strategy,
		"bin_aggregation": bin_aggregation,
		"regularization_grid_start": start,
		"regularization_grid_stop": stop,
		"regularization_grid_size": n_grid,
	})
	

	if (
		"alignment_anchor_time_aligned"
		in metadata
	):
		anchor_matches = np.flatnonzero(
			np.isclose(
				grid,
				0.0,
				rtol=1.0e-9,
				atol=1.0e-12,
			)
		)

		if anchor_matches.size != 1:
			raise ValueError(
				"Aligned regularization grid must contain "
				"exactly one t=0 anchor bin"
			)

		metadata[
			"alignment_anchor_index_aligned"
		] = int(
			anchor_matches[0]
		)

	return TimeSeries(
		values=values,
		times=grid,
		observed_mask=observed_mask,
		interpolated_mask=interpolated_mask,
		errors=errors,
		channel_names=series.channel_names,
		metadata=metadata,
	)
	

