"""Utilities for time-series sampling and regularization."""

from __future__ import annotations

import numpy as np

from .data import TimeSeries


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


def _linear_fill(
	values: np.ndarray,
	observed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Fill internal gaps by linear interpolation."""

	output = values.copy()
	mask = observed_mask.copy()

	x = np.arange(
		values.shape[0],
		dtype=np.float64,
	)

	for channel in range(
		values.shape[1]
	):
		valid = (
			mask[:, channel]
			& np.isfinite(
				output[:, channel]
			)
		)

		if valid.sum() < 2:
			continue

		valid_indices = np.flatnonzero(
			valid
		)

		first = valid_indices[0]
		last = valid_indices[-1]

		fill_region = (
			(~valid)
			& (x >= first)
			& (x <= last)
		)

		if not np.any(
			fill_region
		):
			continue

		output[
			fill_region,
			channel,
		] = np.interp(
			x[fill_region],
			x[valid],
			output[
				valid,
				channel,
			],
		)

		mask[
			fill_region,
			channel,
		] = True

	return (
		output,
		mask,
	)


def regularize_timeseries(
	series: TimeSeries,
	cadence: float | None = None,
	missing_strategy: str = "nan",
) -> TimeSeries:
	"""Project an irregular time series onto a regular temporal grid.

	Multiple observations assigned to the same grid point are averaged.
	Missing grid points can either remain NaN or be linearly interpolated.
	"""

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

	start = float(
		times.min()
	)

	stop = float(
		times.max()
	)

	n_grid = (
		int(
			np.round(
				(stop - start)
				/ cadence
			)
		)
		+ 1
	)

	grid = (
		start
		+ np.arange(
			n_grid,
			dtype=np.float64,
		)
		* cadence
	)

	n_variates = (
		series.n_variates
	)

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

	error_sums = None
	error_counts = None

	if series.errors is not None:
		error_sums = np.zeros(
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

	indices = np.rint(
		(times - start)
		/ cadence
	).astype(
		np.int64
	)

	valid_index = (
		(indices >= 0)
		& (indices < n_grid)
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

			sums[
				grid_index,
				channel,
			] += value

			counts[
				grid_index,
				channel,
			] += 1

			if series.errors is not None:
				error = series.errors[
					source_index,
					channel,
				]

				if np.isfinite(
					error
				):
					error_sums[
						grid_index,
						channel,
					] += error

					error_counts[
						grid_index,
						channel,
					] += 1

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

		valid_errors = (
			error_counts > 0
		)

		errors[
			valid_errors
		] = (
			error_sums[
				valid_errors
			]
			/ error_counts[
				valid_errors
			]
		).astype(
			np.float32
		)

	if missing_strategy == "nan":
		pass

	elif missing_strategy == "linear":
		(
			values,
			observed_mask,
		) = _linear_fill(
			values,
			observed_mask,
		)

	else:
		raise ValueError(
			"Unsupported missing_strategy "
			f"'{missing_strategy}'. "
			"Supported values: nan, linear"
		)

	metadata = (
		series.metadata.copy()
	)

	metadata.update({
		"regularized": True,
		"cadence": cadence,
		"missing_strategy": missing_strategy,
	})

	return TimeSeries(
		values=values,
		times=grid,
		observed_mask=observed_mask,
		errors=errors,
		channel_names=series.channel_names,
		metadata=metadata,
	)
