from __future__ import annotations

import numpy as np

from .data import TimeSeries


def regularize_timeseries_gp(
	series: TimeSeries,
	grid: np.ndarray,
	sigma: float | None = None,
	rho: float | None = None,
	jitter: float | None = None,
) -> TimeSeries:
	"""Fit an independent GP to each channel and predict on a target grid."""

	try:
		from celerite2 import GaussianProcess
		from celerite2 import terms

	except ImportError as exc:
		raise ImportError(
			"GP regularization requires the optional "
			"'celerite2' dependency"
		) from exc

	if series.times is None:
		raise ValueError(
			"Cannot apply GP regularization without timestamps"
		)

	grid = np.asarray(
		grid,
		dtype=np.float64,
	).reshape(-1)

	times = np.asarray(
		series.times,
		dtype=np.float64,
	)

	n_grid = grid.size
	n_variates = series.n_variates

	values = np.full(
		(
			n_grid,
			n_variates,
		),
		np.nan,
		dtype=np.float32,
	)

	errors = np.full(
		(
			n_grid,
			n_variates,
		),
		np.nan,
		dtype=np.float32,
	)

	predicted_mask = np.zeros(
		(
			n_grid,
			n_variates,
		),
		dtype=bool,
	)

	for channel in range(
		n_variates
	):
		valid = (
			series.observed_mask[:, channel]
			& np.isfinite(
				series.values[:, channel]
			)
		)

		if np.sum(valid) < 2:
			continue

		t_obs = times[
			valid
		]
		
		if np.any(
			np.diff(t_obs) <= 0
		):
			raise ValueError(
				"GP regularization requires strictly increasing "
				"observation times"
			)

		y_obs = series.values[
			valid,
			channel,
		].astype(
			np.float64
		)

		channel_sigma = sigma

		if channel_sigma is None:
			channel_sigma = float(
				np.std(
					y_obs
				)
			)

		if (
			not np.isfinite(channel_sigma)
			or channel_sigma <= 0
		):
			channel_sigma = 1.0

		channel_rho = rho

		if channel_rho is None:
			delta = np.diff(
				np.sort(
					t_obs
				)
			)

			delta = delta[
				np.isfinite(delta)
				& (delta > 0)
			]

			if delta.size == 0:
				continue

			channel_rho = float(
				np.median(
					delta
				)
			)

		if (
			not np.isfinite(channel_rho)
			or channel_rho <= 0
		):
			raise ValueError(
				f"Invalid GP rho={channel_rho}"
			)

		kernel = terms.Matern32Term(
			sigma=channel_sigma,
			rho=channel_rho,
		)

		mean = float(
			np.median(
				y_obs
			)
		)

		gp = GaussianProcess(
			kernel,
			mean=mean,
		)

		yerr = None

		if series.errors is not None:
			yerr_candidate = series.errors[
				valid,
				channel,
			].astype(
				np.float64
			)

			if np.all(
				np.isfinite(
					yerr_candidate
				)
				& (yerr_candidate > 0)
			):
				yerr = yerr_candidate

			elif np.any(
				np.isfinite(
					yerr_candidate
				)
			):
				raise ValueError(
					"GP regularization requires either "
					"all finite positive errors or no errors"
				)

		if yerr is None:
			if jitter is None:
				channel_jitter = 1.0e-6

			else:
				channel_jitter = float(
					jitter
				)

			if (
				not np.isfinite(
					channel_jitter
				)
				or channel_jitter <= 0
			):
				raise ValueError(
					"GP jitter must be finite and positive"
				)

			yerr = np.full(
				y_obs.shape,
				channel_jitter,
				dtype=np.float64,
			)

		gp.compute(
			t_obs,
			yerr=yerr,
		)

		mu, var = gp.predict(
			y_obs,
			t=grid,
			return_var=True,
		)

		values[
			:,
			channel,
		] = mu.astype(
			np.float32
		)

		errors[
			:,
			channel,
		] = np.sqrt(
			np.maximum(
				var,
				0.0,
			)
		).astype(
			np.float32
		)

		predicted_mask[
			:,
			channel,
		] = np.isfinite(
			values[
				:,
				channel,
			]
		)

	# - Fill metadata
	metadata = (
		series.metadata.copy()
	)

	metadata.update({
		"regularized": True,
		"regularization_method": "gp",
		"regularization_grid_start": float(grid[0]),
		"regularization_grid_stop": float(grid[-1]),
		"regularization_grid_size": int(grid.size),
		"gp_kernel": "matern32",
		"gp_sigma": sigma,
		"gp_rho": rho,
		"gp_jitter": jitter,
	})
	

	# - Preserve aligned anchor metadata
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
				"Aligned GP grid must contain "
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
		observed_mask=np.zeros(
			values.shape,
			dtype=bool,
		),
		interpolated_mask=np.zeros(
			values.shape,
			dtype=bool,
		),
		predicted_mask=predicted_mask,
		errors=errors,
		channel_names=series.channel_names,
		metadata=metadata,
	)
