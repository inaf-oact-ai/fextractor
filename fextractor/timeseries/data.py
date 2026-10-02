"""Canonical time-series data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimeSeries:
	"""Canonical representation of one univariate or multivariate time series.

	Values are represented internally as ``[time, variate]``.
	"""

	values: np.ndarray
	times: np.ndarray | None = None
	observed_mask: np.ndarray | None = None
	interpolated_mask: np.ndarray | None = None
	predicted_mask: np.ndarray | None = None
	errors: np.ndarray | None = None
	channel_names: tuple[str, ...] | None = None
	bands: np.ndarray | None = None
	metadata: dict = field(default_factory=dict)

	def __post_init__(self) -> None:
	
		# - Validate values
		values = np.asarray(
			self.values,
			dtype=np.float32,
		)

		if values.ndim == 1:
			values = values[:, None]

		if values.ndim != 2:
			raise ValueError(
				"Time-series values must be 1D or 2D, "
				f"got shape {values.shape}"
			)

		self.values = values

		n_time, n_variates = values.shape

		# - Validate times
		if self.times is not None:
			times = np.asarray(
				self.times,
				dtype=np.float64,
			).reshape(-1)

			if times.shape[0] != n_time:
				raise ValueError(
					"Time coordinate length does not match values: "
					f"{times.shape[0]} != {n_time}"
				)

			self.times = times

		# - Validate bands
		if self.bands is not None:
			bands = np.asarray(
				self.bands
			).reshape(-1)

			if bands.shape[0] != n_time:
				raise ValueError(
					"bands length does not match values: "
					f"{bands.shape[0]} != {n_time}"
				)

			self.bands = bands		

		# - Handle observed mask
		if self.observed_mask is None:
			self.observed_mask = np.isfinite(
				values
			)

		else:
			observed_mask = np.asarray(
				self.observed_mask,
				dtype=bool,
			)

			if observed_mask.ndim == 1:
				observed_mask = (
					observed_mask[:, None]
				)

			if observed_mask.shape != values.shape:
				raise ValueError(
					"observed_mask shape does not match values: "
					f"{observed_mask.shape} != {values.shape}"
				)

			self.observed_mask = (
				observed_mask
				& np.isfinite(values)
			)

		# - Handle interpolated mask
		if self.interpolated_mask is None:
			self.interpolated_mask = np.zeros(
				values.shape,
				dtype=bool,
			)

		else:
			interpolated_mask = np.asarray(
				self.interpolated_mask,
				dtype=bool,
			)

			if interpolated_mask.ndim == 1:
				interpolated_mask = (
					interpolated_mask[:, None]
				)

			if interpolated_mask.shape != values.shape:
				raise ValueError(
					"interpolated_mask shape does not match values: "
					f"{interpolated_mask.shape} != {values.shape}"
				)

			self.interpolated_mask = (
				interpolated_mask
				& np.isfinite(values)
			)

		# - Handle predicted mask
		if self.predicted_mask is None:
			self.predicted_mask = np.zeros(
				values.shape,
				dtype=bool,
			)

		else:
			predicted_mask = np.asarray(
				self.predicted_mask,
				dtype=bool,
			)

			if predicted_mask.ndim == 1:
				predicted_mask = (
					predicted_mask[:, None]
				)

			if predicted_mask.shape != values.shape:
				raise ValueError(
					"predicted_mask shape does not match values: "
					f"{predicted_mask.shape} != {values.shape}"
				)

			self.predicted_mask = (
				predicted_mask
				& np.isfinite(values)
			)

		# - Validate masks
		if np.any(
			self.observed_mask
			& self.interpolated_mask
		):
			raise ValueError(
				"A sample cannot be both observed and interpolated"
			)

		if np.any(
			self.observed_mask
			& self.predicted_mask
		):
			raise ValueError(
				"A sample cannot be both observed and predicted"
			)

		if np.any(
			self.interpolated_mask
			& self.predicted_mask
		):
			raise ValueError(
				"A sample cannot be both interpolated and predicted"
			)


		# - Handle errors
		if self.errors is not None:
			errors = np.asarray(
				self.errors,
				dtype=np.float32,
			)

			if errors.ndim == 1:
				errors = errors[:, None]

			if errors.shape != values.shape:
				raise ValueError(
					"errors shape does not match values: "
					f"{errors.shape} != {values.shape}"
				)

			self.errors = errors

		if self.channel_names is not None:
			if len(self.channel_names) != n_variates:
				raise ValueError(
					"channel_names length does not match "
					"number of variates: "
					f"{len(self.channel_names)} != {n_variates}"
				)

	@property
	def n_time(self) -> int:
		"""Return the number of time samples."""
		return self.values.shape[0]

	@property
	def n_variates(self) -> int:
		"""Return the number of variates/channels."""
		return self.values.shape[1]

	@property
	def is_univariate(self) -> bool:
		"""Return whether the series contains one variate."""
		return self.n_variates == 1

	def copy(self) -> "TimeSeries":
		"""Return a deep copy of the numerical data."""

		return TimeSeries(
			values=self.values.copy(),
			times=(
				None
				if self.times is None
				else self.times.copy()
			),
			observed_mask=self.observed_mask.copy(),
			interpolated_mask=self.interpolated_mask.copy(),
			predicted_mask=self.predicted_mask.copy(),
			errors=(
				None
				if self.errors is None
				else self.errors.copy()
			),
			channel_names=self.channel_names,
			bands=(
				None
				if self.bands is None
				else self.bands.copy()
			),
			metadata=self.metadata.copy(),
		)
