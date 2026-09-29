"""Diagnostic plotting utilities for time-series preprocessing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .data import TimeSeries

logger = logging.getLogger(__name__)

SUPPORTED_TIMESERIES_PLOT_MODES = (
	"none",
	"input",
	"processed",
	"both",
)


def recover_physical_times(
	series: TimeSeries,
) -> np.ndarray | None:
	"""Recover the original physical time coordinate when possible.

	The returned coordinate is intended for diagnostics and plotting. The
	preprocessed ``series.times`` array remains the coordinate actually used by
	the downstream representation model.
	"""

	if series.times is None:
		return None

	times = np.asarray(
		series.times,
		dtype=np.float64,
	).reshape(-1)

	metadata = series.metadata

	if (
		"alignment_anchor_time_original"
		in metadata
	):
		anchor_time = metadata[
			"alignment_anchor_time_original"
		]

		if anchor_time is not None:
			return (
				times
				+ float(anchor_time)
			)

	if (
		metadata.get("time_transform") == "origin"
		and "time_transform_reference" in metadata
	):
		return (
			times
			+ float(
				metadata[
					"time_transform_reference"
				]
			)
		)

	return times.copy()


def make_index_coordinate(
	series: TimeSeries,
) -> tuple[np.ndarray, str]:
	"""Return the diagnostic sample/bin coordinate for the lower x-axis."""

	indices = np.arange(
		series.n_time,
		dtype=np.float64,
	)

	anchor_index = series.metadata.get(
		"alignment_anchor_index_aligned"
	)

	if anchor_index is not None:
		indices = (
			indices
			- float(anchor_index)
		)

		return (
			indices,
			"Bin index relative to anchor",
		)

	return (
		indices,
		"Sample index",
	)


def _load_pyplot():
	"""Import matplotlib lazily using a headless-safe backend."""

	try:
		import matplotlib

		matplotlib.use(
			"Agg",
			force=True,
		)

		import matplotlib.pyplot as plt

	except ImportError as exc:
		raise ImportError(
			"Time-series plotting requires the optional "
			"'matplotlib' dependency. Install fextractor[plot]."
		) from exc

	return plt


def _channel_name(
	series: TimeSeries,
	channel: int,
) -> str:
	if series.channel_names is None:
		return f"channel_{channel}"

	return str(
		series.channel_names[channel]
	)


def _format_time_value(
	value: float,
) -> str:
	if not np.isfinite(value):
		return ""

	abs_value = abs(value)

	if abs_value >= 1.0e7:
		return f"{value:.0f}"

	if abs_value >= 1.0e4:
		return f"{value:.3f}".rstrip("0").rstrip(".")

	return f"{value:.6g}"


def _tick_indices(
	n_time: int,
	max_ticks: int = 7,
) -> np.ndarray:
	if n_time <= 0:
		return np.asarray(
			[],
			dtype=np.int64,
		)

	if n_time <= max_ticks:
		return np.arange(
			n_time,
			dtype=np.int64,
		)

	return np.unique(
		np.rint(
			np.linspace(
				0,
				n_time - 1,
				max_ticks,
			)
		).astype(
			np.int64
		)
	)


def _add_dual_time_axis(
	ax,
	series: TimeSeries,
	x_coord: np.ndarray,
	x_label: str,
) -> None:
	"""Add lower index and upper physical-time axes to one subplot."""

	physical_times = recover_physical_times(
		series
	)

	ax.set_xlabel(
		x_label
	)

	if physical_times is None:
		return

	tick_indices = _tick_indices(
		series.n_time
	)

	if tick_indices.size == 0:
		return

	tick_positions = x_coord[
		tick_indices
	]

	ax.set_xticks(
		tick_positions
	)

	upper = ax.twiny()
	upper.set_xlim(
		ax.get_xlim()
	)
	upper.set_xticks(
		tick_positions
	)
	upper.set_xticklabels([
		_format_time_value(
			float(
				physical_times[index]
			)
		)
		for index in tick_indices
	])
	upper.set_xlabel(
		"Physical time"
	)


def _plot_input_channel(
	ax,
	series: TimeSeries,
	channel: int,
	x_coord: np.ndarray,
) -> None:
	values = series.values[
		:,
		channel,
	]

	valid = (
		series.observed_mask[
			:,
			channel,
		]
		& np.isfinite(values)
	)

	if series.errors is not None:
		errors = series.errors[
			:,
			channel,
		]

		valid_errors = (
			valid
			& np.isfinite(errors)
			& (errors >= 0)
		)

		if np.any(valid_errors):
			ax.errorbar(
				x_coord[valid_errors],
				values[valid_errors],
				yerr=errors[valid_errors],
				fmt="o",
				markersize=3,
				linestyle="none",
				capsize=2,
				label="Observed",
			)

		without_errors = (
			valid
			& ~valid_errors
		)

		if np.any(without_errors):
			ax.plot(
				x_coord[without_errors],
				values[without_errors],
				"o",
				markersize=3,
				linestyle="none",
				label="Observed",
			)

	else:
		ax.plot(
			x_coord[valid],
			values[valid],
			"o",
			markersize=3,
			linestyle="none",
			label="Observed",
		)


def _plot_processed_channel(
	ax,
	series: TimeSeries,
	channel: int,
	x_coord: np.ndarray,
) -> None:
	values = series.values[
		:,
		channel,
	]

	finite = np.isfinite(
		values
	)

	predicted = (
		series.predicted_mask[
			:,
			channel,
		]
		& finite
	)

	observed = (
		series.observed_mask[
			:,
			channel,
		]
		& finite
	)

	interpolated = (
		series.interpolated_mask[
			:,
			channel,
		]
		& finite
	)

	if np.any(predicted):
		ax.plot(
			x_coord[predicted],
			values[predicted],
			linewidth=1.5,
			label="GP prediction",
		)

		if series.errors is not None:
			errors = series.errors[
				:,
				channel,
			]

			valid_errors = (
				predicted
				& np.isfinite(errors)
				& (errors >= 0)
			)

			if np.any(valid_errors):
				ax.fill_between(
					x_coord[valid_errors],
					values[valid_errors]
					- errors[valid_errors],
					values[valid_errors]
					+ errors[valid_errors],
					alpha=0.2,
					label="GP +/- 1 sigma",
				)

	else:
		if np.any(finite):
			ax.plot(
				x_coord[finite],
				values[finite],
				linewidth=1.0,
				label="Processed",
			)

	if np.any(observed):
		ax.plot(
			x_coord[observed],
			values[observed],
			"o",
			markersize=3,
			linestyle="none",
			label="Observed bins",
		)

	if np.any(interpolated):
		ax.plot(
			x_coord[interpolated],
			values[interpolated],
			"x",
			markersize=4,
			linestyle="none",
			label="Interpolated",
		)


def _overlay_input_on_processed(
	ax,
	input_series: TimeSeries,
	processed_series: TimeSeries,
	channel: int,
	processed_x: np.ndarray,
) -> None:
	"""Overlay original observations on a GP panel when scales are compatible."""

	if processed_series.metadata.get(
		"value_transform",
		"none",
	) != "none":
		return

	if not np.any(
		processed_series.predicted_mask[
			:,
			channel,
		]
	):
		return

	input_physical = recover_physical_times(
		input_series
	)
	processed_physical = recover_physical_times(
		processed_series
	)

	if (
		input_physical is None
		or processed_physical is None
		or processed_physical.size < 2
	):
		return

	if np.any(
		np.diff(processed_physical) <= 0
	):
		return

	values = input_series.values[
		:,
		channel,
	]

	valid = (
		input_series.observed_mask[
			:,
			channel,
		]
		& np.isfinite(values)
		& np.isfinite(input_physical)
		& (input_physical >= processed_physical[0])
		& (input_physical <= processed_physical[-1])
	)

	if not np.any(valid):
		return

	x_overlay = np.interp(
		input_physical[valid],
		processed_physical,
		processed_x,
	)

	ax.plot(
		x_overlay,
		values[valid],
		"o",
		markersize=2.5,
		linestyle="none",
		label="Input observations",
	)


def plot_timeseries_diagnostic(
	input_series: TimeSeries,
	processed_series: TimeSeries,
	output_path: str | Path,
	mode: str = "both",
	title: str | None = None,
) -> Path | None:
	"""Save a multi-channel time-series preprocessing diagnostic plot.

	Rows correspond to channels. Columns correspond to input and/or processed
	series according to ``mode``. Every subplot uses an index coordinate on the
	lower x-axis and the reconstructed physical time on the upper x-axis.
	"""

	if mode not in SUPPORTED_TIMESERIES_PLOT_MODES:
		raise ValueError(
			f"Unsupported time-series plot mode '{mode}'. "
			f"Supported values: {', '.join(SUPPORTED_TIMESERIES_PLOT_MODES)}"
		)

	if mode == "none":
		return None

	if input_series.n_variates != processed_series.n_variates:
		raise ValueError(
			"Input and processed time series have different "
			"numbers of channels"
		)

	plt = _load_pyplot()

	panels: list[tuple[str, TimeSeries]] = []

	if mode in (
		"input",
		"both",
	):
		panels.append((
			"Input",
			input_series,
		))

	if mode in (
		"processed",
		"both",
	):
		panels.append((
			"Processed",
			processed_series,
		))

	n_rows = input_series.n_variates
	n_cols = len(panels)

	fig_width = max(
		7.0,
		7.0 * n_cols,
	)
	fig_height = max(
		4.0,
		3.6 * n_rows,
	)

	fig, axes = plt.subplots(
		n_rows,
		n_cols,
		figsize=(
			fig_width,
			fig_height,
		),
		squeeze=False,
	)

	for row in range(
		n_rows
	):
		for column, (
			panel_name,
			series,
		) in enumerate(panels):
			ax = axes[
				row,
				column,
			]

			x_coord, x_label = make_index_coordinate(
				series
			)

			if panel_name == "Input":
				_plot_input_channel(
					ax,
					series,
					row,
					x_coord,
				)

			else:
				_plot_processed_channel(
					ax,
					series,
					row,
					x_coord,
				)

				_overlay_input_on_processed(
					ax,
					input_series,
					processed_series,
					row,
					x_coord,
				)

			_add_dual_time_axis(
				ax,
				series,
				x_coord,
				x_label,
			)

			ax.set_ylabel(
				_channel_name(
					series,
					row,
				)
			)

			ax.set_title(
				f"{panel_name}: "
				f"{_channel_name(series, row)}"
			)

			ax.grid(
				True,
				alpha=0.25,
			)

			handles, labels = ax.get_legend_handles_labels()

			if handles:
				unique = dict(
					zip(
						labels,
						handles,
					)
				)

				ax.legend(
					unique.values(),
					unique.keys(),
					loc="best",
					fontsize="small",
				)

	if title:
		fig.suptitle(
			title
		)

	fig.tight_layout()

	output_path = Path(
		output_path
	)
	output_path.parent.mkdir(
		parents=True,
		exist_ok=True,
	)

	fig.savefig(
		output_path,
		dpi=150,
		bbox_inches="tight",
	)
	plt.close(
		fig
	)

	logger.info(
		"Saved time-series diagnostic plot: '%s'",
		output_path,
	)

	return output_path
