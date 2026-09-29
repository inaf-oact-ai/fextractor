"""Time-series model-input sample selection policies."""

from __future__ import annotations

import numpy as np

from .data import TimeSeries


DEFAULT_INPUT_SAMPLE_POLICY = "observed"

SUPPORTED_INPUT_SAMPLE_POLICIES = (
	"observed",
	"completed",
)


def build_input_sample_mask(
	series: TimeSeries,
	policy: str = DEFAULT_INPUT_SAMPLE_POLICY,
) -> np.ndarray:
	"""Return the finite samples that a backend is allowed to consume.

	The masks stored in ``TimeSeries`` describe sample provenance and are not
	modified here. ``observed`` consumes only measured/bin-observed samples.
	``completed`` additionally consumes interpolated and GP-predicted samples.
	"""

	if policy not in SUPPORTED_INPUT_SAMPLE_POLICIES:
		raise ValueError(
		f"Unsupported input_sample_policy '{policy}'. "
		f"Supported values: {', '.join(SUPPORTED_INPUT_SAMPLE_POLICIES)}"
	)

	if (
		policy == "observed"
		and series.metadata.get("regularization_method") == "gp"
	):
		raise ValueError(
			"input_sample_policy='observed' is incompatible with GP "
			"regularization because the GP output series contains predicted "
			"samples rather than observed samples. Use "
			"input_sample_policy='completed' or disable GP regularization."
		)

	if policy == "observed":
		provenance_mask = series.observed_mask

	else:
		provenance_mask = (
			series.observed_mask
			| series.interpolated_mask
			| series.predicted_mask
		)

	return (
		np.asarray(
			provenance_mask,
			dtype=bool,
		)
		& np.isfinite(
			series.values
		)
	)
