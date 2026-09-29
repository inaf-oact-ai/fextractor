"""Integration tests for the light-curve-python (LiCu) backend."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("light_curve")

from fextractor.cli import main as cli_main
from fextractor.config import ExtractorConfig
from fextractor.factory import create_extractor
from fextractor.preprocessing import TimeSeriesPreprocessConfig


pytestmark = pytest.mark.integration


def _write_irregular_multichannel_csv(path) -> None:
	"""Write a deterministic irregular two-channel time series."""

	n_samples = 256
	index = np.arange(
		n_samples,
		dtype=np.float64,
	)

	# Deliberately irregular but strictly increasing sampling.
	dt = (
		0.8
		+ 0.25
		* (
			1.0
			+ np.sin(
				index
				* 0.37
			)
		)
	)

	time = np.cumsum(
		dt
	)

	flux_ratio = (
		1.0
		+ 0.12
		* np.sin(
			2.0
			* np.pi
			* time
			/ 17.3
		)
		+ 0.03
		* np.sin(
			2.0
			* np.pi
			* time
			/ 4.7
		)
	)

	flare_history = (
		0.35
		+ 0.20
		* np.cos(
			2.0
			* np.pi
			* time
			/ 23.1
			+ 0.31
		)
		+ 0.04
		* np.sin(
			2.0
			* np.pi
			* time
			/ 6.2
		)
	)

	# Positive, finite uncertainties exercise the sigma path even though the
	# current default LiCu feature set does not require them.
	flux_ratio_error = (
		0.02
		+ 0.003
		* (
			1.0
			+ np.sin(
				index
				* 0.19
			)
		)
	)

	flare_history_error = (
		0.03
		+ 0.003
		* (
			1.0
			+ np.cos(
				index
				* 0.23
			)
		)
	)

	with path.open(
		"w",
		encoding="utf-8",
	) as handle:
		handle.write(
			"time,flux_ratio,flare_history,"
			"flux_ratio_error,flare_history_error\n"
		)

		for sample_index in range(
			n_samples
		):
			handle.write(
				f"{time[sample_index]:.12g},"
				f"{flux_ratio[sample_index]:.12g},"
				f"{flare_history[sample_index]:.12g},"
				f"{flux_ratio_error[sample_index]:.12g},"
				f"{flare_history_error[sample_index]:.12g}\n"
			)


def test_licu_real_library_multichannel(tmp_path):
	"""Exercise the real library through the normal factory path."""

	path = tmp_path / "licu_series.csv"
	_write_irregular_multichannel_csv(
		path
	)

	config = ExtractorConfig(
		backend="licu",
		device="cpu",
		preprocessing=TimeSeriesPreprocessConfig(
			time_column="time",
			value_columns=(
				"flux_ratio",
				"flare_history",
			),
			error_columns=(
				"flux_ratio_error",
				"flare_history_error",
			),
			channel_names=(
				"flux_ratio",
				"flare_history",
			),
			regularize=False,
		),
		options={
			"feature_set": "default",
			"invalid_feature_policy": "error",
			"min_samples": 5,
		},
	)

	extractor = create_extractor(
		config
	)

	features = extractor.extract(
		path
	)

	metadata = extractor.metadata()

	assert features.ndim == 1
	assert np.all(
		np.isfinite(
			features
		)
	)

	assert metadata["backend"] == "licu"
	assert metadata["modality"] == "timeseries"
	assert metadata["library"] == "light-curve"
	assert metadata["library_version"] is not None
	assert metadata["feature_set"] == "default"
	assert metadata["invalid_feature_policy"] == "error"
	assert metadata["features_per_channel"] > 0
	assert metadata["n_features"] == len(
		features
	)
	assert len(
		metadata["feature_names"]
	) == len(
		features
	)
	assert len(
		metadata["channels"]
	) == 2

	features_per_channel = metadata[
		"features_per_channel"
	]

	assert len(features) == (
		2
		* features_per_channel
	)

	assert metadata[
		"feature_names"
	][0].startswith(
		"flux_ratio."
	)

	assert metadata[
		"feature_names"
	][features_per_channel].startswith(
		"flare_history."
	)

	for channel in metadata[
		"channels"
	]:
		assert channel["n_input"] == 256
		assert channel["n_valid"] == 256
		assert channel["n_features"] == features_per_channel
		assert channel["errors_used"] is True
		assert channel["invalid_features"] == []



def test_licu_cli_json_output(tmp_path):
	"""Exercise the same CLI/output path later used by CAESAR-REST."""

	input_path = tmp_path / "licu_cli_series.csv"
	output_path = tmp_path / "licu_results.json"

	_write_irregular_multichannel_csv(
		input_path
	)

	exit_code = cli_main([
		"--backend=licu",
		f"--inputfile={input_path}",
		f"--outfile={output_path}",
		"--profile=default",
		"--time-column=time",
		"--value-columns",
		"flux_ratio",
		"flare_history",
		"--error-columns",
		"flux_ratio_error",
		"flare_history_error",
		"--channel-names",
		"flux_ratio",
		"flare_history",
		"--feature-set=default",
		"--invalid-feature-policy=error",
		"--min-samples=5",
		"--no-regularize",
		"--device=cpu",
	])

	assert exit_code == 0
	assert output_path.exists()

	with output_path.open(
		"r",
		encoding="utf-8",
	) as handle:
		payload = json.load(
			handle
		)

	assert "feats" in payload
	assert "fextractor" in payload

	features = np.asarray(
		payload["feats"],
		dtype=np.float64,
	)

	metadata = payload[
		"fextractor"
	]

	assert features.ndim == 1
	assert features.size > 0
	assert np.all(
		np.isfinite(
			features
		)
	)

	assert metadata["backend"] == "licu"
	assert metadata["modality"] == "timeseries"
	assert metadata["library"] == "light-curve"
	assert metadata["library_version"] is not None
	assert metadata["feature_set"] == "default"
	assert metadata["invalid_feature_policy"] == "error"
	assert metadata["min_samples"] == 5
	assert metadata["n_features"] == int(
		features.size
	)
	assert len(
		metadata["feature_names"]
	) == int(
		features.size
	)
	assert len(
		metadata["channels"]
	) == 2

	# LiCu must preserve the irregularly sampled input by default. This guards
	# against accidentally applying the regularization required by Chronos.
	preprocessing = metadata[
		"preprocessing"
	]
	assert preprocessing["regularize"] is False
	assert preprocessing["time_transform"] == "none"
	assert preprocessing["value_transform"] == "none"
