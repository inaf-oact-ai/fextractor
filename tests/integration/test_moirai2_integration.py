import numpy as np
import pytest

from fextractor.config import ExtractorConfig
from fextractor.factory import create_extractor
from fextractor.preprocessing import TimeSeriesPreprocessConfig


pytestmark = pytest.mark.integration


def test_moirai2_real_model(tmp_path):
	path = tmp_path / "series.csv"

	time = np.arange(
		1440,
		dtype=np.float64,
	)

	flux_ratio = (
		1.0
		+ 0.1
		* np.sin(
			time
			/ 20.0
		)
	)

	flare_history = np.zeros(
		1440,
		dtype=np.float32,
	)

	flare_history[
		1000:1100
	] = 1.0

	with path.open(
		"w",
		encoding="utf-8",
	) as handle:
		handle.write(
			"time,flux_ratio,flare_history\n"
		)

		for index in range(
			1440
		):
			handle.write(
				f"{time[index]},"
				f"{flux_ratio[index]},"
				f"{flare_history[index]}\n"
			)

	config = ExtractorConfig(
		backend="moirai2",
		device="cpu",
		preprocessing=TimeSeriesPreprocessConfig(
			time_column="time",
			value_columns=(
				"flux_ratio",
				"flare_history",
			),
		),
		options={
			"aggregation": "mean_std",
			"patching_mode": "time_variate",
			"token_order": "by_variate",
		},
	)

	extractor = create_extractor(
		config
	)

	features = extractor.extract(
		path
	)

	print(
		"Moirai2 feature shape:",
		features.shape,
	)

	print(
		"Moirai2 metadata:",
		extractor.metadata(),
	)

	assert features.ndim == 1

	# Moirai-2 small uses d_model=384.
	# mean_std concatenates mean and std.
	assert features.shape == (
		768,
	)

	assert np.all(
		np.isfinite(features)
	)
	
	
def test_moirai2_interleave_time(tmp_path):
	path = tmp_path / "series.csv"

	time = np.arange(
		1440,
		dtype=np.float64,
	)

	flux_ratio = (
		1.0
		+ 0.1
		* np.sin(
			time
			/ 20.0
		)
	)

	flare_history = np.zeros(
		1440,
		dtype=np.float32,
	)

	flare_history[
		1000:1100
	] = 1.0

	with path.open(
		"w",
		encoding="utf-8",
	) as handle:
		handle.write(
			"time,flux_ratio,flare_history\n"
		)

		for index in range(
			1440
		):
			handle.write(
				f"{time[index]},"
				f"{flux_ratio[index]},"
				f"{flare_history[index]}\n"
			)

	common_preprocessing = TimeSeriesPreprocessConfig(
		time_column="time",
		value_columns=(
			"flux_ratio",
			"flare_history",
		),
	)

	config_by_variate = ExtractorConfig(
		backend="moirai2",
		device="cpu",
		preprocessing=common_preprocessing,
		options={
			"aggregation": "mean_std",
			"patching_mode": "time_variate",
			"token_order": "by_variate",
		},
	)

	config_interleave = ExtractorConfig(
		backend="moirai2",
		device="cpu",
		preprocessing=common_preprocessing,
		options={
			"aggregation": "mean_std",
			"patching_mode": "time_variate",
			"token_order": "interleave_time",
		},
	)

	extractor_by_variate = create_extractor(
		config_by_variate
	)

	extractor_interleave = create_extractor(
		config_interleave
	)

	features_by_variate = (
		extractor_by_variate.extract(
			path
		)
	)

	features_interleave = (
		extractor_interleave.extract(
			path
		)
	)

	assert features_by_variate.ndim == 1
	assert features_interleave.ndim == 1

	assert features_by_variate.shape == (
		768,
	)

	assert features_interleave.shape == (
		768,
	)

	assert np.all(
		np.isfinite(
			features_by_variate
		)
	)

	assert np.all(
		np.isfinite(
			features_interleave
		)
	)

	assert not np.allclose(
		features_by_variate,
		features_interleave,
	)
