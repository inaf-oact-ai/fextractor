import numpy as np
import pytest

from fextractor.config import ExtractorConfig
from fextractor.factory import create_extractor
from fextractor.preprocessing import TimeSeriesPreprocessConfig


pytestmark = pytest.mark.integration


def test_chronos2_real_model(tmp_path):
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
		backend="chronos2",
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
			"batch_size": 16,
		},
	)

	extractor = create_extractor(
		config
	)

	features = extractor.extract(
		path
	)

	assert features.ndim == 1
	assert features.shape == (
		1536,
	)

	assert np.all(
		np.isfinite(features)
	)
