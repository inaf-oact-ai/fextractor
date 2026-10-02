import os

import numpy as np
import pytest

from fextractor.config import ExtractorConfig
from fextractor.factory import create_extractor
from fextractor.preprocessing import TimeSeriesPreprocessConfig


pytestmark = pytest.mark.integration


def test_falcon1_real_model(tmp_path):
	model_path = os.environ.get("FALCON1_MODEL_PATH", "ant-intl/Falcon-TST_Large")
	path = tmp_path / "series.csv"

	time = np.arange(2880, dtype=np.float64)
	flux_ratio = 1.0 + 0.1 * np.sin(time / 20.0)
	flare_history = np.zeros(2880, dtype=np.float32)
	flare_history[2000:2100] = 1.0

	with path.open("w", encoding="utf-8") as handle:
		handle.write("time,flux_ratio,flare_history\n")
		for index in range(2880):
			handle.write(f"{time[index]},{flux_ratio[index]},{flare_history[index]}\n")

	config = ExtractorConfig(
		backend="falcon1",
		model=model_path,
		device="cpu",
		preprocessing=TimeSeriesPreprocessConfig(
			time_column="time",
			value_columns=("flux_ratio", "flare_history"),
		),
		options={"aggregation": "mean_std"},
	)

	extractor = create_extractor(config)
	features = extractor.extract(path)

	assert features.ndim == 1
	assert features.shape == (2 * extractor._get_core_model().config.hidden_size,)
	assert np.all(np.isfinite(features))
