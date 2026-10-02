import numpy as np
import pytest

from fextractor.extractors.timeseries.falcon1 import Falcon1FeatureExtractor
from fextractor.timeseries import TimeSeries


def test_falcon1_univariate_input_conversion():
	extractor = Falcon1FeatureExtractor(device="cpu")
	series = TimeSeries(
		values=np.asarray([1.0, 2.0, 3.0]),
		times=np.asarray([0.0, 1.0, 2.0]),
	)

	values, mask = extractor._to_falcon_input(series)

	assert values.shape == (1, 3)
	assert mask.shape == (1, 3)
	np.testing.assert_allclose(values, [[1.0, 2.0, 3.0]])
	assert np.all(mask)


def test_falcon1_multivariate_input_conversion():
	extractor = Falcon1FeatureExtractor(device="cpu")
	series = TimeSeries(
		values=np.asarray([
			[1.0, 10.0],
			[2.0, 20.0],
			[3.0, 30.0],
		]),
		times=np.asarray([0.0, 1.0, 2.0]),
	)

	values, mask = extractor._to_falcon_input(series)

	assert values.shape == (2, 3)
	assert mask.shape == (2, 3)
	np.testing.assert_allclose(values[0], [1.0, 2.0, 3.0])
	np.testing.assert_allclose(values[1], [10.0, 20.0, 30.0])


def test_falcon1_context_length_keeps_latest_samples():
	extractor = Falcon1FeatureExtractor(device="cpu", context_length=2)
	series = TimeSeries(values=np.asarray([1.0, 2.0, 3.0, 4.0]))

	values, mask = extractor._to_falcon_input(series)

	np.testing.assert_allclose(values, [[3.0, 4.0]])
	assert mask.shape == (1, 2)


def test_falcon1_respects_input_sample_policy():
	series = TimeSeries(
		values=np.asarray([1.0, 2.0, 3.0]),
		observed_mask=np.asarray([True, False, True]),
		interpolated_mask=np.asarray([False, True, False]),
	)

	observed = Falcon1FeatureExtractor(device="cpu", input_sample_policy="observed")
	completed = Falcon1FeatureExtractor(device="cpu", input_sample_policy="completed")

	_, observed_mask = observed._to_falcon_input(series)
	_, completed_mask = completed._to_falcon_input(series)

	np.testing.assert_array_equal(observed_mask, [[True, False, True]])
	np.testing.assert_array_equal(completed_mask, [[True, True, True]])


def test_falcon1_rejects_irregular_timestamps():
	extractor = Falcon1FeatureExtractor(device="cpu")
	series = TimeSeries(
		values=np.asarray([1.0, 2.0, 3.0]),
		times=np.asarray([0.0, 1.0, 5.0]),
	)

	with pytest.raises(ValueError, match="regularly sampled"):
		extractor._to_falcon_input(series)


def test_falcon1_rejects_empty_selected_channel():
	extractor = Falcon1FeatureExtractor(device="cpu")
	series = TimeSeries(
		values=np.asarray([
			[1.0, np.nan],
			[2.0, np.nan],
			[3.0, np.nan],
		]),
	)

	with pytest.raises(ValueError, match="at least one selected finite sample per channel"):
		extractor._to_falcon_input(series)
		
		
def test_falcon1_token_representation(monkeypatch):
	extractor = Falcon1FeatureExtractor(device="cpu")
	extractor.torch = pytest.importorskip("torch")

	class FakeConfig:
		mask_pad_value = 255.0

	class FakeCoreModel:
		config = FakeConfig()

	class FakeModel:
		def parameters(self):
			yield extractor.torch.nn.Parameter(extractor.torch.zeros(1))

	extractor.model = FakeModel()

	monkeypatch.setattr(extractor, "_get_core_model", lambda: FakeCoreModel())
	monkeypatch.setattr(
		extractor,
		"_capture_shared_expert_tokens",
		lambda input_tensor: np.asarray([
			[
				[1.0, 2.0],
				[3.0, 4.0],
			]
		], dtype=np.float32),
	)

	series = TimeSeries(values=np.asarray([1.0, 2.0]))

	representation = extractor.extract_tokens(series)

	assert representation.context_tokens.shape == (1, 2, 2)
	np.testing.assert_allclose(
		representation.context_tokens,
		[[[1.0, 2.0], [3.0, 4.0]]],
	)
	
	
def test_falcon1_rejects_native_mask_sentinel(monkeypatch):
	extractor = Falcon1FeatureExtractor(device="cpu")
	extractor.torch = pytest.importorskip("torch")

	class FakeConfig:
		mask_pad_value = 255.0

	class FakeCoreModel:
		config = FakeConfig()

	class FakeModel:
		def parameters(self):
			yield extractor.torch.nn.Parameter(extractor.torch.zeros(1))

	extractor.model = FakeModel()
	monkeypatch.setattr(extractor, "_get_core_model", lambda: FakeCoreModel())

	series = TimeSeries(values=np.asarray([1.0, 255.0, 3.0]))

	with pytest.raises(ValueError, match="native mask sentinel"):
		extractor.extract_tokens(series)
		

