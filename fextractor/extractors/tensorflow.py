"""Generic TensorFlow/Keras representation extractor."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import FeatureExtractor
from ..config import ExtractorConfig
from ..image_utils import ensure_channels, resize_square
from ..preprocessing import ImagePreprocessConfig, apply_image_preprocessing, read_image


class TensorFlowFeatureExtractor(FeatureExtractor):
	"""Extract representations from an exported TensorFlow/Keras encoder.

	The model is intentionally treated as a generic representation model. An
	exported SimCLR encoder therefore requires no SimCLR-specific code.
	"""

	backend = "tensorflow"
	modality = "image"

	def __init__(
		self,
		model_path: str | Path,
		weights_path: str | Path | None = None,
		imgsize: int = 224,
		in_chans: int = 1,
		preprocessing: ImagePreprocessConfig | None = None,
	) -> None:
		super().__init__()
		self.model_path = str(model_path)
		self.weights_path = str(weights_path) if weights_path else None
		self.imgsize = imgsize
		self.in_chans = in_chans
		self.preprocessing = preprocessing or ImagePreprocessConfig()
		self.model = None

	def load_model(self) -> None:
		"""Load the Keras model and optional separate weights file."""
		try:
			from tensorflow.keras.models import load_model
		except ImportError as exc:
			raise RuntimeError("TensorFlow backend requested, but TensorFlow is not installed. Install fextractor[tensorflow].") from exc

		self.model = load_model(self.model_path, compile=False)
		if self.weights_path:
			self.model.load_weights(self.weights_path)

	def prepare(self, source: str | Path) -> np.ndarray:
		"""Read and prepare one model input batch."""
		data = read_image(source)
		data = apply_image_preprocessing(data, self.preprocessing)
		data = resize_square(data, self.imgsize)
		data = ensure_channels(data, self.in_chans)
		return np.expand_dims(np.asarray(data, dtype=np.float32), axis=0)

	def extract(self, source: str | Path) -> np.ndarray:
		"""Return a one-dimensional representation vector."""
		self.ensure_loaded()
		batch = self.prepare(source)
		prediction = self.model.predict(batch, batch_size=1, verbose=0)
		if isinstance(prediction, (list, tuple)):
			if len(prediction) != 1:
				raise ValueError("TensorFlow model returned multiple outputs; select/export the desired encoder output first")
			prediction = prediction[0]
		features = np.asarray(prediction)
		if features.ndim > 1 and features.shape[0] == 1:
			features = features[0]
		return np.asarray(features, dtype=np.float32).reshape(-1)

	def metadata(self) -> dict:
		"""Return model and preprocessing provenance."""
		metadata = super().metadata()
		metadata.update({
			"model": self.model_path,
			"weights": self.weights_path,
			"imgsize": self.imgsize,
			"in_chans": self.in_chans,
			"preprocessing": self.preprocessing.__dict__.copy(),
		})
		return metadata


def create(config: ExtractorConfig) -> TensorFlowFeatureExtractor:
	"""Create a TensorFlow extractor from a backend-neutral configuration."""
	if not config.model:
		raise ValueError("A model path is required for the TensorFlow backend")

	return TensorFlowFeatureExtractor(
		model_path=config.model,
		weights_path=config.model_weights,
		imgsize=config.imgsize if config.imgsize is not None else 224,
		in_chans=config.in_chans if config.in_chans is not None else 1,
		preprocessing=config.preprocessing,
	)
