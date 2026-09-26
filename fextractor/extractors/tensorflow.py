"""Generic TensorFlow/Keras representation extractor."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..base import FeatureExtractor
from ..config import ExtractorConfig
from ..image.utils import ensure_channels, resize_square
from ..preprocessing import ImagePreprocessConfig, apply_image_preprocessing, read_image

logger = logging.getLogger(__name__)

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
		keras_loader: str = "auto",
	) -> None:
		""" Constructor """
		super().__init__()
		self.model_path = str(model_path)
		self.weights_path = str(weights_path) if weights_path else None
		self.imgsize = imgsize
		self.in_chans = in_chans
		self.preprocessing = preprocessing or ImagePreprocessConfig()
		self.keras_loader = keras_loader
		self.active_keras_loader = None
		self.model = None

	def _configure_gpu_memory(self) -> None:
		"""Enable incremental TensorFlow GPU memory allocation."""
		try:
			import tensorflow as tf
		except ImportError:
			return

		gpus = tf.config.list_physical_devices("GPU")

		if not gpus:
			logger.info("No TensorFlow GPU devices detected")
			return

		logger.info(
			"TensorFlow detected %d GPU device(s)",
			len(gpus),
		)

		for gpu in gpus:
			try:
				tf.config.experimental.set_memory_growth(gpu, True)

				logger.debug(
					"Enabled memory growth for GPU '%s'",
					gpu.name,
				)

			except RuntimeError as exc:
				logger.warning(
					"Could not enable memory growth for GPU '%s': %s",
					gpu.name,
					exc,
				)

	
	def load_model(self) -> None:
		"""Load the Keras model and optional separate weights file."""
		self._configure_gpu_memory()

		logger.info(
			"Loading TensorFlow model='%s' loader='%s'",
			self.model_path,
			self.keras_loader,
		)

		if self.keras_loader == "keras":
			self.model = self._load_with_keras()
			active_loader = "keras"

		elif self.keras_loader == "tf_keras":
			self.model = self._load_with_tf_keras()
			active_loader = "tf_keras"

		elif self.keras_loader == "auto":
			try:
				self.model = self._load_with_keras()
				active_loader = "keras"

			except Exception as exc:
				logger.warning(
					"Failed to load model with current tf.keras: %s: %s. "
					"Retrying with legacy tf_keras",
					type(exc).__name__,
					exc,
				)

				self.model = self._load_with_tf_keras()
				active_loader = "tf_keras"

		else:
			raise ValueError(
				f"Unsupported keras loader '{self.keras_loader}'. "
				"Supported values: auto, keras, tf_keras"
			)

		self.active_keras_loader = active_loader

		logger.info(
			"TensorFlow model loaded successfully using loader='%s'",
			active_loader,
		)

		if self.weights_path:
			logger.info(
				"Loading TensorFlow weights='%s'",
				self.weights_path,
			)

			self.model.load_weights(self.weights_path)

			logger.info("TensorFlow weights loaded successfully")


	def _load_with_keras(self):
		"""Load model using the current TensorFlow/Keras implementation."""
		try:
			from tensorflow.keras.models import load_model
		except ImportError as exc:
			raise RuntimeError(
				"TensorFlow backend requested, but TensorFlow is not installed. "
				"Install fextractor[tensorflow]."
			) from exc

		return load_model(
			self.model_path,
			compile=False,
		)

	def _load_with_tf_keras(self):
		"""Load model using legacy tf_keras."""
		try:
			import tf_keras
		except ImportError as exc:
			raise RuntimeError(
				"Legacy Keras loader requested, but tf-keras is not installed. "
				"Install fextractor[tensorflow]."
			) from exc

		return tf_keras.models.load_model(
			self.model_path,
			compile=False,
		)

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

		logger.debug(
			"Extracting TensorFlow representation from '%s'",
			source,
		)

		batch = self.prepare(source)

		logger.debug(
			"TensorFlow input batch shape=%s dtype=%s",
			batch.shape,
			batch.dtype,
		)

		prediction = self.model.predict(
			batch,
			batch_size=1,
			verbose=0,
		)

		if isinstance(prediction, (list, tuple)):
			if len(prediction) != 1:
				raise ValueError(
					"TensorFlow model returned multiple outputs; "
					"select/export the desired encoder output first"
				)

			prediction = prediction[0]

		features = np.asarray(prediction)

		if features.ndim > 1 and features.shape[0] == 1:
			features = features[0]

		features = np.asarray(
			features,
			dtype=np.float32,
		).reshape(-1)

		logger.debug(
			"TensorFlow representation extracted: features=%d",
			features.size,
		)

		return features

	def metadata(self) -> dict:
		"""Return model and preprocessing provenance."""
		metadata = super().metadata()
		metadata.update({
			"model": self.model_path,
			"weights": self.weights_path,
			"imgsize": self.imgsize,
			"in_chans": self.in_chans,
			"keras_loader": self.keras_loader,
			"active_keras_loader": self.active_keras_loader,
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
		keras_loader=config.get_option("keras_loader", "auto"),
	)
