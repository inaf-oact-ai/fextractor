"""Single-channel ML embeddings from light-curve-python."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from ...config import ExtractorConfig
from ...preprocessing import TimeSeriesPreprocessConfig
from ...timeseries import DEFAULT_INPUT_SAMPLE_POLICY, TimeSeries
from ...timeseries.aggregation import (
	TokenRepresentation,
	aggregate_token_representation,
)
from .base import TimeSeriesFeatureExtractor


logger = logging.getLogger(__name__)


DEFAULT_OUTPUT = "mean"
DEFAULT_AGGREGATION = "mean"
DEFAULT_MIN_SAMPLES = 2

SUPPORTED_OUTPUTS = (
	"mean",
	"max",
	"sequence",
)


@dataclass(frozen=True)
class LiCuEmbeddingModelSpec:
	"""Static description of one supported light-curve embedding model."""

	name: str
	class_name: str
	filename: str
	input_kind: str
	embedding_dim: int
	default_reduction: str
	supported_outputs: tuple[str, ...]
	size: str | None = None
	aliases: tuple[str, ...] = ()


_MODEL_SPECS = (
	LiCuEmbeddingModelSpec(
		name="astromer1",
		class_name="Astromer1",
		filename="astromer1.onnx",
		input_kind="time_value",
		embedding_dim=256,
		default_reduction="non-overlapping-windows",
		supported_outputs=("mean", "max", "sequence"),
	),
	LiCuEmbeddingModelSpec(
		name="astromer1-ztfdr20",
		class_name="Astromer1ZTF",
		filename="astromer1_ztfdr20.onnx",
		input_kind="time_value",
		embedding_dim=256,
		default_reduction="non-overlapping-windows",
		supported_outputs=("mean", "max", "sequence"),
		aliases=(
			"astromer1_ztfdr20",
			"astromer1ztf",
			"astromer1-ztf",
		),
	),
	LiCuEmbeddingModelSpec(
		name="astromer2",
		class_name="Astromer2",
		filename="astromer2.onnx",
		input_kind="time_value",
		embedding_dim=256,
		default_reduction="non-overlapping-windows",
		supported_outputs=("mean", "max", "sequence"),
	),
	LiCuEmbeddingModelSpec(
		name="moment1-small",
		class_name="Moment1",
		filename="moment1-small.onnx",
		input_kind="value",
		embedding_dim=512,
		default_reduction="end",
		supported_outputs=("mean", "sequence"),
		size="small",
		aliases=("moment-small",),
	),
	LiCuEmbeddingModelSpec(
		name="moment1-base",
		class_name="Moment1",
		filename="moment1-base.onnx",
		input_kind="value",
		embedding_dim=768,
		default_reduction="end",
		supported_outputs=("mean", "sequence"),
		size="base",
		aliases=("moment-base",),
	),
	LiCuEmbeddingModelSpec(
		name="moment1-large",
		class_name="Moment1",
		filename="moment1-large.onnx",
		input_kind="value",
		embedding_dim=1024,
		default_reduction="end",
		supported_outputs=("mean", "sequence"),
		size="large",
		aliases=("moment-large",),
	),
)

_MODEL_BY_NAME: dict[str, LiCuEmbeddingModelSpec] = {}
for _spec in _MODEL_SPECS:
	_MODEL_BY_NAME[_spec.name] = _spec
	for _alias in _spec.aliases:
		_MODEL_BY_NAME[_alias] = _spec


SUPPORTED_MODELS = tuple(
	spec.name
	for spec in _MODEL_SPECS
)


def _normalise_model_key(value: str) -> str:
	"""Normalize a model name or model-directory basename."""

	return value.strip().lower().replace("_", "-")


def get_model_spec(model_ref: str | Path) -> LiCuEmbeddingModelSpec:
	"""Resolve a symbolic model name, model directory, or ONNX file."""

	path = Path(model_ref)

	candidates = [
		str(model_ref),
		path.name,
	]

	if path.suffix.lower() == ".onnx":
		candidates.append(path.stem)

	for candidate in candidates:
		key = _normalise_model_key(candidate)

		# Preserve the historical underscore in the actual ZTF filename while
		# accepting a path to that file as a model selector.
		if key == "astromer1-ztfdr20":
			return _MODEL_BY_NAME["astromer1-ztfdr20"]

		try:
			return _MODEL_BY_NAME[key]
		except KeyError:
			continue

	raise ValueError(
		"Unsupported LiCu embedding model "
		f"'{model_ref}'. Supported models: {', '.join(SUPPORTED_MODELS)}"
	)


def resolve_model_file(
	model_ref: str | Path,
	spec: LiCuEmbeddingModelSpec,
) -> Path:
	"""Resolve an ONNX file from an explicit file or model directory."""

	path = Path(model_ref).expanduser()

	if path.is_file():
		if path.suffix.lower() != ".onnx":
			raise ValueError(
				f"LiCu embedding model file must be ONNX, got '{path}'"
			)
		return path

	if path.is_dir():
		model_file = path / spec.filename
		if model_file.is_file():
			return model_file

		raise FileNotFoundError(
			f"Expected LiCu embedding model file '{spec.filename}' "
			f"inside directory '{path}'"
		)

	raise FileNotFoundError(
		f"LiCu embedding model path does not exist: '{path}'. "
		"Pass the model directory (for example /opt/models/astromer2) "
		"or the ONNX file itself."
	)


class LiCuEmbeddingFeatureExtractor(TimeSeriesFeatureExtractor):
	"""Extract pretrained single-channel light-curve embeddings.

	Supported models are Astromer 1, Astromer 1 ZTF DR20, Astromer 2, and
	MOMENT-1 small/base/large.  Every canonical fextractor channel is processed
	independently by the selected single-channel model and the resulting channel
	vectors are concatenated in channel order.

	The model must be supplied as a local directory or ONNX file.  No network
	download is performed by this backend.
	"""

	backend = "licu_embed"
	modality = "timeseries"

	def __init__(
		self,
		model_name: str,
		device: str = "cuda",
		output: str = DEFAULT_OUTPUT,
		aggregation: str = DEFAULT_AGGREGATION,
		reduction: str | None = None,
		min_samples: int = DEFAULT_MIN_SAMPLES,
		preprocessing: TimeSeriesPreprocessConfig | None = None,
		input_sample_policy: str = DEFAULT_INPUT_SAMPLE_POLICY,
	) -> None:
		super().__init__(
			preprocessing=preprocessing,
			input_sample_policy=input_sample_policy,
		)

		self.model_name = str(model_name)
		self.model_spec = get_model_spec(self.model_name)

		if output not in self.model_spec.supported_outputs:
			raise ValueError(
				f"Output '{output}' is not supported by "
				f"{self.model_spec.name}. Supported outputs: "
				+ ", ".join(self.model_spec.supported_outputs)
			)

		if min_samples < 1:
			raise ValueError(
				"LiCu embedding min_samples must be at least 1"
			)

		self.requested_device = device
		self.device = device
		self.output = output
		self.aggregation = aggregation
		self.reduction = (
			reduction
			or self.model_spec.default_reduction
		)
		self.min_samples = int(min_samples)

		self.model_path: str | None = None
		self.provider: str | None = None
		self.providers: list[str] = []
		self.library_version: str | None = None
		self.model = None
		self.channel_metadata: list[dict] = []

	def _select_onnx_providers(self, ort) -> list[str]:
		"""Translate the requested fextractor device into ONNX providers."""

		available = list(ort.get_available_providers())

		if self.requested_device.lower().startswith("cuda"):
			if "CUDAExecutionProvider" in available:
				providers = ["CUDAExecutionProvider"]
				if "CPUExecutionProvider" in available:
					providers.append("CPUExecutionProvider")
				self.device = self.requested_device
				return providers

			logger.warning(
				"CUDA requested for LiCu embedding model but "
				"CUDAExecutionProvider is unavailable; falling back to CPU"
			)
			self.device = "cpu"

		if "CPUExecutionProvider" not in available:
			raise RuntimeError(
				"ONNX Runtime CPUExecutionProvider is unavailable. "
				f"Available providers: {available}"
			)

		return ["CPUExecutionProvider"]

	def load_model(self) -> None:
		"""Load one local LICU ONNX embedding model."""

		try:
			import light_curve
			from light_curve import embed as licu_embed
			import onnxruntime as ort
		except ImportError as exc:
			raise RuntimeError(
				"LiCu embedding backend requires light-curve and an "
				"ONNX Runtime variant (onnxruntime or onnxruntime-gpu)."
			) from exc

		model_path = resolve_model_file(
			self.model_name,
			self.model_spec,
		)

		self.library_version = getattr(
			light_curve,
			"__version__",
			None,
		)
		self.providers = self._select_onnx_providers(ort)

		logger.info(
			"Loading LiCu embedding model='%s' path='%s' "
			"requested_device='%s' providers=%s output='%s' reduction='%s'",
			self.model_spec.name,
			model_path,
			self.requested_device,
			self.providers,
			self.output,
			self.reduction,
		)

		session = ort.InferenceSession(
			str(model_path),
			providers=self.providers,
		)

		model_class = getattr(
			licu_embed,
			self.model_spec.class_name,
		)

		kwargs = {
			"session": session,
			"output": self.output,
			"reduction": self.reduction,
		}

		if self.model_spec.size is not None:
			kwargs["size"] = self.model_spec.size

		self.model = model_class(
			**kwargs
		)

		self.model_path = str(model_path)
		self.provider = (
			session.get_providers()[0]
			if session.get_providers()
			else None
		)

		logger.info(
			"LiCu embedding model loaded successfully: "
			"model='%s' provider='%s' embedding_dim=%d",
			self.model_spec.name,
			self.provider,
			self.model_spec.embedding_dim,
		)

	def _channel_name(
		self,
		series: TimeSeries,
		channel_index: int,
	) -> str:
		if series.channel_names is not None:
			return str(series.channel_names[channel_index])
		return f"channel_{channel_index}"

	def _select_channel_samples(
		self,
		series: TimeSeries,
		channel_index: int,
	) -> tuple[np.ndarray | None, np.ndarray]:
		"""Select finite samples for one independent model invocation."""

		values = np.asarray(
			series.values[:, channel_index],
			dtype=np.float64,
		)
		selected = self.get_input_sample_mask(series)[:, channel_index]
		valid = selected & np.isfinite(values)

		time = None
		if series.times is not None:
			time_all = np.asarray(
				series.times,
				dtype=np.float64,
			)
			valid &= np.isfinite(time_all)
			time = time_all[valid]
		elif self.model_spec.input_kind == "time_value":
			raise ValueError(
				f"{self.model_spec.name} requires explicit timestamps"
			)

		values = values[valid]

		if values.size < self.min_samples:
			raise ValueError(
				f"LiCu embedding channel "
				f"'{self._channel_name(series, channel_index)}' has only "
				f"{values.size} valid selected samples; at least "
				f"{self.min_samples} are required"
			)

		if time is not None:
			order = np.argsort(time, kind="stable")
			time = time[order]
			values = values[order]

		return time, values

	def _run_channel_model(
		self,
		time: np.ndarray | None,
		values: np.ndarray,
	) -> np.ndarray:
		"""Run the configured LICU model for one channel."""

		if self.model is None:
			raise RuntimeError(
				"LiCu embedding model is not loaded"
			)

		if self.model_spec.input_kind == "time_value":
			if time is None:
				raise RuntimeError(
					"Internal error: timestamp array is missing"
				)
			embedding = self.model(time, values)
		else:
			embedding = self.model(values)

		embedding = np.asarray(
			embedding,
			dtype=np.float32,
		)

		if embedding.ndim < 1:
			raise RuntimeError(
				"Unexpected LiCu embedding shape: "
				f"{embedding.shape}"
			)

		if embedding.shape[-1] != self.model_spec.embedding_dim:
			raise RuntimeError(
				"Unexpected LiCu embedding dimension for "
				f"{self.model_spec.name}: got {embedding.shape[-1]}, "
				f"expected {self.model_spec.embedding_dim}; "
				f"full shape={embedding.shape}"
			)

		return embedding

	def _aggregate_channel_embedding(
		self,
		embedding: np.ndarray,
	) -> np.ndarray:
		"""Reduce model-native axes to one fixed vector per channel."""

		tokens = embedding.reshape(
			-1,
			embedding.shape[-1],
		)

		return aggregate_token_representation(
			TokenRepresentation(
				context_tokens=tokens,
			),
			strategy=self.aggregation,
		)

	def extract_timeseries(
		self,
		series: TimeSeries,
	) -> np.ndarray:
		"""Embed every canonical channel independently and concatenate."""

		if self.model is None:
			raise RuntimeError(
				"LiCu embedding model is not loaded"
			)

		all_features: list[np.ndarray] = []
		channel_metadata: list[dict] = []

		for channel_index in range(series.n_variates):
			channel_name = self._channel_name(
				series,
				channel_index,
			)
			time, values = self._select_channel_samples(
				series,
				channel_index,
			)

			logger.info(
				"Extracting LiCu embedding: model='%s' channel='%s' "
				"input_samples=%d valid_samples=%d",
				self.model_spec.name,
				channel_name,
				series.n_time,
				values.size,
			)

			embedding = self._run_channel_model(
				time,
				values,
			)
			features = self._aggregate_channel_embedding(
				embedding
			)

			all_features.append(features)
			channel_metadata.append({
				"name": channel_name,
				"n_input": int(series.n_time),
				"n_valid": int(values.size),
				"native_embedding_shape": list(embedding.shape),
				"n_features": int(features.size),
			})

		self.channel_metadata = channel_metadata

		if not all_features:
			raise ValueError(
				"LiCu embedding received a time series with no channels"
			)

		return np.concatenate(
			all_features,
		).astype(
			np.float32,
			copy=False,
		)

	def metadata(self) -> dict:
		"""Return model, runtime, and embedding provenance."""

		metadata = super().metadata()
		metadata.update({
			"library": "light-curve",
			"library_version": self.library_version,
			"model": self.model_spec.name,
			"model_path": self.model_path,
			"requested_device": self.requested_device,
			"device": self.device,
			"provider": self.provider,
			"providers": self.providers,
			"output": self.output,
			"aggregation": self.aggregation,
			"reduction": self.reduction,
			"embedding_dim": self.model_spec.embedding_dim,
			"min_samples": self.min_samples,
			"channel_mode": "independent",
			"channels": self.channel_metadata,
			"preprocessing": self.preprocessing.__dict__.copy(),
		})
		return metadata


def create(
	config: ExtractorConfig,
) -> LiCuEmbeddingFeatureExtractor:
	"""Create a local LICU ML embedding extractor from ExtractorConfig."""

	if not config.model:
		raise ValueError(
		"LiCu embedding backend requires --model pointing to a local "
		"model directory or ONNX file"
	)

	preprocessing = config.preprocessing
	if not isinstance(
		preprocessing,
		TimeSeriesPreprocessConfig,
	):
		preprocessing = TimeSeriesPreprocessConfig()

	return LiCuEmbeddingFeatureExtractor(
		model_name=config.model,
		device=config.device,
		output=config.get_option(
			"licu_embed_output",
			DEFAULT_OUTPUT,
		),
		aggregation=config.get_option(
			"aggregation",
			DEFAULT_AGGREGATION,
		),
		reduction=config.get_option(
			"licu_embed_reduction",
			None,
		),
		min_samples=config.get_option(
			"min_samples",
			DEFAULT_MIN_SAMPLES,
		),
		preprocessing=preprocessing,
		input_sample_policy=config.get_option(
			"input_sample_policy",
			DEFAULT_INPUT_SAMPLE_POLICY,
		),
	)
