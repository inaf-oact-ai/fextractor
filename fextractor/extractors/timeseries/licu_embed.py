"""ML embeddings from light-curve-python."""

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
	default_output: str
	default_reduction: str
	supported_outputs: tuple[str, ...]
	size: str | None = None
	multiband: bool = False
	requires_errors: bool = False
	n_model_bands: int | None = None
	band_map: dict[str, int] | None = None
	default_mag_zp: float | None = None
	aliases: tuple[str, ...] = ()


_MODEL_SPECS = (
	LiCuEmbeddingModelSpec(
		name="astromer1",
		class_name="Astromer1",
		filename="astromer1.onnx",
		input_kind="time_value",
		embedding_dim=256,
		default_output="mean",
		default_reduction="non-overlapping-windows",
		supported_outputs=("mean", "max", "sequence"),
	),
	LiCuEmbeddingModelSpec(
		name="astromer1-ztfdr20",
		class_name="Astromer1ZTF",
		filename="astromer1_ztfdr20.onnx",
		input_kind="time_value",
		embedding_dim=256,
		default_output="mean",
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
		default_output="mean",
		default_reduction="non-overlapping-windows",
		supported_outputs=("mean", "max", "sequence"),
	),
	LiCuEmbeddingModelSpec(
		name="moment1-small",
		class_name="Moment1",
		filename="moment1-small.onnx",
		input_kind="value",
		embedding_dim=512,
		default_output="mean",
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
		default_output="mean",
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
		default_output="mean",
		default_reduction="end",
		supported_outputs=("mean", "sequence"),
		size="large",
		aliases=("moment-large",),
	),
	LiCuEmbeddingModelSpec(
		name="astra-clr",
		class_name="AstraCLR",
		filename="astra_clr.onnx",
		input_kind="time_value_error_band",
		embedding_dim=512,
		default_output="mean",
		default_reduction="beginning",
		supported_outputs=("mean",),
		multiband=True,
		requires_errors=True,
		n_model_bands=3,
		band_map={
			"g": 0,
			"r": 1,
			"i": 2,
		},
	),
	LiCuEmbeddingModelSpec(
		name="atat",
		class_name="ATAT",
		filename="atat.onnx",
		input_kind="time_value_band",
		embedding_dim=192,
		default_output="token",
		default_reduction="non-overlapping-windows",
		supported_outputs=(
			"token",
			"mean",
			"sequence",
		),
		multiband=True,
		requires_errors=False,
		n_model_bands=6,
		band_map={
			"u": 0,
			"g": 1,
			"r": 2,
			"i": 3,
			"z": 4,
			"y": 5,
		},
		default_mag_zp=31.4,
	),
	LiCuEmbeddingModelSpec(
		name="atcat",
		class_name="ATCAT",
		filename="atcat_f32.onnx",
		input_kind="time_value_error_band",
		embedding_dim=384,
		default_output="last",
		default_reduction="non-overlapping-windows",
		supported_outputs=(
			"last",
			"mean",
			"sequence",
		),
		multiband=True,
		requires_errors=True,
		n_model_bands=6,
		band_map={
			"u": 0,
			"g": 1,
			"r": 2,
			"i": 3,
			"z": 4,
			"y": 5,
		},
		default_mag_zp=31.4,
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
	"""Extract pretrained light-curve embeddings.

	Supported single-channel models are Astromer 1, Astromer 1 ZTF DR20,
	Astromer 2, and MOMENT-1 small/base/large. Supported multiband models
	are AstraCLR, ATAT, and ATCAT.

	Single-channel models process each canonical fextractor channel
	independently and concatenate the resulting vectors. Multiband models
	process the complete observation stream using timestamps, values,
	optional uncertainties, and per-observation band labels.

	The model must be supplied as a local directory or ONNX file.
	No network download is performed by this backend.
	"""

	backend = "licu_embed"
	modality = "timeseries"

	def __init__(
		self,
		model_name: str,
		device: str = "cuda",
		output: str | None = None,
		aggregation: str = DEFAULT_AGGREGATION,
		reduction: str | None = None,
		min_samples: int = DEFAULT_MIN_SAMPLES,
		mag_zp: float | None = None,
		allow_extra_bands: bool = False,
		preprocessing: TimeSeriesPreprocessConfig | None = None,
		input_sample_policy: str = DEFAULT_INPUT_SAMPLE_POLICY,
	) -> None:
	
		super().__init__(
			preprocessing=preprocessing,
			input_sample_policy=input_sample_policy,
		)

		self.model_name = str(model_name)
		self.model_spec = get_model_spec(self.model_name)

		if output is None:
			output = (
				self.model_spec.default_output
			)
			
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
			
		if self.model_spec.multiband:
			if self.preprocessing.layout == "wide":
				raise ValueError(
					f"{self.model_spec.name} requires long-layout "
					"observation data; wide layout is not supported"
				)

			if self.preprocessing.regularize:
				raise ValueError(
					f"{self.model_spec.name} does not support "
					"time-series regularization"
				)

			if self.preprocessing.value_transform != "none":
				raise ValueError(
					f"{self.model_spec.name} requires "
					"value_transform='none'"
				)

			if (
				self.model_spec.name == "astra-clr"
				and self.preprocessing.time_transform != "none"
			):
				raise ValueError(
					"AstraCLR requires absolute MJD timestamps; "
					"time_transform must be 'none'"
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

		self.mag_zp = (
			self.model_spec.default_mag_zp
			if mag_zp is None
			else float(mag_zp)
		)

		self.allow_extra_bands = bool(
			allow_extra_bands
		)

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
			"reduction": self.reduction,
		}

		if len(
			self.model_spec.supported_outputs
		) > 1:
			kwargs["output"] = (
				self.output
			)

		if self.model_spec.size is not None:
			kwargs["size"] = (
				self.model_spec.size
			)

		if self.model_spec.multiband:
			kwargs["allow_extra_bands"] = (
				self.allow_extra_bands
			)

		if (
			self.model_spec.default_mag_zp is not None
			and self.mag_zp is not None
		):
			kwargs["mag_zp"] = (
				self.mag_zp
			)		

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

	def _normalise_multiband_labels(
		self,
		bands: np.ndarray,
	) -> np.ndarray:
		"""Map supported string/integer band labels to model band indices."""

		if self.model_spec.band_map is None:
			raise RuntimeError(
				f"Model '{self.model_spec.name}' has no band mapping"
			)

		if self.model_spec.n_model_bands is None:
			raise RuntimeError(
				f"Model '{self.model_spec.name}' has no model-band count"
			)

		output = np.full(
			bands.shape[0],
			-1,
			dtype=np.int64,
		)

		for index, value in enumerate(
			bands
		):
			if isinstance(
				value,
				(
					int,
					np.integer,
				),
			):
				band_index = int(
					value
				)

			else:
				key = str(
					value
				).strip().lower()

				if key not in self.model_spec.band_map:
					if self.allow_extra_bands:
						continue

					raise ValueError(
						f"Unsupported band '{value}' for "
						f"{self.model_spec.name}. Supported bands: "
						+ ", ".join(
							self.model_spec.band_map.keys()
						)
					)

				band_index = (
					self.model_spec.band_map[
						key
					]
				)

			if (
				band_index < 0
				or band_index >= self.model_spec.n_model_bands
			):
				if self.allow_extra_bands:
					continue

				raise ValueError(
					f"Band index {band_index} is invalid for "
					f"{self.model_spec.name}; expected range "
					f"0..{self.model_spec.n_model_bands - 1}"
				)

			output[
				index
			] = band_index

		return output

	def _select_multiband_samples(
		self,
		series: TimeSeries,
	) -> tuple[
		np.ndarray,
		np.ndarray,
		np.ndarray | None,
		np.ndarray,
	]:
		"""Select valid observations for one multiband model invocation."""

		if series.n_variates != 1:
			raise ValueError(
				f"{self.model_spec.name} requires exactly one "
				f"value channel, got {series.n_variates}"
			)

		if series.times is None:
			raise ValueError(
				f"{self.model_spec.name} requires explicit timestamps"
			)

		if series.bands is None:
			raise ValueError(
				f"{self.model_spec.name} requires one band label "
				"per observation"
			)

		if (
			self.model_spec.requires_errors
			and series.errors is None
		):
			raise ValueError(
				f"{self.model_spec.name} requires measurement errors"
			)

		values = np.asarray(
			series.values[
				:,
				0,
			],
			dtype=np.float64,
		)

		times = np.asarray(
			series.times,
			dtype=np.float64,
		)

		bands = self._normalise_multiband_labels(
			np.asarray(
				series.bands
			)
		)

		selected = self.get_input_sample_mask(
			series
		)[
			:,
			0,
		]

		valid = (
			selected
			& np.isfinite(values)
			& np.isfinite(times)
			& (bands >= 0)
		)

		errors = None

		if series.errors is not None:
			errors = np.asarray(
				series.errors[
					:,
					0,
				],
				dtype=np.float64,
			)

			if self.model_spec.requires_errors:
				valid &= np.isfinite(
					errors
				)

		times = times[
			valid
		]

		values = values[
			valid
		]

		bands = bands[
			valid
		]

		if errors is not None:
			errors = errors[
				valid
			]

		if values.size < self.min_samples:
			raise ValueError(
				f"{self.model_spec.name} has only "
				f"{values.size} valid selected observations; "
				f"at least {self.min_samples} are required"
			)

		if (
			self.model_spec.name == "astra-clr"
			and errors is not None
			and np.any(
				errors <= 0
			)
		):
			raise ValueError(
				"AstraCLR requires strictly positive "
				"magnitude uncertainties"
			)

		order = np.argsort(
			times,
			kind="stable",
		)

		times = times[
			order
		]

		values = values[
			order
		]

		bands = bands[
			order
		]

		if errors is not None:
			errors = errors[
				order
			]

		return (
			times,
			values,
			errors,
			bands,
		)

	def _run_multiband_model(
		self,
		time: np.ndarray,
		values: np.ndarray,
		errors: np.ndarray | None,
		bands: np.ndarray,
	) -> np.ndarray:
		"""Run one LiCu multiband model on the full observation stream."""

		if self.model is None:
			raise RuntimeError(
				"LiCu embedding model is not loaded"
			)

		if self.model_spec.name == "astra-clr":
			if errors is None:
				raise RuntimeError(
					"AstraCLR measurement errors are missing"
				)

			embedding = self.model(
				time,
				values,
				errors,
				bands,
			)

		elif self.model_spec.name == "atat":
			embedding = self.model(
				time,
				values,
				bands,
			)

		elif self.model_spec.name == "atcat":
			if errors is None:
				raise RuntimeError(
					"ATCAT measurement errors are missing"
				)

			embedding = self.model(
				time,
				values,
				errors,
				bands,
			)

		else:
			raise RuntimeError(
				f"Unsupported LiCu multiband model "
				f"'{self.model_spec.name}'"
			)

		embedding = np.asarray(
			embedding,
			dtype=np.float32,
		)

		if embedding.ndim < 1:
			raise RuntimeError(
				"Unexpected LiCu embedding shape: "
				f"{embedding.shape}"
			)

		if (
			embedding.shape[-1]
			!= self.model_spec.embedding_dim
		):
			raise RuntimeError(
				"Unexpected LiCu embedding dimension for "
				f"{self.model_spec.name}: got {embedding.shape[-1]}, "
				f"expected {self.model_spec.embedding_dim}; "
				f"full shape={embedding.shape}"
			)

		return embedding
		
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

	def _extract_single_channel_models(
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

	def _extract_multiband_model(
		self,
		series: TimeSeries,
	) -> np.ndarray:
		"""Embed one complete multiband observation stream."""

		(
			time,
			values,
			errors,
			bands,
		) = self._select_multiband_samples(
			series
		)

		logger.info(
			"Extracting LiCu multiband embedding: "
			"model='%s' input_samples=%d valid_samples=%d bands=%s",
			self.model_spec.name,
			series.n_time,
			values.size,
			sorted(
				set(
					int(value)
					for value in bands
				)
			),
		)

		embedding = self._run_multiband_model(
			time,
			values,
			errors,
			bands,
		)

		features = self._aggregate_channel_embedding(
			embedding
		)

		self.channel_metadata = [
			{
				"name": "multiband",
				"n_input": int(
					series.n_time
				),
				"n_valid": int(
					values.size
				),
				"bands": sorted(
					set(
						int(value)
						for value in bands
					)
				),
				"native_embedding_shape": list(
					embedding.shape
				),
				"n_features": int(
					features.size
				),
			}
		]

		return features.astype(
			np.float32,
			copy=False,
		)


	def extract_timeseries(
		self,
		series: TimeSeries,
	) -> np.ndarray:
		"""Extract a LiCu single-channel or multiband representation."""

		if self.model_spec.multiband:
			return self._extract_multiband_model(
				series
			)

		return self._extract_single_channel_models(
			series
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
			"multiband": self.model_spec.multiband,
			"requires_errors": self.model_spec.requires_errors,
			"mag_zp": self.mag_zp,
			"allow_extra_bands": self.allow_extra_bands,
			"channel_mode": (
				"multiband"
				if self.model_spec.multiband
				else "independent"
			),
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
			None,
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
		mag_zp=config.get_option(
			"licu_mag_zp",
			None,
		),
		allow_extra_bands=config.get_option(
			"licu_allow_extra_bands",
			False,
		),
		preprocessing=preprocessing,
		input_sample_policy=config.get_option(
			"input_sample_policy",
			DEFAULT_INPUT_SAMPLE_POLICY,
		),
	)
