"""Falcon-1 time-series representation extractor."""

from __future__ import annotations

import logging

import numpy as np

from ...config import ExtractorConfig
from ...preprocessing import TimeSeriesPreprocessConfig
from ...timeseries import (
	DEFAULT_INPUT_SAMPLE_POLICY,
	TimeSeries,
	is_regular_timeseries,
)
from ...timeseries.aggregation import TokenRepresentation
from .base import TokenTimeSeriesFeatureExtractor


logger = logging.getLogger(__name__)


DEFAULT_MODEL = "ant-intl/Falcon-TST_Large"
DEFAULT_AGGREGATION = "mean_std"


class Falcon1FeatureExtractor(TokenTimeSeriesFeatureExtractor):
	"""Extract latent representations using Ant International Falcon-1."""

	backend = "falcon1"
	modality = "timeseries"

	def __init__(
		self,
		model_name: str = DEFAULT_MODEL,
		device: str = "cuda",
		aggregation: str = DEFAULT_AGGREGATION,
		context_length: int | None = None,
		preprocessing: TimeSeriesPreprocessConfig | None = None,
		input_sample_policy: str = DEFAULT_INPUT_SAMPLE_POLICY,
	) -> None:
		super().__init__(
			aggregation=aggregation,
			preprocessing=preprocessing,
			input_sample_policy=input_sample_policy,
		)

		if context_length is not None and context_length <= 0:
			raise ValueError("context_length must be greater than 0")

		self.model_name = model_name
		self.requested_device = device
		self.device = device
		self.context_length = context_length
		self.model = None
		self.torch = None

	def load_model(self) -> None:
		"""Load Falcon-1 through the official Hugging Face custom model code."""

		try:
			import torch
			from transformers import AutoModel
		except ImportError as exc:
			raise RuntimeError(
				"Falcon-1 backend requested, but its dependencies are not installed. "
				"Install fextractor[falcon]."
			) from exc

		self.torch = torch

		if "cuda" in self.device and not torch.cuda.is_available():
			logger.warning("CUDA requested for Falcon-1 but no CUDA device is available; falling back to CPU")
			self.device = "cpu"

		logger.info(
			"Loading Falcon-1 model='%s' requested_device='%s' device='%s'",
			self.model_name,
			self.requested_device,
			self.device,
		)

		self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
		self.model = self.model.to(self.device)
		self.model.eval()

		self._get_core_model()
		logger.info("Falcon-1 model loaded successfully: model='%s' device='%s'", self.model_name, self.device)

	def _get_core_model(self):
		"""Return the FalconTSTModel object containing decoder/MoE internals."""

		if self.model is None:
			raise RuntimeError("Falcon-1 model is not loaded")

		core_model = getattr(self.model, "model", None)
		if core_model is None or not hasattr(core_model, "decoder"):
			raise RuntimeError(
				"Unsupported Falcon-1 model structure: expected a top-level '.model.decoder' hierarchy"
			)

		return core_model

	def _validate_sampling(self, series: TimeSeries) -> None:
		"""Reject irregular sampling unless preprocessing regularized it."""

		if series.times is None:
			return

		if not is_regular_timeseries(series.times):
			raise ValueError(
				"Falcon-1 requires regularly sampled input. The supplied timestamps are irregular. "
				"Enable time-series regularization."
			)

	def _to_falcon_input(self, series: TimeSeries) -> tuple[np.ndarray, np.ndarray]:
		"""Convert canonical [T,C] values and sample mask to Falcon [C,T] arrays."""

		self._validate_sampling(series)
		values = np.asarray(series.values, dtype=np.float32).copy()
		input_mask = self.get_input_sample_mask(series).copy()

		if self.context_length is not None:
			values = values[-self.context_length:, :]
			input_mask = input_mask[-self.context_length:, :]

		if np.any(input_mask.sum(axis=0) == 0):
			raise ValueError("Falcon-1 requires at least one selected finite sample per channel")

		return values.T, input_mask.T

	def _capture_shared_expert_tokens(self, input_tensor) -> np.ndarray:
		"""Run one Falcon inference step and capture shared-expert hidden states."""

		core_model = self._get_core_model()
		torch = self.torch

		if torch is None:
			raise RuntimeError("Falcon-1 torch runtime is not initialized")

		config = core_model.config
		mask_pad_value = float(config.mask_pad_value)
		seq_length = int(core_model.seq_length)

		if input_tensor.shape[1] > seq_length:
			input_tensor = input_tensor[:, -seq_length:]
		elif input_tensor.shape[1] < seq_length:
			pad_len = seq_length - input_tensor.shape[1]
			input_tensor = torch.nn.functional.pad(input_tensor, pad=(pad_len, 0), mode="constant", value=mask_pad_value)

		input_mask = input_tensor != mask_pad_value
		if torch.any(input_mask.sum(dim=1) == 0):
			raise ValueError("Falcon-1 received a channel with no usable samples after padding/masking")

		input_tensor, _, _ = core_model.revin(input_tensor, input_mask)
		rotary_pos_emb = core_model.rotary_pos_emb(seq_length, device=input_tensor.device)

		captured = []
		handles = []

		def capture_hidden_states(_module, _inputs, output):
			captured.append(output.detach())

		for layer in core_model.decoder.layers:
			handles.append(layer.shared_experts.final_layernorm.register_forward_hook(capture_hidden_states))

		try:
			with torch.no_grad():
				core_model._inference_step(
					input=input_tensor,
					input_mask=input_mask,
					rotary_pos_emb=rotary_pos_emb,
				)
		finally:
			for handle in handles:
				handle.remove()

		if len(captured) != len(core_model.decoder.layers):
			raise RuntimeError(
				"Unexpected Falcon-1 hidden-state capture count: "
				f"expected {len(core_model.decoder.layers)}, received {len(captured)}"
			)

		# Each shared expert returns [patch, channel, hidden]. All shared experts
		# use the same patch size, so tokens from successive MoE layers can be
		# concatenated along the token axis into [channel, token, hidden].
		layer_tokens = [hidden.transpose(0, 1).contiguous() for hidden in captured]
		tokens = torch.cat(layer_tokens, dim=1).float().cpu().numpy()
		return np.asarray(tokens, dtype=np.float32)

	def extract_tokens(self, series: TimeSeries) -> TokenRepresentation:
		"""Return Falcon shared-expert contextual token representations."""

		if self.model is None or self.torch is None:
			raise RuntimeError("Falcon-1 model is not loaded")

		values, sample_mask = self._to_falcon_input(series)
		core_model = self._get_core_model()
		mask_pad_value = float(core_model.config.mask_pad_value)

		if np.any(values[sample_mask] == mask_pad_value):
			raise ValueError(
				f"Falcon-1 uses {mask_pad_value:g} as its native mask sentinel, but that value is present in selected input samples. "
				"Apply a value transform or scaling step before extraction."
			)

		values[~sample_mask] = mask_pad_value
		parameter = next(self.model.parameters())
		input_tensor = self.torch.as_tensor(values, dtype=parameter.dtype, device=self.device)
		tokens = self._capture_shared_expert_tokens(input_tensor)

		return TokenRepresentation(context_tokens=tokens)

	def metadata(self) -> dict:
		"""Return Falcon model and preprocessing provenance."""

		metadata = super().metadata()
		metadata.update({
			"model": self.model_name,
			"requested_device": self.requested_device,
			"device": self.device,
			"aggregation": self.aggregation,
			"context_length": self.context_length,
			"channel_mode": "independent",
			"representation_scope": "shared_experts_all_layers",
			"model_normalization": "revin",
			"preprocessing": self.preprocessing.__dict__.copy(),
		})

		if self.model is not None:
			core_model = self._get_core_model()
			metadata["native_context_length"] = int(core_model.seq_length)
			metadata["hidden_size"] = int(core_model.config.hidden_size)
			metadata["num_hidden_layers"] = int(core_model.config.num_hidden_layers)
			metadata["shared_patch_size"] = int(core_model.config.shared_patch_size)

		return metadata


def create(config: ExtractorConfig) -> Falcon1FeatureExtractor:
	"""Create a Falcon-1 extractor from ExtractorConfig."""

	preprocessing = config.preprocessing
	if not isinstance(preprocessing, TimeSeriesPreprocessConfig):
		preprocessing = TimeSeriesPreprocessConfig()

	return Falcon1FeatureExtractor(
		model_name=config.model or DEFAULT_MODEL,
		device=config.device,
		aggregation=config.get_option("aggregation", DEFAULT_AGGREGATION),
		context_length=config.get_option("context_length", None),
		preprocessing=preprocessing,
		input_sample_policy=config.get_option("input_sample_policy", DEFAULT_INPUT_SAMPLE_POLICY),
	)
