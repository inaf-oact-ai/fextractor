"""Moirai-2 time-series representation extractor."""

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


DEFAULT_MODEL = "Salesforce/moirai-2.0-R-small"
DEFAULT_AGGREGATION = "mean_std"
DEFAULT_PATCHING_MODE = "time_variate"
DEFAULT_TOKEN_ORDER = "by_variate"


class Moirai2FeatureExtractor(
	TokenTimeSeriesFeatureExtractor
):
	"""Extract contextual representations using Salesforce Moirai-2."""

	backend = "moirai2"
	modality = "timeseries"

	def __init__(
		self,
		model_name: str = DEFAULT_MODEL,
		device: str = "cuda",
		aggregation: str = DEFAULT_AGGREGATION,
		patching_mode: str = DEFAULT_PATCHING_MODE,
		token_order: str = DEFAULT_TOKEN_ORDER,
		preprocessing: TimeSeriesPreprocessConfig | None = None,
		input_sample_policy: str = DEFAULT_INPUT_SAMPLE_POLICY,
	) -> None:
		super().__init__(
			aggregation=aggregation,
			preprocessing=preprocessing,
			input_sample_policy=input_sample_policy,
		)

		if patching_mode not in (
			"time_only",
			"time_variate",
		):
			raise ValueError(
				f"Unsupported Moirai-2 patching mode "
				f"'{patching_mode}'"
			)

		if token_order not in (
			"by_variate",
			"interleave_time",
		):
			raise ValueError(
				f"Unsupported Moirai-2 token order "
				f"'{token_order}'"
			)

		self.model_name = model_name
		self.requested_device = device
		self.device = device

		self.patching_mode = patching_mode
		self.token_order = token_order

		self.model = None
		self.torch = None

		self._last_repr = None
		self._hooked = False

	def load_model(self) -> None:
		"""Load Moirai-2 using uni2ts."""

		try:
			import torch
			from uni2ts.model.moirai2 import Moirai2Module

		except ImportError as exc:
			raise RuntimeError(
				"Moirai-2 backend requested, but uni2ts "
				"is not installed. Install fextractor[moirai]."
			) from exc

		self.torch = torch

		if (
			"cuda" in self.device
			and not torch.cuda.is_available()
		):
			logger.warning(
				"CUDA requested for Moirai-2 but no CUDA "
				"device is available; falling back to CPU"
			)

			self.device = "cpu"

		logger.info(
			"Loading Moirai-2 model='%s' requested_device='%s' "
			"device='%s'",
			self.model_name,
			self.requested_device,
			self.device,
		)

		self.model = Moirai2Module.from_pretrained(
			self.model_name
		)

		self.model = self.model.to(
			self.device
		)

		self.model.eval()

		self._hooked = self._register_representation_hook()

		logger.info(
			"Moirai-2 model loaded successfully: "
			"model='%s' device='%s' patch_size=%s "
			"patching_mode='%s' token_order='%s'",
			self.model_name,
			self.device,
			getattr(
				self.model,
				"patch_size",
				None,
			),
			self.patching_mode,
			self.token_order,
		)

	def _register_representation_hook(
		self,
	) -> bool:
		"""Try to capture the internal contextual representation."""

		for name in (
			"transformer",
			"encoder",
			"backbone",
			"model",
			"net",
		):
			module = getattr(
				self.model,
				name,
				None,
			)

			if module is None:
				continue

			try:
				def hook(
					module,
					inputs,
					output,
				):
					value = None

					if (
						isinstance(
							output,
							(list, tuple),
						)
						and len(output) > 0
					):
						value = output[0]

					elif isinstance(
						output,
						dict,
					):
						for key in (
							"reprs",
							"hidden_states",
							"x",
							"last_hidden_state",
						):
							candidate = output.get(
								key
							)

							if self.torch.is_tensor(
								candidate
							):
								value = candidate
								break

					else:
						value = output

					if self.torch.is_tensor(
						value
					):
						self._last_repr = value

				module.register_forward_hook(
					hook
				)

				logger.debug(
					"Registered Moirai-2 representation "
					"hook on '%s'",
					name,
				)

				return True

			except Exception:
				continue

		logger.warning(
			"Could not register a Moirai-2 representation "
			"hook; direct model outputs will be inspected"
		)

		return False

	def _get_representations(
		self,
		output,
	):
		"""Retrieve contextual representations from model output."""

		if self.torch.is_tensor(
			self._last_repr
		):
			return self._last_repr

		if isinstance(
			output,
			dict,
		):
			for key in (
				"reprs",
				"hidden_states",
				"x",
				"last_hidden_state",
			):
				value = output.get(
					key
				)

				if self.torch.is_tensor(
					value
				):
					return value

		if (
			isinstance(
				output,
				(list, tuple),
			)
			and len(output) > 0
			and self.torch.is_tensor(
				output[0]
			)
		):
			return output[0]

		if self.torch.is_tensor(
			output
		):
			return output

		raise RuntimeError(
			"Could not retrieve Moirai-2 contextual "
			"representations"
		)

	def _validate_sampling(
		self,
		series: TimeSeries,
	) -> None:
		"""Reject irregular sampling unless preprocessing regularized it."""

		if series.times is None:
			return

		if not is_regular_timeseries(
			series.times
		):
			raise ValueError(
				"Moirai-2 requires regularly sampled input "
				"because the backend operates on ordered "
				"sample positions rather than arbitrary "
				"timestamps. Enable time-series regularization."
			)

	def _pack_for_moirai(
		self,
		values,
		observed_mask,
	) -> dict:
		"""Pack raw [B,T,C] data into Moirai-2 token inputs."""

		torch = self.torch

		if values.ndim != 3:
			raise ValueError(
				"Moirai-2 input must have shape [B,T,C], "
				f"got {tuple(values.shape)}"
			)

		batch_size, length, n_channels = (
			values.shape
		)

		device = values.device
		dtype = values.dtype

		if observed_mask.ndim == 2:
			observed_mask = (
				observed_mask
				.unsqueeze(-1)
				.expand(
					batch_size,
					length,
					n_channels,
				)
			)

		if observed_mask.ndim != 3:
			raise ValueError(
				"Moirai-2 observed mask must have shape "
				"[B,T] or [B,T,C]"
			)

		observed_mask = observed_mask.to(
			device=device,
			dtype=torch.bool,
		)

		patch_size = int(
			getattr(
				self.model,
				"patch_size",
				16,
			)
		)

		in_proj = getattr(
			self.model,
			"in_proj",
			None,
		)

		if (
			in_proj is None
			or not hasattr(
				in_proj,
				"hidden_layer",
			)
		):
			raise RuntimeError(
				"Moirai-2 model does not expose "
				"in_proj.hidden_layer; cannot infer "
				"token input width"
			)

		expected_width = int(
			in_proj.hidden_layer.in_features
		)

		# Moirai token embedding concatenates values + mask.
		concat_factor = 2

		denominator = (
			concat_factor
			* patch_size
		)

		if (
			expected_width
			% denominator
			!= 0
		):
			raise RuntimeError(
				"Cannot infer Moirai-2 channel grouping: "
				f"input_width={expected_width}, "
				f"patch_size={patch_size}"
			)

		channels_per_token = (
			expected_width
			// denominator
		)

		if channels_per_token < 1:
			raise RuntimeError(
				"Invalid inferred Moirai-2 channels per token: "
				f"{channels_per_token}"
			)

		# Trim to a whole number of temporal patches.
		patched_length = (
			length
			// patch_size
		) * patch_size

		if patched_length <= 0:
			raise ValueError(
				"Time series is shorter than the Moirai-2 "
				f"patch size ({patch_size})"
			)

		if patched_length != length:
			logger.debug(
				"Trimming Moirai-2 input length from %d to %d "
				"to match patch_size=%d",
				length,
				patched_length,
				patch_size,
			)

			values = values[
				:,
				:patched_length,
				:,
			]

			observed_mask = observed_mask[
				:,
				:patched_length,
				:,
			]

			length = patched_length

		n_time_tokens = (
			length
			// patch_size
		)

		if self.patching_mode == "time_only":
			if n_channels < channels_per_token:
				n_pad = (
					channels_per_token
					- n_channels
				)

				padding = torch.zeros(
					batch_size,
					length,
					n_pad,
					dtype=dtype,
					device=device,
				)

				values = torch.cat(
					[
						values,
						padding,
					],
					dim=-1,
				)

				observed_mask = torch.cat(
					[
						observed_mask,
						torch.zeros_like(
							padding,
							dtype=torch.bool,
						),
					],
					dim=-1,
				)

				n_channels = channels_per_token

			elif n_channels > channels_per_token:
				logger.warning(
					"Moirai-2 time_only patching supports "
					"%d channel(s) for this model; truncating "
					"input from %d channel(s)",
					channels_per_token,
					n_channels,
				)

				values = values[
					...,
					:channels_per_token,
				]

				observed_mask = observed_mask[
					...,
					:channels_per_token,
				]

				n_channels = channels_per_token

			n_tokens = n_time_tokens

			target = (
				values
				.contiguous()
				.view(
					batch_size,
					n_tokens,
					patch_size
					* channels_per_token,
				)
			)

			token_observed = (
				observed_mask
				.contiguous()
				.view(
					batch_size,
					n_tokens,
					patch_size
					* channels_per_token,
				)
			)

			time_id = (
				torch.arange(
					n_tokens,
					device=device,
				)
				.view(1, -1)
				.expand(
					batch_size,
					n_tokens,
				)
			)

			sample_id = (
				torch.arange(
					batch_size,
					device=device,
				)
				.view(-1, 1)
				.expand(
					batch_size,
					n_tokens,
				)
			)

			variate_id = torch.zeros(
				batch_size,
				n_tokens,
				dtype=torch.long,
				device=device,
			)

		else:
			# time_variate:
			# keep channel groups as distinct token variates.
			remainder = (
				n_channels
				% channels_per_token
			)

			if remainder:
				padded_channels = (
					(
						n_channels
						+ channels_per_token
						- 1
					)
					// channels_per_token
				) * channels_per_token

				n_pad = (
					padded_channels
					- n_channels
				)

				padding = torch.zeros(
					batch_size,
					length,
					n_pad,
					dtype=dtype,
					device=device,
				)

				values = torch.cat(
					[
						values,
						padding,
					],
					dim=-1,
				)

				observed_mask = torch.cat(
					[
						observed_mask,
						torch.zeros_like(
							padding,
							dtype=torch.bool,
						),
					],
					dim=-1,
				)

				n_channels = padded_channels

			n_variates = (
				n_channels
				// channels_per_token
			)

			values_5d = (
				values
				.contiguous()
				.view(
					batch_size,
					n_time_tokens,
					patch_size,
					n_variates,
					channels_per_token,
				)
			)

			mask_5d = (
				observed_mask
				.contiguous()
				.view(
					batch_size,
					n_time_tokens,
					patch_size,
					n_variates,
					channels_per_token,
				)
			)

			n_tokens = (
				n_variates
				* n_time_tokens
			)

			if self.token_order == "by_variate":
				values_block = (
					values_5d
					.permute(
						0,
						3,
						1,
						2,
						4,
					)
					.contiguous()
				)

				mask_block = (
					mask_5d
					.permute(
						0,
						3,
						1,
						2,
						4,
					)
					.contiguous()
				)

				time_grid = (
					torch.arange(
						n_time_tokens,
						device=device,
					)
					.view(
						1,
						1,
						n_time_tokens,
					)
					.expand(
						batch_size,
						n_variates,
						n_time_tokens,
					)
				)

				variate_grid = (
					torch.arange(
						n_variates,
						device=device,
					)
					.view(
						1,
						n_variates,
						1,
					)
					.expand(
						batch_size,
						n_variates,
						n_time_tokens,
					)
				)

			else:
				values_block = (
					values_5d
					.permute(
						0,
						1,
						3,
						2,
						4,
					)
					.contiguous()
				)

				mask_block = (
					mask_5d
					.permute(
						0,
						1,
						3,
						2,
						4,
					)
					.contiguous()
				)

				time_grid = (
					torch.arange(
						n_time_tokens,
						device=device,
					)
					.view(
						1,
						n_time_tokens,
						1,
					)
					.expand(
						batch_size,
						n_time_tokens,
						n_variates,
					)
				)

				variate_grid = (
					torch.arange(
						n_variates,
						device=device,
					)
					.view(
						1,
						1,
						n_variates,
					)
					.expand(
						batch_size,
						n_time_tokens,
						n_variates,
					)
				)

			target = values_block.view(
				batch_size,
				n_tokens,
				patch_size
				* channels_per_token,
			)

			token_observed = mask_block.view(
				batch_size,
				n_tokens,
				patch_size
				* channels_per_token,
			)

			time_id = time_grid.reshape(
				batch_size,
				n_tokens,
			)

			variate_id = (
				variate_grid
				.reshape(
					batch_size,
					n_tokens,
				)
				.long()
			)

			sample_id = (
				torch.arange(
					batch_size,
					device=device,
				)
				.view(-1, 1)
				.expand(
					batch_size,
					n_tokens,
				)
			)

		prediction_mask = torch.zeros(
			batch_size,
			n_tokens,
			dtype=torch.bool,
			device=device,
		)

		return {
			"target": target,
			"observed_mask": token_observed,
			"sample_id": sample_id,
			"time_id": time_id,
			"variate_id": variate_id,
			"prediction_mask": prediction_mask,
		}

	def extract_tokens(
		self,
		series: TimeSeries,
	) -> TokenRepresentation:
		"""Return contextual Moirai-2 token representations."""

		if self.model is None:
			raise RuntimeError(
				"Moirai-2 model is not loaded"
			)

		self._validate_sampling(
			series
		)

		values = np.asarray(
			series.values,
			dtype=np.float32,
		)
		
		input_mask = self.get_input_sample_mask(
			series
		)

		# Moirai receives finite placeholder values together
		# with the explicit observation mask.
		values = np.where(
			input_mask,
			values,
			0.0,
		).astype(
			np.float32,
			copy=False,
		)

		target = self.torch.as_tensor(
			values,
			dtype=self.torch.float32,
			device=self.device,
		).unsqueeze(0)

		mask = self.torch.as_tensor(
			input_mask,
			dtype=self.torch.bool,
			device=self.device,
		).unsqueeze(0)

		packed = self._pack_for_moirai(
			target,
			mask,
		)

		# Avoid accidentally reusing a representation
		# captured during an earlier forward pass.
		self._last_repr = None

		with self.torch.inference_mode():
			output = self.model(
				packed["target"],
				packed["observed_mask"],
				packed["sample_id"],
				packed["time_id"],
				packed["variate_id"],
				packed["prediction_mask"],
				True,
			)

		representations = self._get_representations(
			output
		)

		if representations.ndim == 2:
			batch_size, n_tokens = (
				packed["sample_id"].shape
			)

			representations = representations.view(
				batch_size,
				n_tokens,
				-1,
			)

		if representations.ndim != 3:
			raise RuntimeError(
				"Unexpected Moirai-2 representation shape: "
				f"{tuple(representations.shape)}"
			)

		if representations.shape[:2] != (
			packed["sample_id"].shape
		):
			raise RuntimeError(
				"Moirai-2 representation/token shape mismatch: "
				f"representations={tuple(representations.shape)}, "
				f"tokens={tuple(packed['sample_id'].shape)}"
			)

		token_observed = packed[
			"observed_mask"
		]

		valid_observed = (
			token_observed.any(
				dim=-1
			)
			if token_observed.ndim == 3
			else token_observed
		)

		valid = (
			valid_observed
			& ~packed[
				"prediction_mask"
			]
		)

		# fextractor processes one TimeSeries at a time,
		# therefore batch dimension is exactly one.
		context_tokens = representations[
			0,
			valid[0],
			:,
		]

		if context_tokens.shape[0] == 0:
			raise ValueError(
				"Moirai-2 produced no valid contextual tokens"
			)

		context_tokens = (
			context_tokens
			.detach()
			.cpu()
			.numpy()
			.astype(
				np.float32,
				copy=False,
			)
		)

		return TokenRepresentation(
			context_tokens=context_tokens
		)

	def metadata(self) -> dict:
		"""Return Moirai-2 model and preprocessing provenance."""

		metadata = super().metadata()

		metadata.update({
			"model": self.model_name,
			"requested_device": self.requested_device,
			"device": self.device,
			"aggregation": self.aggregation,
			"patching_mode": self.patching_mode,
			"token_order": self.token_order,
			"patch_size": (
				getattr(
					self.model,
					"patch_size",
					None,
				)
				if self.model is not None
				else None
			),
			"preprocessing": (
				self.preprocessing
				.__dict__
				.copy()
			),
		})

		return metadata


def create(
	config: ExtractorConfig,
) -> Moirai2FeatureExtractor:
	"""Create a Moirai-2 extractor from ExtractorConfig."""

	preprocessing = config.preprocessing

	if not isinstance(
		preprocessing,
		TimeSeriesPreprocessConfig,
	):
		preprocessing = TimeSeriesPreprocessConfig()

	return Moirai2FeatureExtractor(
		model_name=(
			config.model
			or DEFAULT_MODEL
		),
		device=config.device,
		aggregation=config.get_option(
			"aggregation",
			DEFAULT_AGGREGATION,
		),
		patching_mode=config.get_option(
			"patching_mode",
			DEFAULT_PATCHING_MODE,
		),
		token_order=config.get_option(
			"token_order",
			DEFAULT_TOKEN_ORDER,
		),
		preprocessing=preprocessing,
		input_sample_policy=config.get_option(
			"input_sample_policy",
			DEFAULT_INPUT_SAMPLE_POLICY,
		),
	)
