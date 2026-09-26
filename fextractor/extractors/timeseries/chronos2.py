"""Chronos-2 time-series representation extractor."""

from __future__ import annotations

import logging

import numpy as np

from ...config import ExtractorConfig
from ...preprocessing import TimeSeriesPreprocessConfig
from ...timeseries import (
	TimeSeries,
	is_regular_timeseries,
)
from ...timeseries.aggregation import TokenRepresentation
from .base import TokenTimeSeriesFeatureExtractor


logger = logging.getLogger(__name__)


DEFAULT_MODEL = "amazon/chronos-2"
DEFAULT_AGGREGATION = "mean_std"


class Chronos2FeatureExtractor(
	TokenTimeSeriesFeatureExtractor
):
	"""Extract representations using Amazon Chronos-2."""

	backend = "chronos2"
	modality = "timeseries"

	def __init__(
		self,
		model_name: str = DEFAULT_MODEL,
		device: str = "cuda",
		aggregation: str = DEFAULT_AGGREGATION,
		context_length: int | None = None,
		batch_size: int = 256,
		preprocessing: TimeSeriesPreprocessConfig | None = None,
	) -> None:
		super().__init__(
			aggregation=aggregation,
			preprocessing=preprocessing,
		)

		self.model_name = model_name
		self.requested_device = device
		self.device = device
		self.context_length = context_length
		self.batch_size = batch_size

		self.pipeline = None
		self.torch = None

	def load_model(self) -> None:
		"""Load Chronos-2 using the official chronos package."""

		try:
			import torch
			from chronos import Chronos2Pipeline

		except ImportError as exc:
			raise RuntimeError(
				"Chronos-2 backend requested, but "
				"chronos-forecasting is not installed. "
				"Install fextractor[chronos]."
			) from exc

		self.torch = torch

		if (
			"cuda" in self.device
			and not torch.cuda.is_available()
		):
			logger.warning(
				"CUDA requested for Chronos-2 but no CUDA "
				"device is available; falling back to CPU"
			)

			self.device = "cpu"

		logger.info(
			"Loading Chronos-2 model='%s' requested_device='%s' "
			"device='%s'",
			self.model_name,
			self.requested_device,
			self.device,
		)

		self.pipeline = Chronos2Pipeline.from_pretrained(
			self.model_name,
			device_map=self.device,
			torch_dtype=torch.float32,
		)

		logger.info(
			"Chronos-2 model loaded successfully: "
			"model='%s' device='%s'",
			self.model_name,
			self.device,
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
				"Chronos-2 requires regularly sampled input. "
				"The supplied timestamps are irregular. "
				"Enable time-series regularization."
			)

	def _to_chronos_input(
		self,
		series: TimeSeries,
	) -> np.ndarray:
		"""Convert canonical [T,C] input into Chronos representation."""

		self._validate_sampling(
			series
		)

		values = np.asarray(
			series.values,
			dtype=np.float32,
		).copy()

		values[
			~series.observed_mask
		] = np.nan

		if series.is_univariate:
			return values[:, 0]

		return values.T

	def extract_tokens(
		self,
		series: TimeSeries,
	) -> TokenRepresentation:
		"""Return contextual Chronos-2 patch representations."""

		if self.pipeline is None:
			raise RuntimeError(
				"Chronos-2 model is not loaded"
			)

		input_values = self._to_chronos_input(
			series
		)

		embeddings, _ = self.pipeline.embed(
			[
				input_values,
			],
			batch_size=self.batch_size,
			context_length=self.context_length,
		)

		if len(embeddings) != 1:
			raise RuntimeError(
				"Expected one Chronos embedding result, "
				f"received {len(embeddings)}"
			)

		features = embeddings[0]

		if hasattr(
			features,
			"detach",
		):
			features = (
				features
				.detach()
				.cpu()
				.numpy()
			)

		features = np.asarray(
			features,
			dtype=np.float32,
		)

		if (
			features.ndim != 3
			or features.shape[1] < 3
		):
			raise RuntimeError(
				"Unexpected Chronos-2 embedding shape: "
				f"{features.shape}"
			)

		# Chronos-2 returns:
		#
		# [context patches, REG token, masked output patch]
		#
		# shape:
		# [n_variates, num_patches + 2, d_model]

		context_tokens = features[
			:,
			:-2,
			:,
		]

		reg_tokens = features[
			:,
			-2:-1,
			:,
		]

		future_tokens = features[
			:,
			-1:,
			:,
		]

		return TokenRepresentation(
			context_tokens=context_tokens,
			special_tokens={
				"reg": reg_tokens,
				"future": future_tokens,
			},
		)

	def metadata(self) -> dict:
		"""Return Chronos model and preprocessing provenance."""

		metadata = super().metadata()

		metadata.update({
			"model": self.model_name,
			"requested_device": self.requested_device,
			"device": self.device,
			"aggregation": self.aggregation,
			"context_length": self.context_length,
			"batch_size": self.batch_size,
			"preprocessing": self.preprocessing.__dict__.copy(),
		})

		return metadata


def create(
	config: ExtractorConfig,
) -> Chronos2FeatureExtractor:
	"""Create a Chronos-2 extractor from ExtractorConfig."""

	preprocessing = config.preprocessing

	if not isinstance(
		preprocessing,
		TimeSeriesPreprocessConfig,
	):
		preprocessing = TimeSeriesPreprocessConfig()

	return Chronos2FeatureExtractor(
		model_name=(
			config.model
			or DEFAULT_MODEL
		),
		device=config.device,
		aggregation=config.get_option(
			"aggregation",
			DEFAULT_AGGREGATION,
		),
		context_length=config.get_option(
			"context_length",
			None,
		),
		batch_size=config.get_option(
			"batch_size",
			256,
		),
		preprocessing=preprocessing,
	)
