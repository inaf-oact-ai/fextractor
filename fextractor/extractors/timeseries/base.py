"""Base classes for time-series feature extractors."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import numpy as np

from ...base import FeatureExtractor
from ...preprocessing import (
	TimeSeriesPreprocessConfig,
	apply_timeseries_preprocessing,
)
from ...timeseries import (
	TimeSeries,
	read_timeseries,
)
from ...timeseries.data import TimeSeries

from ...timeseries.aggregation import (
	TokenRepresentation,
	aggregate_token_representation,
)


class TimeSeriesFeatureExtractor(FeatureExtractor):
	"""Base class for time-series feature extractors."""

	modality = "timeseries"

	def __init__(
		self,
		preprocessing: TimeSeriesPreprocessConfig | None = None,
	) -> None:
		super().__init__()

		self.preprocessing = (
			preprocessing
			or TimeSeriesPreprocessConfig()
		)

	def prepare(
		self,
		source: str | Path,
	) -> TimeSeries:
		"""Read and domain-preprocess one time series."""


		if isinstance(source, TimeSeries):
			series = source.copy()
			
		else:
			series = read_timeseries(
				source,
				time_column=self.preprocessing.time_column,
				value_columns=self.preprocessing.value_columns,
				error_columns=self.preprocessing.error_columns,
				layout=self.preprocessing.layout,
				value_prefixes=self.preprocessing.value_prefixes,
				channel_names=self.preprocessing.channel_names,
				error_prefixes=self.preprocessing.error_prefixes,
				time_prefix=self.preprocessing.time_prefix,
				time_start_column=self.preprocessing.time_start_column,
				cadence_column=self.preprocessing.cadence_column,
				label_column=self.preprocessing.label_column,
				metadata_columns=self.preprocessing.metadata_columns,
			)

		return apply_timeseries_preprocessing(
			series,
			self.preprocessing,
		)

	def extract(
		self,
		source: str | Path,
	) -> np.ndarray:
		"""Extract one one-dimensional representation."""

		self.ensure_loaded()

		series = self.prepare(
			source
		)

		features = self.extract_timeseries(
			series
		)

		return np.asarray(
			features,
			dtype=np.float32,
		).reshape(-1)

	@abstractmethod
	def extract_timeseries(
		self,
		series: TimeSeries,
	) -> np.ndarray:
		"""Extract features from an already prepared time series."""
		
class TokenTimeSeriesFeatureExtractor(
	TimeSeriesFeatureExtractor
):
	"""Base class for models exposing token-level representations."""

	def __init__(
		self,
		aggregation: str = "mean_std",
		preprocessing: TimeSeriesPreprocessConfig | None = None,
	) -> None:
		super().__init__(
			preprocessing=preprocessing,
		)

		self.aggregation = aggregation

	def extract_timeseries(
		self,
		series: TimeSeries,
	) -> np.ndarray:
		representation = self.extract_tokens(
			series
		)

		return aggregate_token_representation(
			representation,
			strategy=self.aggregation,
		)

	@abstractmethod
	def extract_tokens(
		self,
		series: TimeSeries,
	) -> TokenRepresentation:
		"""Return model-native contextual token representations."""
