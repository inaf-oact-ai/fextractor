"""Hand-crafted time-series feature extraction using light-curve-python."""

from __future__ import annotations

import logging

import numpy as np

from ...config import ExtractorConfig
from ...preprocessing import TimeSeriesPreprocessConfig
from ...timeseries import (
	DEFAULT_INPUT_SAMPLE_POLICY,
	TimeSeries,
)
from .base import TimeSeriesFeatureExtractor


logger = logging.getLogger(__name__)


DEFAULT_FEATURE_SET = "default"
DEFAULT_INVALID_FEATURE_POLICY = "zero"
DEFAULT_MIN_SAMPLES = 5

SUPPORTED_FEATURE_SETS = (
	"basic",
	"default",
	"full",
)

SUPPORTED_INVALID_FEATURE_POLICIES = (
	"error",
	"zero",
)


class LiCuFeatureExtractor(TimeSeriesFeatureExtractor):
	"""Extract deterministic hand-crafted features with light-curve-python.

	Each canonical fextractor channel is treated as an independent univariate
	light curve. Features are extracted per channel and concatenated in channel
	order. Native astronomical multiband features from light-curve-python are
	intentionally not enabled here because fextractor channels are generic
	variates and must not be interpreted as photometric passbands implicitly.
	"""

	backend = "licu"
	modality = "timeseries"

	def __init__(
		self,
		feature_set: str = DEFAULT_FEATURE_SET,
		invalid_feature_policy: str = DEFAULT_INVALID_FEATURE_POLICY,
		min_samples: int = DEFAULT_MIN_SAMPLES,
		preprocessing: TimeSeriesPreprocessConfig | None = None,
		input_sample_policy: str = DEFAULT_INPUT_SAMPLE_POLICY,
	) -> None:
		super().__init__(
			preprocessing=preprocessing,
			input_sample_policy=input_sample_policy,
		)

		if feature_set not in SUPPORTED_FEATURE_SETS:
			raise ValueError(
				f"Unsupported LiCu feature_set '{feature_set}'. "
				f"Supported values: {', '.join(SUPPORTED_FEATURE_SETS)}"
			)

		if invalid_feature_policy not in SUPPORTED_INVALID_FEATURE_POLICIES:
			raise ValueError(
				"Unsupported LiCu invalid_feature_policy "
				f"'{invalid_feature_policy}'. Supported values: "
				f"{', '.join(SUPPORTED_INVALID_FEATURE_POLICIES)}"
			)

		if min_samples < 2:
			raise ValueError(
				"LiCu min_samples must be at least 2"
			)

		self.feature_set = feature_set
		self.invalid_feature_policy = invalid_feature_policy
		self.min_samples = int(min_samples)

		self.licu = None
		self.extractor = None
		self.library_version = None
		self.base_feature_names: tuple[str, ...] = ()
		self.feature_names: tuple[str, ...] = ()
		self.channel_metadata: list[dict] = []

	def _build_feature_objects(self):
		"""Build the configured deterministic feature collection."""

		if self.licu is None:
			raise RuntimeError(
				"light-curve-python is not loaded"
			)

		licu = self.licu

		basic = [
			licu.Amplitude(),
			licu.InterPercentileRange(quantile=0.1),
			licu.Mean(),
			licu.Median(),
			licu.MedianAbsoluteDeviation(),
			licu.Skew(),
			licu.StandardDeviation(),
			licu.Duration(),
			licu.ObservationCount(),
		]

		default_extra = [
			licu.AndersonDarlingNormal(),
			licu.BeyondNStd(nstd=1.0),
			licu.BiweightScale(),
			licu.Cusum(),
			licu.Eta(),
			licu.EtaE(),
			licu.Kurtosis(),
			licu.MaximumSlope(),
			licu.MaximumTimeInterval(),
			licu.MinimumTimeInterval(),
			licu.PercentAmplitude(),
			licu.TimeMean(),
			licu.TimeStandardDeviation(),
		]

		full_extra = [
			licu.LaflerKinmanStringLength(),
			licu.MeanVariance(),
			licu.MedianBufferRangePercentage(quantile=0.1),
			licu.PercentDifferenceMagnitudePercentile(quantile=0.1),
			licu.QnScale(),
		]

		if self.feature_set == "basic":
			return basic

		if self.feature_set == "default":
			return basic + default_extra

		if self.feature_set == "full":
			return basic + default_extra + full_extra

		raise RuntimeError(
			f"Unexpected LiCu feature_set '{self.feature_set}'"
		)

	def load_model(self) -> None:
		"""Import light-curve-python and construct the configured extractor."""

		try:
			import light_curve as licu

		except ImportError as exc:
			raise RuntimeError(
				"LiCu backend requested, but light-curve is not installed. "
				"Install fextractor[licu]."
			) from exc

		self.licu = licu
		self.library_version = getattr(
			licu,
			"__version__",
			None,
		)

		feature_objects = self._build_feature_objects()
		self.extractor = licu.Extractor(
			*feature_objects
		)

		self.base_feature_names = tuple(
			str(name)
			for name in self.extractor.names
		)

		logger.info(
			"LiCu extractor loaded: library_version='%s' "
			"feature_set='%s' features_per_channel=%d",
			self.library_version,
			self.feature_set,
			len(self.base_feature_names),
		)

	def _channel_name(
		self,
		series: TimeSeries,
		channel_index: int,
	) -> str:
		"""Return a stable logical name for one channel."""

		if series.channel_names is not None:
			return str(
				series.channel_names[
					channel_index
				]
			)

		return f"channel_{channel_index}"

	def _select_valid_samples(
		self,
		series: TimeSeries,
		channel_index: int,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
		"""Return finite samples selected by the configured input policy."""

		if series.times is None:
			raise ValueError(
				"LiCu requires explicit timestamps"
			)

		time = np.asarray(
			series.times,
			dtype=np.float64,
		)
		values = np.asarray(
			series.values[:, channel_index],
			dtype=np.float64,
		)

		
		input_mask = self.get_input_sample_mask(
			series
		)

		valid = (
			input_mask[:, channel_index]
			& np.isfinite(time)
		)		
		
		errors = None

		if series.errors is not None:
			errors_all = np.asarray(
				series.errors[:, channel_index],
				dtype=np.float64,
			)

			valid_errors = (
				np.isfinite(errors_all)
				& (errors_all > 0.0)
			)

			if np.all(
				valid_errors[valid]
			):
				errors = errors_all[valid]
			else:
				logger.warning(
					"LiCu channel '%s': ignoring uncertainties because "
					"one or more selected errors are non-finite or non-positive",
					self._channel_name(
						series,
						channel_index,
					),
				)

		time = time[valid]
		values = values[valid]

		if time.size < self.min_samples:
			raise ValueError(
				f"LiCu channel '{self._channel_name(series, channel_index)}' "
				f"has only {time.size} valid prepared samples; "
				f"at least {self.min_samples} are required"
			)

		order = np.argsort(
			time,
			kind="stable",
		)

		time = time[order]
		values = values[order]

		if errors is not None:
			errors = errors[order]

		return (
			time,
			values,
			errors,
		)

	def _handle_invalid_features(
		self,
		channel_name: str,
		features: np.ndarray,
	) -> tuple[np.ndarray, list[str]]:
		"""Apply the configured policy to non-finite feature values."""

		invalid_mask = ~np.isfinite(
			features
		)

		invalid_features = [
			name
			for name, invalid in zip(
				self.base_feature_names,
				invalid_mask,
			)
			if invalid
		]

		if not invalid_features:
			return (
				features,
				[],
			)

		if self.invalid_feature_policy == "error":
			raise ValueError(
				f"LiCu channel '{channel_name}' produced non-finite "
				"feature value(s): "
				+ ", ".join(
					invalid_features
				)
			)

		features = features.copy()
		features[invalid_mask] = 0.0

		logger.warning(
			"LiCu channel '%s': replacing non-finite feature value(s) "
			"with zero: %s",
			channel_name,
			", ".join(
				invalid_features
			),
		)

		return (
			features,
			invalid_features,
		)

	def extract_timeseries(
		self,
		series: TimeSeries,
	) -> np.ndarray:
		"""Extract and concatenate LiCu features for all channels."""

		if self.extractor is None:
			raise RuntimeError(
				"LiCu extractor is not loaded"
			)

		all_features: list[float] = []
		all_feature_names: list[str] = []
		channel_metadata: list[dict] = []

		for channel_index in range(
			series.n_variates
		):
			channel_name = self._channel_name(
				series,
				channel_index,
			)

			(
				time,
				values,
				errors,
			) = self._select_valid_samples(
				series,
				channel_index,
			)

			logger.info(
				"Extracting LiCu features: channel='%s' "
				"input_samples=%d valid_samples=%d errors=%s",
				channel_name,
				series.n_time,
				time.size,
				"yes" if errors is not None else "no",
			)

			features = np.asarray(
				self.extractor(
					time,
					values,
					errors,
					sorted=True,
					check=True,
				),
				dtype=np.float64,
			).reshape(-1)

			if features.size != len(
				self.base_feature_names
			):
				raise RuntimeError(
					"LiCu feature-name/value mismatch: "
					f"{len(self.base_feature_names)} names vs "
					f"{features.size} values"
				)

			(
				features,
				invalid_features,
			) = self._handle_invalid_features(
				channel_name,
				features,
			)

			channel_feature_names = [
				f"{channel_name}.{name}"
				for name in self.base_feature_names
			]

			all_features.extend(
				features.tolist()
			)
			all_feature_names.extend(
				channel_feature_names
			)

			channel_metadata.append({
				"name": channel_name,
				"n_input": int(
					series.n_time
				),
				"n_valid": int(
					time.size
				),
				"n_features": int(
					features.size
				),
				"n_observed": int(
					np.sum(
						series.observed_mask[:, channel_index]
					)
				),
				"n_interpolated": int(
					np.sum(
						series.interpolated_mask[:, channel_index]
					)
				),
				"n_predicted": int(
					np.sum(
						series.predicted_mask[:, channel_index]
					)
				),
				"errors_used": bool(
					errors is not None
				),
				"invalid_features": invalid_features,
			})

		self.feature_names = tuple(
			all_feature_names
		)
		self.channel_metadata = channel_metadata

		return np.asarray(
			all_features,
			dtype=np.float32,
		)

	def metadata(self) -> dict:
		"""Return LiCu configuration and feature-schema provenance."""

		metadata = super().metadata()

		metadata.update({
			"library": "light-curve",
			"library_version": self.library_version,
			"feature_set": self.feature_set,
			"invalid_feature_policy": self.invalid_feature_policy,
			"min_samples": self.min_samples,
			"features_per_channel": len(
				self.base_feature_names
			),
			"n_features": len(
				self.feature_names
			),
			"feature_names": list(
				self.feature_names
			),
			"channels": self.channel_metadata,
			"preprocessing": self.preprocessing.__dict__.copy(),
		})

		return metadata


def create(
	config: ExtractorConfig,
) -> LiCuFeatureExtractor:
	"""Create a LiCu extractor from ExtractorConfig."""

	preprocessing = config.preprocessing

	if not isinstance(
		preprocessing,
		TimeSeriesPreprocessConfig,
	):
		preprocessing = TimeSeriesPreprocessConfig()

	return LiCuFeatureExtractor(
		feature_set=config.get_option(
			"feature_set",
			DEFAULT_FEATURE_SET,
		),
		invalid_feature_policy=config.get_option(
			"invalid_feature_policy",
			DEFAULT_INVALID_FEATURE_POLICY,
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
