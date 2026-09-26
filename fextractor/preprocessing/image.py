"""Scientific image domain preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval


@dataclass(frozen=True)
class ImagePreprocessConfig:
	"""Domain-level image preprocessing options.

	Model-specific resize, channel handling, and model-native normalization are
	handled by each extractor backend. This configuration contains scientific
	input transformations applied before model-specific preprocessing.
	"""

	clip_data: bool = False
	sigma_low: float = 5.0
	sigma_up: float = 30.0
	zscale: bool = False
	zscale_contrast: float = 0.25
	norm_min: float = 0.0
	norm_max: float = 1.0
	set_zero_to_min: bool = False


@dataclass(frozen=True)
class ImagePreprocessProfile:
	"""Named combination of preprocessing and model-input defaults."""
	name: str
	preprocessing: ImagePreprocessConfig
	imgsize: int | None = None
	in_chans: int | None = None


_PROFILES = {
	"simclr_radio": ImagePreprocessProfile(
		name="simclr_radio",
		preprocessing=ImagePreprocessConfig(
			clip_data=False,
			zscale=True,
			zscale_contrast=0.25,
			norm_min=0.0,
			norm_max=1.0,
			set_zero_to_min=False,
		),
		imgsize=224,
		in_chans=1,
	),

	"default": ImagePreprocessProfile(
		name="default",
		preprocessing=ImagePreprocessConfig(
			clip_data=False,
			zscale=False,
			zscale_contrast=0.25,
			norm_min=0.0,
			norm_max=1.0,
			set_zero_to_min=False,
		),
		imgsize=None,
		in_chans=None,
	),
}


def list_profiles() -> tuple[str, ...]:
	"""Return available profile names."""
	return tuple(sorted(_PROFILES))


def get_profile(name: str) -> ImagePreprocessProfile:
	"""Return a named preprocessing profile."""
	try:
		return _PROFILES[name]
	except KeyError as exc:
		raise KeyError(f"Unknown preprocessing profile '{name}'. Available: {', '.join(list_profiles())}") from exc


def _replace_invalid(data: np.ndarray, set_zero_to_min: bool) -> np.ndarray:
	data = np.asarray(data, dtype=np.float32).copy()
	if set_zero_to_min:
		valid = np.logical_and(np.isfinite(data), data != 0)
	else:
		valid = np.isfinite(data)

	values = data[valid]
	if values.size == 0:
		raise ValueError("Input image is entirely zero/non-finite after validity selection")

	fill_value = float(values.min()) if set_zero_to_min else 0.0
	data[~valid] = fill_value
	return data


def _sigma_clip_image(data: np.ndarray, sigma_low: float, sigma_up: float) -> np.ndarray:
	valid = np.logical_and(data != 0, np.isfinite(data))
	values = data[valid]
	if values.size == 0:
		return data

	_, lower, upper = sigma_clip(
		values,
		sigma_lower=sigma_low,
		sigma_upper=sigma_up,
		masked=True,
		return_bounds=True,
	)
	output = data.copy()
	output[output < lower] = lower
	output[output > upper] = upper
	return output


def _zscale_image(data: np.ndarray, contrast: float) -> np.ndarray:
	return np.asarray(ZScaleInterval(contrast=contrast)(data), dtype=np.float32)


def _minmax_normalize(data: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
	data_min = float(np.nanmin(data))
	data_max = float(np.nanmax(data))
	if not np.isfinite(data_min) or not np.isfinite(data_max):
		raise ValueError("Cannot normalize an image with non-finite extrema")
	if data_max == data_min:
		return np.full_like(data, norm_min, dtype=np.float32)
	return ((data - data_min) / (data_max - data_min) * (norm_max - norm_min) + norm_min).astype(np.float32)


def apply_image_preprocessing(data: np.ndarray, config: ImagePreprocessConfig) -> np.ndarray:
	"""Apply the common domain-level preprocessing pipeline."""
	output = _replace_invalid(data, config.set_zero_to_min)
	if config.clip_data:
		output = _sigma_clip_image(output, config.sigma_low, config.sigma_up)
	if config.zscale:
		output = _zscale_image(output, config.zscale_contrast)
	output = _minmax_normalize(output, config.norm_min, config.norm_max)
	return output


def override_config(config: ImagePreprocessConfig, **changes) -> ImagePreprocessConfig:
	"""Return a copy of a frozen preprocessing configuration with overrides."""
	return replace(config, **changes)
