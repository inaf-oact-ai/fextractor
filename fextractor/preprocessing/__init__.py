"""Domain preprocessing helpers and named profiles."""

from .image import (
	ImagePreprocessConfig,
	ImagePreprocessProfile,
	apply_image_preprocessing,
	get_profile as get_image_profile,
	list_profiles as list_image_profiles,
)
from .timeseries import (
	TimeSeriesPreprocessConfig,
	TimeSeriesPreprocessProfile,
	apply_timeseries_preprocessing,
	get_profile as get_timeseries_profile,
	list_profiles as list_timeseries_profiles,
)


def get_profile(
	name: str,
	modality: str = "image",
):
	"""Return a preprocessing profile for one modality."""

	if modality == "image":
		return get_image_profile(
			name
		)

	if modality == "timeseries":
		return get_timeseries_profile(
			name
		)

	raise KeyError(
		f"Unsupported preprocessing modality '{modality}'"
	)


def list_profiles(
	modality: str = "image",
) -> tuple[str, ...]:
	"""Return preprocessing profiles for one modality."""

	if modality == "image":
		return list_image_profiles()

	if modality == "timeseries":
		return list_timeseries_profiles()

	return ()


__all__ = [
	"ImagePreprocessConfig",
	"ImagePreprocessProfile",
	"TimeSeriesPreprocessConfig",
	"TimeSeriesPreprocessProfile",
	"apply_image_preprocessing",
	"apply_timeseries_preprocessing",
	"get_image_profile",
	"get_profile",
	"get_timeseries_profile",
	"list_image_profiles",
	"list_profiles",
	"list_timeseries_profiles",
]
