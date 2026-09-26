from fextractor.registry import (
	get_backend_modality,
	get_backend_spec,
	list_backends,
	list_modalities,
)
	
def test_backends_are_registered():
	assert list_backends() == (
		"tensorflow",
		"dinov2",
		"dinov3",
		"dinov2_legacy",
		"siglip",
		"siglip2",
		"chronos2",
	)
	
def test_backend_modality():
	assert get_backend_modality("dinov2") == "image"
	assert get_backend_modality("siglip2") == "image"


def test_registered_modalities():
	assert list_modalities() == ("image",)
	
def test_timeseries_backend_is_registered():
	assert list_backends(
		modality="timeseries",
	) == (
		"chronos2",
	)


def test_registered_modalities():
	assert list_modalities() == (
		"image",
		"timeseries",
	)


def test_chronos_alias():
	spec = get_backend_spec(
		"chronos"
	)

	assert spec.name == "chronos2"
	assert spec.modality == "timeseries"	
