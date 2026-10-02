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
		"moirai2",
		"falcon1",
		"licu",
		"licu_embed",
	)
	
def test_backend_modality():
	assert get_backend_modality("dinov2") == "image"
	assert get_backend_modality("siglip2") == "image"
	assert get_backend_modality("chronos2") == "timeseries"
	assert get_backend_modality("moirai2") == "timeseries"
	assert get_backend_spec("falcon").modality == "timeseries"
	assert get_backend_modality("licu") == "timeseries"
	assert get_backend_modality("licu_embed") == "timeseries"

def test_backend_aliases():
	assert get_backend_spec("tf").name == "tensorflow"
	assert get_backend_spec("chronos").name == "chronos2"
	assert get_backend_spec("moirai").name == "moirai2"
	assert get_backend_spec("falcon").name == "falcon1"
	assert get_backend_spec("lightcurve").name == "licu"
	assert get_backend_spec("light-curve").name == "licu"
	assert get_backend_spec("licu-embed").name == "licu_embed"
	assert get_backend_spec("lightcurve-embed").name == "licu_embed"
	assert get_backend_spec("light-curve-embed").name == "licu_embed"

	assert get_backend_spec("chronos").modality == "timeseries"
	assert get_backend_spec("lightcurve").modality == "timeseries"
	assert get_backend_spec("licu-embed").modality == "timeseries"
		
def test_timeseries_backend_is_registered():
	assert list_backends(
		modality="timeseries",
	) == (
		"chronos2",
		"moirai2",
		"falcon1",
		"licu",
		"licu_embed",
	)


def test_registered_modalities():
	assert list_modalities() == (
		"image",
		"timeseries",
	)


