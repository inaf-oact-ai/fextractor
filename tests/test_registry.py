from fextractor.registry import list_backends


def test_backends_are_registered():
	assert list_backends() == (
		"tensorflow",
		"dinov2",
		"dinov3",
		"dinov2_legacy",
		"siglip",
		"siglip2",
	)
