from fextractor.config import ExtractorConfig
from fextractor.extractors.image.siglip2 import create
from fextractor.preprocessing import ImagePreprocessConfig


def test_siglip2_factory_options():
	config = ExtractorConfig(
		backend="siglip2",
		model="test-model",
		device="cpu",
		imgsize=256,
		preprocessing=ImagePreprocessConfig(),
		options={
			"reset_meanstd": True,
			"reset_rescale": True,
		},
	)

	extractor = create(
		config
	)

	assert extractor.imgsize == 256
	assert extractor.reset_meanstd is True
	assert extractor.reset_rescale is True
