import numpy as np

from fextractor.preprocessing.image import ImagePreprocessConfig, apply_image_preprocessing, get_profile


def test_minmax_normalization():
	data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
	config = ImagePreprocessConfig(norm_min=0.0, norm_max=1.0)
	output = apply_image_preprocessing(data, config)
	assert output.min() == 0.0
	assert output.max() == 1.0


def test_constant_image_is_safe():
	data = np.full((4, 4), 7.0, dtype=np.float32)
	output = apply_image_preprocessing(data, ImagePreprocessConfig())
	assert np.all(output == 0.0)


def test_simclr_radio_profile():
	profile = get_profile("simclr_radio")
	assert profile.imgsize == 224
	assert profile.in_chans == 1
	assert profile.preprocessing.zscale is True
	assert profile.preprocessing.zscale_contrast == 0.25
