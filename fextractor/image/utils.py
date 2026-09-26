"""Shared image utility functions."""

from __future__ import annotations

import numpy as np
from skimage.transform import resize as sk_resize


SUPPORTED_IMAGE_EXTENSIONS = {".fits", ".fit", ".fts", ".png", ".jpg", ".jpeg"}


def resize_square(
	image: np.ndarray,
	size: int,
	order: int = 3,
	preserve_range: bool = True,
	anti_aliasing: bool = False,
) -> np.ndarray:
	"""Resize while preserving aspect ratio, then zero-pad to a square.

	This follows the behavior used by the original TensorFlow macro: the
	shortest side is first scaled to the requested size, unless that would make
	the longest side exceed it; the result is then centered in a square canvas.
	"""
	if image.ndim not in (2, 3):
		raise ValueError(f"Unsupported image ndim={image.ndim}; expected 2 or 3")

	original_dtype = image.dtype
	height, width = image.shape[:2]
	if height <= 0 or width <= 0:
		raise ValueError("Cannot resize an empty image")

	scale = max(1.0, float(size) / float(min(height, width)))
	if round(max(height, width) * scale) > size:
		scale = float(size) / float(max(height, width))

	if scale != 1.0:
		new_height = max(1, round(height * scale))
		new_width = max(1, round(width * scale))
		output_shape = (new_height, new_width)
		if image.ndim == 3:
			output_shape = (new_height, new_width, image.shape[2])
		image = sk_resize(
			image,
			output_shape,
			order=order,
			mode="constant",
			cval=0,
			clip=True,
			preserve_range=preserve_range,
			anti_aliasing=anti_aliasing,
		)

	height, width = image.shape[:2]
	top = (size - height) // 2
	bottom = size - height - top
	left = (size - width) // 2
	right = size - width - left

	if image.ndim == 2:
		padding = ((top, bottom), (left, right))
	else:
		padding = ((top, bottom), (left, right), (0, 0))

	image = np.pad(image, padding, mode="constant", constant_values=0)
	return image.astype(original_dtype, copy=False)


def ensure_channels(image: np.ndarray, channels: int) -> np.ndarray:
	"""Return HWC data with the requested number of channels."""
	if channels not in (1, 3):
		raise ValueError("Only one-channel and three-channel image inputs are supported")

	if image.ndim == 2:
		if channels == 1:
			return image[..., np.newaxis]
		return np.repeat(image[..., np.newaxis], 3, axis=-1)

	if image.ndim != 3:
		raise ValueError(f"Unsupported image ndim={image.ndim}; expected 2 or 3")

	current = image.shape[-1]
	if current == channels:
		return image
	if current == 1 and channels == 3:
		return np.repeat(image, 3, axis=-1)
	if current >= 3 and channels == 1:
		return image[..., :3].mean(axis=-1, keepdims=True)
	if current >= 3 and channels == 3:
		return image[..., :3]

	raise ValueError(f"Cannot convert image with {current} channels to {channels}")


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
	"""Convert normalized image data to RGB uint8 for standard vision processors."""
	image = ensure_channels(image, 3)
	if np.issubdtype(image.dtype, np.integer):
		return np.clip(image, 0, 255).astype(np.uint8)

	finite = image[np.isfinite(image)]
	if finite.size == 0:
		raise ValueError("Image contains no finite pixels")

	if finite.min() >= 0.0 and finite.max() <= 1.0:
		image = image * 255.0

	return np.clip(image, 0.0, 255.0).astype(np.uint8)
