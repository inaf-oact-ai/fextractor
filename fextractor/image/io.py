"""Image file readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from astropy.io import fits


SUPPORTED_IMAGE_EXTENSIONS = {
	".fits",
	".fit",
	".fts",
	".png",
	".jpg",
	".jpeg",
}


def read_image(filename: str | Path) -> np.ndarray:
	"""Read FITS, PNG, or JPEG image data into a NumPy array."""

	path = Path(filename)
	ext = path.suffix.lower()

	if ext in {".fits", ".fit", ".fts"}:
		with fits.open(path, memmap=False) as hdul:
			data = np.asarray(hdul[0].data)

	elif ext in {".png", ".jpg", ".jpeg"}:
		with Image.open(path) as image:
			data = np.asarray(image)

	else:
		raise ValueError(
			f"Unsupported image extension '{ext}' for '{path}'"
		)

	if data is None or data.size == 0:
		raise ValueError(
			f"No image data found in '{path}'"
		)

	data = np.squeeze(data)

	if data.ndim not in (2, 3):
		raise ValueError(
			f"Expected a 2D or 3D image in '{path}', "
			f"got shape {data.shape}"
		)

	return data

