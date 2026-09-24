"""Command-line interface for fextractor."""

from __future__ import annotations

import logging
import time
import argparse

from .config import ExtractorConfig
from .factory import create_extractor
from .io import read_datalist, save_datalist_json, save_feature_vector
from .logging_utils import configure_logging
from .preprocessing import ImagePreprocessConfig, get_profile, list_profiles
from .registry import list_backends
from .runner import extract_datalist


def build_parser() -> argparse.ArgumentParser:
	"""Build the command-line parser."""
	parser = argparse.ArgumentParser(description="Extract features/representations from pretrained models.")
	parser.add_argument("--backend", required=True, choices=list_backends())

	inputs = parser.add_mutually_exclusive_group(required=True)
	inputs.add_argument("--image", help="Single FITS/PNG/JPEG input image")
	inputs.add_argument("--inputfile", help="Input datalist JSON")

	parser.add_argument("--outfile", required=True, help="Output JSON file")
	parser.add_argument("--datalist-key", default="data")
	parser.add_argument("--nmax", type=int, default=-1)
	parser.add_argument("--skip-errors", action="store_true")
	parser.add_argument("--profile", choices=list_profiles())
	parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO", help="Logging level (default=INFO)")

	parser.add_argument("--model", help="Model file/path/name, depending on backend")
	parser.add_argument("--model-weights", help="Optional TensorFlow weights file")
	parser.add_argument("--keras-loader", choices=("auto", "keras", "tf_keras"), default="auto", help="TensorFlow: model loader to use (default=auto)")
	
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--imgsize", type=int)
	parser.add_argument("--in-chans", type=int)

	parser.add_argument("--clip-data", action=argparse.BooleanOptionalAction, default=None)
	parser.add_argument("--zscale", action=argparse.BooleanOptionalAction, default=None)
	parser.add_argument("--zscale-contrast", type=float)
	parser.add_argument("--norm-min", type=float)
	parser.add_argument("--norm-max", type=float)
	parser.add_argument("--set-zero-to-min", action=argparse.BooleanOptionalAction, default=None)

	parser.add_argument("--reset-meanstd", action="store_true", help="SigLIP: reset processor mean/std")
	parser.add_argument("--reset-rescale", action="store_true", help="SigLIP: disable processor rescaling")
	return parser


def _resolve_preprocessing(args) -> tuple[ImagePreprocessConfig, int | None, int | None]:
	"""Resolve the current image preprocessing profile and CLI overrides."""
	if args.profile:
		profile = get_profile(args.profile)
		base = profile.preprocessing
		imgsize = profile.imgsize
		in_chans = profile.in_chans
	else:
		base = ImagePreprocessConfig()
		imgsize = None
		in_chans = None

	changes = {}
	for field_name, arg_name in (
		("clip_data", "clip_data"),
		("zscale", "zscale"),
		("zscale_contrast", "zscale_contrast"),
		("norm_min", "norm_min"),
		("norm_max", "norm_max"),
		("set_zero_to_min", "set_zero_to_min"),
	):
		value = getattr(args, arg_name)
		if value is not None:
			changes[field_name] = value

	config_data = base.__dict__.copy()
	config_data.update(changes)
	preprocessing = ImagePreprocessConfig(**config_data)

	if args.imgsize is not None:
		imgsize = args.imgsize
	if args.in_chans is not None:
		in_chans = args.in_chans

	return preprocessing, imgsize, in_chans


def _config_from_args(args) -> ExtractorConfig:
	"""Translate CLI arguments into a backend-neutral extractor configuration."""
	preprocessing, imgsize, in_chans = _resolve_preprocessing(args)

	return ExtractorConfig(
		backend=args.backend,
		model=args.model,
		model_weights=args.model_weights,
		device=args.device,
		imgsize=imgsize,
		in_chans=in_chans,
		preprocessing=preprocessing,
		options={
			"keras_loader": args.keras_loader,
			"reset_meanstd": args.reset_meanstd,
			"reset_rescale": args.reset_rescale,
		},
	)

def main(argv=None) -> int:
	"""CLI entry point."""
	parser = build_parser()
	args = parser.parse_args(argv)

	configure_logging(args.log_level)

	start_time = time.perf_counter()

	logger.info("Starting fextractor")
	logger.info(
		"Configuration: backend='%s' model='%s' profile='%s'",
		args.backend,
		args.model,
		args.profile,
	)

	logger.debug(
		"CLI arguments: %s",
		vars(args),
	)

	try:
		config = _config_from_args(args)

		logger.info(
			"Creating extractor backend='%s'",
			config.backend,
		)

		extractor = create_extractor(config)

		if args.image:
			logger.info(
				"Extracting representation from image='%s'",
				args.image,
			)

			features = extractor.extract(args.image)

			logger.info(
				"Extracted representation with %d features",
				len(features),
			)

			save_feature_vector(
				features,
				args.outfile,
				metadata=extractor.metadata(),
			)

		else:
			logger.info(
				"Reading datalist='%s' key='%s'",
				args.inputfile,
				args.datalist_key,
			)

			datalist = read_datalist(
				args.inputfile,
				key=args.datalist_key,
			)

			logger.info(
				"Loaded datalist entries=%d",
				len(datalist),
			)

			datalist = extract_datalist(
				datalist,
				extractor,
				nmax=args.nmax,
				skip_errors=args.skip_errors,
			)

			save_datalist_json(
				datalist,
				args.outfile,
				key=args.datalist_key,
				metadata=extractor.metadata(),
			)

		elapsed = time.perf_counter() - start_time

		logger.info(
			"Output written to '%s'",
			args.outfile,
		)

		logger.info(
			"fextractor completed successfully in %.3fs",
			elapsed,
		)

	except Exception:
		elapsed = time.perf_counter() - start_time

		logger.exception(
			"fextractor failed after %.3fs",
			elapsed,
		)

		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
