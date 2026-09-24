"""Command-line interface for fextractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_datalist, save_datalist_json, save_feature_vector
from .preprocessing import ImagePreprocessConfig, get_profile, list_profiles
from .registry import create_extractor, list_backends
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

	parser.add_argument("--model", help="Model file/path/name, depending on backend")
	parser.add_argument("--model-weights", help="Optional TensorFlow weights file")
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
	return ImagePreprocessConfig(**config_data), args.imgsize or imgsize, args.in_chans or in_chans


def _create_from_args(args):
	preprocessing, imgsize, in_chans = _resolve_preprocessing(args)

	if args.backend == "tensorflow":
		if not args.model:
			raise ValueError("--model is required for the TensorFlow backend")
		return create_extractor(
			"tensorflow",
			model_path=args.model,
			weights_path=args.model_weights,
			imgsize=imgsize or 224,
			in_chans=in_chans or 1,
			preprocessing=preprocessing,
		)

	if args.backend == "dinov2":
		return create_extractor(
			"dinov2",
			model_name=args.model or "dinov2_vits14",
			device=args.device,
			imgsize=imgsize or 224,
			preprocessing=preprocessing,
		)

	if args.backend == "siglip":
		return create_extractor(
			"siglip",
			model_name=args.model or "google/siglip-so400m-patch14-384",
			device=args.device,
			imgsize=imgsize,
			reset_meanstd=args.reset_meanstd,
			reset_rescale=args.reset_rescale,
			preprocessing=preprocessing,
		)

	raise ValueError(f"Unsupported backend: {args.backend}")


def main(argv=None) -> int:
	"""CLI entry point."""
	parser = build_parser()
	args = parser.parse_args(argv)

	try:
		extractor = _create_from_args(args)
		if args.image:
			features = extractor.extract(args.image)
			save_feature_vector(features, args.outfile, metadata=extractor.metadata())
		else:
			datalist = read_datalist(args.inputfile, key=args.datalist_key)
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
	except Exception as exc:
		parser.error(str(exc))

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
