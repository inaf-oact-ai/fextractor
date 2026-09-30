"""Command-line interface for fextractor."""

from __future__ import annotations

from pathlib import Path
import logging
import time
import argparse

from .config import ExtractorConfig
from .factory import create_extractor
from .io import (
	detect_input_type,
	read_datalist,
	save_datalist_json,
	save_feature_vector,
)
from .logging_utils import configure_logging
from .runner import extract_datalist

from .preprocessing import (
	ImagePreprocessConfig,
	TimeSeriesPreprocessConfig,
	get_profile,
	#list_profiles,
)

from .registry import (
	get_backend_spec,
	list_backends,
)
from .timeseries import (
	SUPPORTED_AGGREGATIONS,
	SUPPORTED_INPUT_SAMPLE_POLICIES,
	SUPPORTED_TIMESERIES_PLOT_MODES,
	plot_timeseries_diagnostic,
)


from .extractors.timeseries.base import (
	TimeSeriesFeatureExtractor,
)


logger = logging.getLogger(__name__)


##########################################
###       OPTION PARSER
##########################################		

def build_parser() -> argparse.ArgumentParser:
	"""Build the command-line parser."""
	parser = argparse.ArgumentParser(description="Extract features/representations from pretrained models.")

	# == MANDATORY OPTIONS ==
	parser.add_argument("--backend", required=True, choices=list_backends())
	parser.add_argument("--inputfile", required=True, help="Input data file or JSON datalist")

	# == INPUT DATA OPTIONS ==
	parser.add_argument("--datalist-key", default="data")
	parser.add_argument("--nmax", type=int, default=-1)
	##parser.add_argument("--profile", choices=list_profiles())
	parser.add_argument("--profile", default="default", help="Domain preprocessing profile")
	
	# == OUTPUT DATA/SAVE OPTIONS ==
	parser.add_argument("--outfile", default="fextractor_results.json", help="Output JSON file (default: fextractor_results.json)")
	
	# == MODEL OPTIONS ==
	parser.add_argument("--model", help="Model file/path/name, depending on backend")
	parser.add_argument("--model-weights", help="Optional TensorFlow weights file")
	parser.add_argument("--keras-loader", choices=("auto", "keras", "tf_keras"), default="auto", help="TensorFlow: model loader to use (default=auto)")
	
	# == IMAGE OPTIONS ==
	parser.add_argument("--imgsize", type=int)
	parser.add_argument("--in-chans", type=int)
	parser.add_argument("--clip-data", action=argparse.BooleanOptionalAction, default=None)
	parser.add_argument("--zscale", action=argparse.BooleanOptionalAction, default=None)
	parser.add_argument("--zscale-contrast", type=float)
	parser.add_argument("--norm-min", type=float)
	parser.add_argument("--norm-max", type=float)
	parser.add_argument("--set-zero-to-min", action=argparse.BooleanOptionalAction, default=None)
	parser.add_argument("--reset-meanstd", action="store_true", help="SigLIP/SigLIP2: reset processor mean/std")
	parser.add_argument("--reset-rescale", action="store_true", help="SigLIP/SigLIP2: disable processor rescaling")
	
	# == TIME-SERIES OPTIONS ==
	parser.add_argument("--time-column", default=None, help="Time-series timestamp column")
	parser.add_argument("--value-columns", nargs="+", default=None, help="Time-series value column(s)")
	parser.add_argument("--error-columns", nargs="+", default=None, help="Time-series uncertainty column(s)")
	parser.add_argument("--regularize", action=argparse.BooleanOptionalAction, default=None, help="Regularize timestamps onto a fixed grid")
	parser.add_argument("--cadence", type=float, default=None, help="Regularization cadence in timestamp units")
	parser.add_argument("--missing-strategy", choices=("nan", "linear", "pchip", "akima", "cubic"), default=None, help="Missing-value treatment after regularization")
	parser.add_argument("--aggregation", choices=SUPPORTED_AGGREGATIONS, default=None, help="Token aggregation strategy")
	parser.add_argument("--context-length", type=int, default=None, help="Optional backend context length")
	parser.add_argument("--batch-size", type=int, default=None, help="Backend embedding batch size")
	
	parser.add_argument("--time-transform", choices=("none", "origin"), default=None, help="Time-coordinate transform")
	parser.add_argument("--value-transform", choices=("none", "maxabs", "minmax", "standard", "asinh"), default=None, help="Channel-wise value transform")
	parser.add_argument("--value-transform-scale", type=float, default=None, help="Optional scale parameter used by value transforms such as asinh")
	parser.add_argument("--alignment", choices=("none", "peak-max", "peak-min", "peak-abs"), default=None, help="Feature used as temporal alignment anchor")
	parser.add_argument("--alignment-window-before", type=float, default=None, help="Physical time to retain before the alignment anchor. If None, align and keep the entire series.")
	parser.add_argument("--alignment-window-after", type=float, default=None, help="Physical time to retain after the alignment anchor. If None, align and keep the entire series.")
	
	parser.add_argument("--bin-aggregation", choices=("mean", "inverse-variance"), default=None, help="Aggregation used when multiple observations fall in one time bin")
	parser.add_argument("--regularization-method", choices=("bin", "gp"), default=None, help="Time-series regularization method")
	parser.add_argument("--gp-sigma", type=float, default=None, help="Gaussian Process Matern-3/2 kernel amplitude. If omitted, inferred independently per channel.")
	parser.add_argument("--gp-rho", type=float, default=None, help="Gaussian Process Matern-3/2 correlation length scale in timestamp units. If omitted, inferred independently per channel.")
	parser.add_argument("--gp-jitter", type=float, default=None, help="Gaussian Process noise floor used when measurement errors are unavailable.")	
	
	parser.add_argument("--timeseries-plot", choices=SUPPORTED_TIMESERIES_PLOT_MODES, default="none", help="Save time-series diagnostic plots: none, input, processed, or both")
	parser.add_argument("--timeseries-plot-dir", default=None, help="Directory used for time-series diagnostic plots. If omitted, plots are saved in the same directory as the output JSON file")	
	
	parser.add_argument(
		"--input-sample-policy",
		choices=SUPPORTED_INPUT_SAMPLE_POLICIES,
		default=None,
		help=(
			"Samples from the prepared time series exposed to the backend. "
			"'observed' uses only measured/bin-observed samples; "
			"'completed' also uses interpolated and GP-predicted samples. "
			"Default: observed."
		),
	)
	
	# - MOIRAI OPTIONS
	parser.add_argument("--patching-mode", choices=("time_only", "time_variate"), default=None, help="Moirai-2 patching mode. If omitted, backend default is used.")
	parser.add_argument("--token-order", choices=("by_variate", "interleave_time"), default=None, help="Moirai-2 token ordering for time_variate patching. If omitted, backend default is used.")
	
	# - LICU OPTIONS
	parser.add_argument("--feature-set", choices=("basic", "default", "full"), default=None, help="LiCu handcrafted feature set. If omitted, backend default is used.")
	parser.add_argument("--invalid-feature-policy", choices=("error", "zero",), default=None, help="LiCu policy for non-finite feature values. If omitted, backend default is used.")
	parser.add_argument("--min-samples", type=int, default=None, help="LiCu minimum number of valid selected samples required per channel.")
	parser.add_argument("--licu-embed-output", choices=("mean", "max", "sequence"), default=None, help="LiCu ML embedding output. Model-specific validation is applied by the backend.")
	parser.add_argument("--licu-embed-reduction", choices=("beginning", "end", "middle", "non-overlapping-windows"), default=None, help="LiCu ML light-curve reduction/windowing strategy. If omitted, the model default is used.")
	
	# - TIME SERIES INPUT LAYOUT OPTIONS
	parser.add_argument("--timeseries-layout", choices=("long", "wide"), default=None, help="Time-series tabular layout")
	parser.add_argument("--value-prefixes", nargs="+", default=None, help="Wide-layout time-series column prefixes")
	parser.add_argument("--channel-names", nargs="+", default=None, help="Wide-layout channel names")
	parser.add_argument("--error-prefixes", nargs="+", default=None, help="Wide-layout time-series uncertainty/error prefixes")
	parser.add_argument("--time-prefix", default=None, help="Prefix of indexed timestamp columns for irregular wide-layout input")
	parser.add_argument("--time-start-column", default=None, help="Column containing the initial timestamp for regular wide-layout input")
	parser.add_argument("--cadence-column", default=None, help="Column containing the sampling cadence for regular wide-layout input")
	parser.add_argument("--label-column", default=None, help="Optional sample-level label column")
	parser.add_argument("--metadata-columns", nargs="+", default=None, help="Additional sample-level metadata columns")
	parser.add_argument("--time-start-key", default=None, help="Inline JSON field containing the start timestamp")
	parser.add_argument("--cadence-key", default=None, help="Inline JSON field containing the sampling cadence")
	
	# == RUN OPTIONS ==
	parser.add_argument("--skip-errors", action="store_true")
	parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO", help="Logging level (default=INFO)")
	parser.add_argument("--device", default="cuda")
	
	return parser



##########################################
###       PROCESSOR CONFIG
##########################################

def _resolve_image_preprocessing(
	args,
) -> tuple[ImagePreprocessConfig, int | None, int | None]:
	"""Resolve image preprocessing profile and CLI overrides."""

	if args.profile:
		profile = get_profile(
			args.profile,
			modality="image",
		)

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
		value = getattr(
			args,
			arg_name,
		)

		if value is not None:
			changes[field_name] = value

	config_data = base.__dict__.copy()
	config_data.update(
		changes
	)

	preprocessing = ImagePreprocessConfig(
		**config_data
	)

	if args.imgsize is not None:
		imgsize = args.imgsize

	if args.in_chans is not None:
		in_chans = args.in_chans

	return (
		preprocessing,
		imgsize,
		in_chans,
	)


def _resolve_timeseries_preprocessing(
	args,
) -> TimeSeriesPreprocessConfig:
	"""Resolve time-series preprocessing profile and CLI overrides."""

	if args.profile:
		profile = get_profile(
			args.profile,
			modality="timeseries",
		)

		base = profile.preprocessing

	else:
		base = TimeSeriesPreprocessConfig()

	changes = {}

	if args.time_column is not None:
		changes["time_column"] = (
			args.time_column
		)

	if args.value_columns is not None:
		changes["value_columns"] = tuple(
			args.value_columns
		)

	if args.error_columns is not None:
		changes["error_columns"] = tuple(
			args.error_columns
		)

	if args.regularize is not None:
		changes["regularize"] = (
			args.regularize
		)

	if args.cadence is not None:
		changes["cadence"] = (
			args.cadence
		)

	if args.missing_strategy is not None:
		changes["missing_strategy"] = (
			args.missing_strategy
		)
		
	if args.timeseries_layout is not None:
		changes["layout"] = (
			args.timeseries_layout
		)

	if args.value_prefixes is not None:
		changes["value_prefixes"] = tuple(
			args.value_prefixes
		)

	if args.channel_names is not None:
		changes["channel_names"] = tuple(
			args.channel_names
		)
		
	if args.error_prefixes is not None:
		changes["error_prefixes"] = tuple(
			args.error_prefixes
		)

	if args.time_prefix is not None:
		changes["time_prefix"] = (
			args.time_prefix
		)

	if args.time_start_column is not None:
		changes["time_start_column"] = (
			args.time_start_column
		)

	if args.cadence_column is not None:
		changes["cadence_column"] = (
			args.cadence_column
		)

	if args.label_column is not None:
		changes["label_column"] = (
			args.label_column
		)

	if args.metadata_columns is not None:
		changes["metadata_columns"] = tuple(
			args.metadata_columns
		)

	if args.time_start_key is not None:
		changes["time_start_key"] = (
			args.time_start_key
		)

	if args.cadence_key is not None:
		changes["cadence_key"] = (
			args.cadence_key
		)

	if args.time_transform is not None:
		changes["time_transform"] = (
			args.time_transform
		)

	if args.value_transform is not None:
		changes["value_transform"] = (
			args.value_transform
		)

	if args.value_transform_scale is not None:
		changes["value_transform_scale"] = (
			args.value_transform_scale
		)
		
	if args.alignment is not None:
		changes["alignment"] = (
			args.alignment
		)

	if args.alignment_window_before is not None:
		changes["alignment_window_before"] = (
			args.alignment_window_before
		)

	if args.alignment_window_after is not None:
		changes["alignment_window_after"] = (
			args.alignment_window_after
		)	

	if args.bin_aggregation is not None:
		changes["bin_aggregation"] = (
			args.bin_aggregation
		)
		
	if args.regularization_method is not None:
		changes["regularization_method"] = (
			args.regularization_method
		)

	if args.gp_sigma is not None:
		changes["gp_sigma"] = (
			args.gp_sigma
		)

	if args.gp_rho is not None:
		changes["gp_rho"] = (
			args.gp_rho
		)

	if args.gp_jitter is not None:
		changes["gp_jitter"] = (
			args.gp_jitter
		)

	# - Update config
	config_data = base.__dict__.copy()
	config_data.update(
		changes
	)

	return TimeSeriesPreprocessConfig(
		**config_data
	)

def _config_from_args(
	args,
) -> ExtractorConfig:
	"""Translate CLI arguments into backend-neutral configuration."""

	spec = get_backend_spec(
		args.backend
	)

	imgsize = None
	in_chans = None

	if spec.modality == "image":
		(
			preprocessing,
			imgsize,
			in_chans,
		) = _resolve_image_preprocessing(
			args
		)

	elif spec.modality == "timeseries":
		preprocessing = (
			_resolve_timeseries_preprocessing(
				args
			)
		)

	else:
		raise ValueError(
			f"Unsupported backend modality '{spec.modality}' "
			f"for backend '{spec.name}'"
		)

	options = {
		"keras_loader": args.keras_loader,
		"reset_meanstd": args.reset_meanstd,
		"reset_rescale": args.reset_rescale,
	}

	if args.aggregation is not None:
		options["aggregation"] = (
			args.aggregation
		)

	if args.context_length is not None:
		options["context_length"] = (
			args.context_length
		)

	if args.batch_size is not None:
		options["batch_size"] = (
			args.batch_size
		)
		
	if args.patching_mode is not None:
		options["patching_mode"] = (
			args.patching_mode
		)

	if args.token_order is not None:
		options["token_order"] = (
			args.token_order
		)
		
	if args.feature_set is not None:
		options["feature_set"] = (
			args.feature_set
		)

	if args.invalid_feature_policy is not None:
		options["invalid_feature_policy"] = (
			args.invalid_feature_policy
		)

	if args.min_samples is not None:
		options["min_samples"] = (
			args.min_samples
		)
			
	if args.licu_embed_output is not None:
		options["licu_embed_output"] = (
			args.licu_embed_output
		)

	if args.licu_embed_reduction is not None:
		options["licu_embed_reduction"] = (
			args.licu_embed_reduction
		)

	if args.input_sample_policy is not None:
		options["input_sample_policy"] = (
			args.input_sample_policy
		)
		
	return ExtractorConfig(
		backend=args.backend,
		model=args.model,
		model_weights=args.model_weights,
		device=args.device,
		imgsize=imgsize,
		in_chans=in_chans,
		preprocessing=preprocessing,
		options=options,
	)
	
##########################################
###       MAIN
##########################################		

def main(argv=None) -> int:
	"""CLI entry point."""
	
	# - Parse options
	parser = build_parser()
	args = parser.parse_args(argv)

	# - Configure logging
	configure_logging(args.log_level)

	# - Run feat extraction
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
	
		# - Parse config and backend specification
		config = _config_from_args(
			args
		)

		spec = get_backend_spec(
			config.backend
		)

		# - Detect input type using backend modality
		input_type = detect_input_type(
			args.inputfile,
			modality=spec.modality,
		)

		logger.info(
			"Detected input type='%s' modality='%s' file='%s'",
			input_type,
			spec.modality,
			args.inputfile,
		)

		# - Create extractor
		logger.info(
			"Creating extractor backend='%s'",
			config.backend,
		)

		extractor = create_extractor(
			config
		)
		
		# - Resolve output plot dirs
		timeseries_plot_dir = None

		if args.timeseries_plot != "none":
			if args.timeseries_plot_dir is not None:
				timeseries_plot_dir = Path(
					args.timeseries_plot_dir
				)

			else:
				timeseries_plot_dir = (
					Path(
						args.outfile
					).resolve().parent
				)

			timeseries_plot_dir.mkdir(
				parents=True,
				exist_ok=True,
			)

			logger.info(
				"Time-series diagnostic plotting enabled: "
				"mode='%s' directory='%s'",
				args.timeseries_plot,
				timeseries_plot_dir,
			)
			
	
		# - Extract features
		if input_type in (
			"image",
			"timeseries",
		):
			logger.info(
				"Extracting representation from %s='%s'",
				input_type,
				args.inputfile,
			)

			if (
				input_type == "timeseries"
				and args.timeseries_plot != "none"
				and isinstance(
					extractor,
					TimeSeriesFeatureExtractor,
				)
			):
				(
					input_series,
					processed_series,
				) = extractor.prepare_with_input(
					args.inputfile
				)

				plot_path = (
					timeseries_plot_dir
					/ (
						f"{Path(args.inputfile).stem}"
						"_timeseries.png"
					)
				)

				plot_timeseries_diagnostic(
					input_series,
					processed_series,
					plot_path,
					mode=args.timeseries_plot,
					title=Path(
						args.inputfile
					).name,
				)

				features = (
					extractor.extract_prepared(
						processed_series
					)
				)

			else:
				features = extractor.extract(
					args.inputfile
				)			

			logger.info(
				"Extracted representation with %d features",
				len(features),
			)

			# - Save features
			save_feature_vector(
				features,
				args.outfile,
				metadata=extractor.metadata(),
			)

		elif input_type == "datalist":
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
				timeseries_plot=args.timeseries_plot,
				timeseries_plot_dir=timeseries_plot_dir,
			)			

			# - Save features
			save_datalist_json(
				datalist,
				args.outfile,
				key=args.datalist_key,
				metadata=extractor.metadata(),
			)

		else:
			raise RuntimeError(
				f"Unsupported detected input type '{input_type}'"
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
