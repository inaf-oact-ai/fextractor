#!/usr/bin/env python2.7

from __future__ import print_function

import argparse
import csv
import json
import os
import sys

import numpy as np
import FATS


BACKEND_NAME = "fats"
MODEL_NAME = "FATS"
MODEL_VERSION = "1.3.6"


def build_parser():
	parser = argparse.ArgumentParser(
		description=(
			"Extract handcrafted time-series features "
			"using the legacy FATS package"
		)
	)

	parser.add_argument(
		"--inputfile",
		required=True,
		help="Input CSV time-series file",
	)

	parser.add_argument(
		"--outfile",
		default="fextractor_results.json",
		help="Output JSON filename",
	)

	parser.add_argument(
		"--time-column",
		default=None,
		help="Timestamp column name",
	)

	parser.add_argument(
		"--value-columns",
		nargs="+",
		required=True,
		help="One or more time-series value columns",
	)

	parser.add_argument(
		"--error-columns",
		nargs="+",
		default=None,
		help=(
			"Optional uncertainty columns. "
			"Must contain one column per value column."
		),
	)

	parser.add_argument(
		"--channel-names",
		nargs="+",
		default=None,
		help=(
			"Optional logical names assigned to "
			"the value channels"
		),
	)

	return parser


def validate_args(args):
	if (
		args.error_columns is not None
		and len(args.error_columns)
		!= len(args.value_columns)
	):
		raise ValueError(
			"error-columns must contain one column "
			"per value column"
		)

	if (
		args.channel_names is not None
		and len(args.channel_names)
		!= len(args.value_columns)
	):
		raise ValueError(
			"channel-names must contain one name "
			"per value column"
		)

	ext = os.path.splitext(
		args.inputfile
	)[1].lower()

	if ext != ".csv":
		raise ValueError(
			"Initial FATS runtime supports CSV input only, "
			"got '%s'"
			% ext
		)


def read_csv_columns(
	filename,
	required_columns,
):
	with open(
		filename,
		"rb",
	) as handle:
		reader = csv.DictReader(
			handle
		)

		if reader.fieldnames is None:
			raise ValueError(
				"CSV file has no header"
			)

		missing_columns = [
			name
			for name in required_columns
			if name not in reader.fieldnames
		]

		if missing_columns:
			raise ValueError(
				"Column(s) not found in CSV: %s. "
				"Available columns: %s"
				% (
					", ".join(
						missing_columns
					),
					", ".join(
						reader.fieldnames
					),
				)
			)

		data = dict(
			(
				name,
				[],
			)
			for name in required_columns
		)

		for row in reader:
			for name in required_columns:
				raw_value = row.get(
					name,
					"",
				)

				try:
					value = float(
						raw_value
					)

				except (
					TypeError,
					ValueError,
				):
					value = np.nan

				data[name].append(
					value
				)

	output = {}

	for name, values in data.items():
		output[name] = np.asarray(
			values,
			dtype=np.float64,
		)

	return output


def select_valid_samples(
	time,
	values,
	errors=None,
):
	values = np.asarray(
		values,
		dtype=np.float64,
	)

	valid = np.isfinite(
		values
	)

	if time is not None:
		time = np.asarray(
			time,
			dtype=np.float64,
		)

		valid &= np.isfinite(
			time
		)

	if errors is not None:
		errors = np.asarray(
			errors,
			dtype=np.float64,
		)

		valid &= np.isfinite(
			errors
		)

		# Several FATS features divide by error^2.
		# Zero or negative uncertainties are therefore
		# not valid measurement-error entries.
		valid &= errors > 0.0

	values = values[
		valid
	]

	if time is not None:
		time = time[
			valid
		]

	if errors is not None:
		errors = errors[
			valid
		]

	return (
		time,
		values,
		errors,
	)


def sort_by_time(
	time,
	values,
	errors=None,
):
	if time is None:
		return (
			time,
			values,
			errors,
		)

	order = np.argsort(
		time
	)

	time = time[
		order
	]

	values = values[
		order
	]

	if errors is not None:
		errors = errors[
			order
		]

	return (
		time,
		values,
		errors,
	)


def extract_channel(
	values,
	time=None,
	errors=None,
):
	if len(values) == 0:
		raise ValueError(
			"No valid observations remain "
			"after filtering"
		)

	data_fields = [
		"magnitude",
	]

	data_arrays = [
		values,
	]

	if time is not None:
		data_fields.append(
			"time"
		)

		data_arrays.append(
			time
		)

	if errors is not None:
		if time is None:
			raise ValueError(
				"FATS uncertainty input requires "
				"timestamps"
			)

		data_fields.append(
			"error"
		)

		data_arrays.append(
			errors
		)

	feature_space = FATS.FeatureSpace(
		Data=data_fields
	)

	feature_space.calculateFeature(
		np.asarray(
			data_arrays
		)
	)

	feature_names = list(
		feature_space.result(
			method="features"
		)
	)

	feature_values = np.asarray(
		feature_space.result(
			method="array"
		),
		dtype=np.float64,
	).reshape(-1)

	if (
		len(feature_names)
		!= len(feature_values)
	):
		raise RuntimeError(
			"FATS feature-name/value mismatch: "
			"%d names vs %d values"
			% (
				len(feature_names),
				len(feature_values),
			)
		)

	if not np.all(
		np.isfinite(
			feature_values
		)
	):
		bad_features = []

		for (
			name,
			value,
		) in zip(
			feature_names,
			feature_values,
		):
			if not np.isfinite(
				value
			):
				bad_features.append(
					name
				)

		raise RuntimeError(
			"FATS returned non-finite feature "
			"value(s): %s"
			% ", ".join(
				bad_features
			)
		)

	return (
		feature_names,
		feature_values,
	)


def extract_features(
	columns,
	time_column,
	value_columns,
	error_columns=None,
	channel_names=None,
):
	time = None

	if time_column is not None:
		time = columns[
			time_column
		]

	if channel_names is None:
		channel_names = list(
			value_columns
		)

	all_feature_names = []
	all_feature_values = []

	channel_metadata = []

	for (
		channel_index,
		value_column,
	) in enumerate(
		value_columns
	):
		channel_name = channel_names[
			channel_index
		]

		values = columns[
			value_column
		]

		errors = None

		if error_columns is not None:
			error_column = error_columns[
				channel_index
			]

			errors = columns[
				error_column
			]

		else:
			error_column = None

		channel_time = (
			None
			if time is None
			else time.copy()
		)

		n_input = len(
			values
		)

		(
			channel_time,
			values,
			errors,
		) = select_valid_samples(
			time=channel_time,
			values=values,
			errors=errors,
		)

		(
			channel_time,
			values,
			errors,
		) = sort_by_time(
			time=channel_time,
			values=values,
			errors=errors,
		)

		n_valid = len(
			values
		)

		print(
			"INFO: Extracting FATS features "
			"channel='%s' input_samples=%d "
			"valid_samples=%d errors=%s"
			% (
				channel_name,
				n_input,
				n_valid,
				(
					"yes"
					if errors is not None
					else "no"
				),
			)
		)

		(
			names,
			features,
		) = extract_channel(
			values=values,
			time=channel_time,
			errors=errors,
		)

		channel_feature_names = [
			"%s.%s"
			% (
				channel_name,
				name,
			)
			for name in names
		]

		all_feature_names.extend(
			channel_feature_names
		)

		all_feature_values.extend(
			features.tolist()
		)

		channel_metadata.append({
			"name": channel_name,
			"value_column": value_column,
			"error_column": error_column,
			"n_input": int(
				n_input
			),
			"n_valid": int(
				n_valid
			),
			"n_features": int(
				len(features)
			),
		})

	return (
		all_feature_names,
		np.asarray(
			all_feature_values,
			dtype=np.float64,
		),
		channel_metadata,
	)


def save_output(
	filename,
	features,
	feature_names,
	channel_metadata,
	time_column,
):
	payload = {
		"feats": [
			float(
				value
			)
			for value in features
		],
		"fextractor": {
			"backend": BACKEND_NAME,
			"modality": "timeseries",
			"model": MODEL_NAME,
			"model_version": MODEL_VERSION,
			"time_column": time_column,
			"n_features": int(
				len(features)
			),
			"feature_names": list(
				feature_names
			),
			"channels": channel_metadata,
		},
	}

	with open(
		filename,
		"w",
	) as handle:
		json.dump(
			payload,
			handle,
			indent=2,
			allow_nan=False,
		)

		handle.write(
			"\n"
		)


def main(argv=None):
	parser = build_parser()

	args = parser.parse_args(
		argv
	)

	validate_args(
		args
	)

	required_columns = []

	if args.time_column is not None:
		required_columns.append(
			args.time_column
		)

	required_columns.extend(
		args.value_columns
	)

	if args.error_columns is not None:
		required_columns.extend(
			args.error_columns
		)

	# Preserve order while removing duplicates.
	required_columns = list(
		dict.fromkeys(
			required_columns
		)
	)

	print(
		"INFO: Starting FATS extractor"
	)

	print(
		"INFO: input='%s'"
		% args.inputfile
	)

	print(
		"INFO: value_columns=%s"
		% ",".join(
			args.value_columns
		)
	)

	if args.error_columns is not None:
		print(
			"INFO: error_columns=%s"
			% ",".join(
				args.error_columns
			)
		)

	columns = read_csv_columns(
		filename=args.inputfile,
		required_columns=required_columns,
	)

	channel_names = (
		args.channel_names
		if args.channel_names is not None
		else args.value_columns
	)

	(
		feature_names,
		features,
		channel_metadata,
	) = extract_features(
		columns=columns,
		time_column=args.time_column,
		value_columns=args.value_columns,
		error_columns=args.error_columns,
		channel_names=channel_names,
	)

	save_output(
		filename=args.outfile,
		features=features,
		feature_names=feature_names,
		channel_metadata=channel_metadata,
		time_column=args.time_column,
	)

	print(
		"INFO: FATS extraction completed "
		"channels=%d features=%d"
		% (
			len(channel_names),
			len(features),
		)
	)

	print(
		"INFO: Output written to '%s'"
		% args.outfile
	)

	return 0


if __name__ == "__main__":
	try:
		sys.exit(
			main()
		)

	except Exception as exc:
		print(
			"ERROR: FATS extraction failed: %s"
			% exc,
			file=sys.stderr,
		)

		raise
