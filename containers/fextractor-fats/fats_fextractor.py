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
		help="Input CSV time-series file or JSON datalist",
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
		default=None,
		help=(
			"One or more time-series value columns "
			"for long layout"
		),
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
		"--timeseries-layout",
		choices=[
			"long",
			"wide",
		],
		default="long",
	)

	parser.add_argument(
		"--value-prefixes",
		nargs="+",
		default=None,
	)

	parser.add_argument(
		"--error-prefixes",
		nargs="+",
		default=None,
	)

	parser.add_argument(
		"--time-prefix",
		default=None,
	)

	parser.add_argument(
		"--time-start-column",
		default=None,
	)

	parser.add_argument(
		"--cadence-column",
		default=None,
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
	
	parser.add_argument(
		"--datalist-key",
		default="data",
	)

	parser.add_argument(
		"--nmax",
		type=int,
		default=-1,
	)

	parser.add_argument(
		"--time-start-key",
		default=None,
	)

	parser.add_argument(
		"--cadence-key",
		default=None,
	)

	return parser


def validate_args(args):
	ext = os.path.splitext(
		args.inputfile
	)[1].lower()

	if ext not in (
		".csv",
		".json",
	):
		raise ValueError(
			"FATS runtime supports CSV and JSON input, "
			"got '%s'"
			% ext
		)

	# - Validate JSON input
	if ext == ".json":
		if (
			args.time_start_key is None
			and args.cadence_key is not None
		):
			raise ValueError(
				"--time-start-key and --cadence-key "
				"must be supplied together"
			)

		if (
			args.time_start_key is not None
			and args.cadence_key is None
		):
			raise ValueError(
				"--time-start-key and --cadence-key "
				"must be supplied together"
			)

		if (
			args.time_column is not None
			and (
				args.time_start_key is not None
				or args.cadence_key is not None
			)
		):
			raise ValueError(
				"Inline JSON input must use either "
				"--time-column or "
				"--time-start-key + --cadence-key, "
				"not both"
			)

		if (
			args.error_columns is not None
			and not args.value_columns
		):
			raise ValueError(
				"--error-columns requires --value-columns "
				"for inline JSON records"
			)

		if (
			args.error_columns is not None
			and len(args.error_columns)
			!= len(args.value_columns)
		):
			raise ValueError(
				"error-columns must contain one field "
				"per value field"
			)

		if (
			args.channel_names is not None
			and args.value_columns is not None
			and len(args.channel_names)
			!= len(args.value_columns)
		):
			raise ValueError(
				"channel-names must contain one name "
				"per value field"
			)

		return	

	# - Validate CSV input
	if args.timeseries_layout == "long":
		if not args.value_columns:
			raise ValueError(
				"Long layout requires --value-columns"
			)

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

		if (
			args.error_columns is not None
			and args.time_column is None
		):
			raise ValueError(
				"Long-layout FATS input requires "
				"--time-column when --error-columns "
				"are provided"
			)

	elif args.timeseries_layout == "wide":
		if not args.value_prefixes:
			raise ValueError(
				"Wide layout requires --value-prefixes"
			)

		if args.value_columns is not None:
			raise ValueError(
				"--value-columns is not used with wide layout"
			)

		if args.error_columns is not None:
			raise ValueError(
				"--error-columns is not used with wide layout; "
				"use --error-prefixes"
			)

		if (
			args.error_prefixes is not None
			and len(args.error_prefixes)
			!= len(args.value_prefixes)
		):
			raise ValueError(
				"error-prefixes must contain one prefix "
				"per value prefix"
			)

		if (
			args.channel_names is not None
			and len(args.channel_names)
			!= len(args.value_prefixes)
		):
			raise ValueError(
				"channel-names must contain one name "
				"per value prefix"
			)

		has_time_prefix = (
			args.time_prefix is not None
		)

		has_time_start = (
			args.time_start_column is not None
		)

		has_cadence = (
			args.cadence_column is not None
		)

		if (
			has_time_prefix
			and (
				has_time_start
				or has_cadence
			)
		):
			raise ValueError(
				"Wide layout must use either --time-prefix "
				"or --time-start-column + --cadence-column, "
				"not both"
			)

		if has_time_start != has_cadence:
			raise ValueError(
				"--time-start-column and --cadence-column "
				"must be supplied together"
			)

	else:
		raise ValueError(
			"Unsupported time-series layout '%s'"
			% args.timeseries_layout
		)

def build_inline_channels(
	record,
	value_keys,
	error_keys=None,
	time_key=None,
	time_start_key=None,
	cadence_key=None,
	channel_names=None,
):
	if not value_keys:
		raise ValueError(
			"Inline time-series records require value columns"
		)

	if channel_names is None:
		channel_names = list(
			value_keys
		)

	if len(channel_names) != len(value_keys):
		raise ValueError(
			"channel-names must contain one name "
			"per value field"
		)

	if (
		error_keys is not None
		and len(error_keys) != len(value_keys)
	):
		raise ValueError(
			"error-columns must contain one field "
			"per value field"
		)

	value_arrays = []

	for key in value_keys:
		if key not in record:
			raise KeyError(
				"Time-series field '%s' not found "
				"in inline record"
				% key
			)

		values = np.asarray(
			record[key],
			dtype=np.float64,
		).reshape(-1)

		value_arrays.append(
			values
		)

	lengths = set(
		len(values)
		for values in value_arrays
	)

	if len(lengths) != 1:
		raise ValueError(
			"Inline time-series fields have "
			"different lengths: %s"
			% sorted(
				lengths
			)
		)

	n_time = lengths.pop()

	error_arrays = None

	if error_keys is not None:
		error_arrays = []

		for key in error_keys:
			if key not in record:
				raise KeyError(
					"Time-series error field '%s' "
					"not found in inline record"
					% key
				)

			errors = np.asarray(
				record[key],
				dtype=np.float64,
			).reshape(-1)

			if len(errors) != n_time:
				raise ValueError(
					"Time-series error field '%s' "
					"has length %d, expected %d"
					% (
						key,
						len(errors),
						n_time,
					)
				)

			error_arrays.append(
				errors
			)

	time = None

	if time_key is not None:
		if time_key not in record:
			raise KeyError(
				"Time field '%s' not found "
				"in inline record"
				% time_key
			)

		time = np.asarray(
			record[time_key],
			dtype=np.float64,
		).reshape(-1)

		if len(time) != n_time:
			raise ValueError(
				"Time field '%s' has length %d, expected %d"
				% (
					time_key,
					len(time),
					n_time,
				)
			)

	elif (
		time_start_key is not None
		or cadence_key is not None
	):
		if (
			time_start_key is None
			or cadence_key is None
		):
			raise ValueError(
				"time-start-key and cadence-key "
				"must be supplied together"
			)

		if time_start_key not in record:
			raise KeyError(
				"Time-start field '%s' not found"
				% time_start_key
			)

		if cadence_key not in record:
			raise KeyError(
				"Cadence field '%s' not found"
				% cadence_key
			)

		time_start = float(
			record[
				time_start_key
			]
		)
		
		if not np.isfinite(
			time_start
		):
			raise ValueError(
				"Time start must be finite"
			)

		cadence = float(
			record[
				cadence_key
			]
		)

		if (
			not np.isfinite(cadence)
			or cadence <= 0.0
		):
			raise ValueError(
				"Cadence must be positive and finite"
			)

		time = (
			time_start
			+ np.arange(
				n_time,
				dtype=np.float64,
			)
			* cadence
		)

	channels = []

	for channel_index, values in enumerate(
		value_arrays
	):
		errors = None

		if error_arrays is not None:
			errors = error_arrays[
				channel_index
			]

		channels.append({
			"name": channel_names[
				channel_index
			],
			"values": values,
			"time": (
				None
				if time is None
				else time.copy()
			),
			"errors": errors,
			"value_source": value_keys[
				channel_index
			],
			"error_source": (
				None
				if error_keys is None
				else error_keys[
					channel_index
				]
			),
		})

	return channels

def indexed_columns(
	fieldnames,
	prefix,
):
	columns = []

	for name in fieldnames:
		if not name.startswith(
			prefix
		):
			continue

		suffix = name[
			len(prefix):
		]

		if not suffix.isdigit():
			continue

		columns.append(
			(
				int(suffix),
				name,
			)
		)

	columns.sort(
		key=lambda item: item[0]
	)

	return [
		name
		for _, name in columns
	]


def read_json_datalist(
	filename,
	key="data",
):
	with open(
		filename,
		"r",
	) as handle:
		payload = json.load(
			handle
		)

	if not isinstance(
		payload,
		dict,
	):
		raise ValueError(
			"JSON datalist root must be an object"
		)

	if key not in payload:
		raise KeyError(
			"JSON datalist key '%s' not found"
			% key
		)

	items = payload[
		key
	]

	if not isinstance(
		items,
		list,
	):
		raise ValueError(
			"JSON datalist field '%s' must be a list"
			% key
		)

	return (
		payload,
		items,
	)


def extract_json_datalist(
	filename,
	args,
):
	payload, items = read_json_datalist(
		filename,
		key=args.datalist_key,
	)

	output_items = []

	for index, item in enumerate(
		items
	):
		if (
			args.nmax >= 0
			and index >= args.nmax
		):
			break

		if not isinstance(
			item,
			dict,
		):
			raise ValueError(
				"Datalist entry %d must be an object"
				% index
			)

		if "filepath" in item:
			source = item[
				"filepath"
			]

			(
				feature_names,
				features,
				channel_metadata,
				time_metadata,
			) = extract_file(
				source,
				args,
			)

			result_item = dict(
				item
			)

			result_item[
				"feats"
			] = [
				float(value)
				for value in features
			]

		else:
			if not args.value_columns:
				raise ValueError(
					"Inline JSON time-series records "
					"require --value-columns"
				)

			channel_names = (
				args.channel_names
				if args.channel_names is not None
				else args.value_columns
			)

			channels = build_inline_channels(
				record=item,
				value_keys=args.value_columns,
				error_keys=args.error_columns,
				time_key=args.time_column,
				time_start_key=args.time_start_key,
				cadence_key=args.cadence_key,
				channel_names=channel_names,
			)

			(
				feature_names,
				features,
				channel_metadata,
			) = extract_features(
				channels=channels
			)

			result_item = dict(
				item
			)

			result_item[
				"feats"
			] = [
				float(value)
				for value in features
			]

		output_items.append(
			result_item
		)

	payload[
		args.datalist_key
	] = output_items

	payload[
		"fextractor"
	] = {
		"backend": BACKEND_NAME,
		"modality": "timeseries",
		"model": MODEL_NAME,
		"model_version": MODEL_VERSION,
	}

	return payload

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


def parse_float(
	value,
):
	try:
		return float(
			value
		)

	except (
		TypeError,
		ValueError,
	):
		return np.nan
		
def read_wide_timeseries(
	filename,
	value_prefixes,
	error_prefixes=None,
	time_prefix=None,
	time_start_column=None,
	cadence_column=None,
	channel_names=None,
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

		rows = list(
			reader
		)

		if len(rows) != 1:
			raise ValueError(
				"Wide time-series layout expects exactly "
				"one row per input file"
			)

		fieldnames = reader.fieldnames
		row = rows[0]

	if not value_prefixes:
		raise ValueError(
			"value-prefixes must be specified "
			"for wide layout"
		)

	if channel_names is None:
		channel_names = list(
			value_prefixes
		)

	if (
		len(channel_names)
		!= len(value_prefixes)
	):
		raise ValueError(
			"channel-names must contain one name "
			"per value prefix"
		)

	if (
		error_prefixes is not None
		and len(error_prefixes)
		!= len(value_prefixes)
	):
		raise ValueError(
			"error-prefixes must contain one prefix "
			"per value prefix"
		)

	#################################
	## VALUE ARRAYS
	#################################

	value_arrays = []

	for prefix in value_prefixes:
		columns = indexed_columns(
			fieldnames,
			prefix,
		)

		if not columns:
			raise ValueError(
				"No wide value columns found "
				"with prefix '%s'"
				% prefix
			)

		values = [
			parse_float(
				row.get(
					name,
					"",
				)
			)
			for name in columns
		]

		value_arrays.append(
			np.asarray(
				values,
				dtype=np.float64,
			)
		)

	lengths = set(
		len(values)
		for values in value_arrays
	)

	if len(lengths) != 1:
		raise ValueError(
			"Wide value channels have different lengths: %s"
			% sorted(
				lengths
			)
		)

	n_time = lengths.pop()

	#################################
	## ERROR ARRAYS
	#################################

	error_arrays = None

	if error_prefixes is not None:
		error_arrays = []

		for prefix in error_prefixes:
			columns = indexed_columns(
				fieldnames,
				prefix,
			)

			if not columns:
				raise ValueError(
					"No wide error columns found "
					"with prefix '%s'"
					% prefix
				)

			if len(columns) != n_time:
				raise ValueError(
					"Wide error-channel length does not match "
					"value-channel length for prefix '%s': "
					"%d != %d"
					% (
						prefix,
						len(columns),
						n_time,
					)
				)

			errors = [
				parse_float(
					row.get(
						name,
						"",
					)
				)
				for name in columns
			]

			error_arrays.append(
				np.asarray(
					errors,
					dtype=np.float64,
				)
			)

	#################################
	## TIME ARRAY
	#################################

	time = None

	if time_prefix is not None:
		time_columns = indexed_columns(
			fieldnames,
			time_prefix,
		)

		if not time_columns:
			raise ValueError(
				"No wide time columns found "
				"with prefix '%s'"
				% time_prefix
			)

		if len(time_columns) != n_time:
			raise ValueError(
				"Wide time-column length does not match "
				"value-channel length: %d != %d"
				% (
					len(time_columns),
					n_time,
				)
			)

		time = np.asarray(
			[
				parse_float(
					row.get(
						name,
						"",
					)
				)
				for name in time_columns
			],
			dtype=np.float64,
		)

	elif (
		time_start_column is not None
		or cadence_column is not None
	):
		if (
			time_start_column is None
			or cadence_column is None
		):
			raise ValueError(
				"time-start-column and cadence-column "
				"must be supplied together"
			)

		if time_start_column not in fieldnames:
			raise ValueError(
				"Column '%s' not found"
				% time_start_column
			)

		if cadence_column not in fieldnames:
			raise ValueError(
				"Column '%s' not found"
				% cadence_column
			)

		time_start = parse_float(
			row[
				time_start_column
			]
		)

		cadence = parse_float(
			row[
				cadence_column
			]
		)

		if not np.isfinite(
			time_start
		):
			raise ValueError(
				"Wide time start must be finite"
			)

		if (
			not np.isfinite(cadence)
			or cadence <= 0.0
		):
			raise ValueError(
				"Wide cadence must be positive and finite"
			)

		time = (
			time_start
			+ np.arange(
				n_time,
				dtype=np.float64,
			)
			* cadence
		)

	#################################
	## NORMALIZED CHANNELS
	#################################

	channels = []

	for channel_index, values in enumerate(
		value_arrays
	):
		errors = None

		if error_arrays is not None:
			errors = error_arrays[
				channel_index
			]

		channels.append({
			"name": channel_names[
				channel_index
			],
			"values": values,
			"time": (
				None
				if time is None
				else time.copy()
			),
			"errors": errors,
			"value_source": value_prefixes[
				channel_index
			],
			"error_source": (
				None
				if error_prefixes is None
				else error_prefixes[
					channel_index
				]
			),
		})

	return channels
	
	
def build_long_channels(
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

	channels = []

	for channel_index, value_column in enumerate(
		value_columns
	):
		errors = None
		error_column = None

		if error_columns is not None:
			error_column = error_columns[
				channel_index
			]

			errors = columns[
				error_column
			]

		channels.append({
			"name": channel_names[
				channel_index
			],
			"values": columns[
				value_column
			],
			"time": (
				None
				if time is None
				else time.copy()
			),
			"errors": errors,
			"value_source": value_column,
			"error_source": error_column,
		})

	return channels	
	
	
def extract_file(
	filename,
	args,
):
	ext = os.path.splitext(
		filename
	)[1].lower()

	if ext != ".csv":
		raise ValueError(
			"FATS filepath entries currently support CSV files only, "
			"got '%s'"
			% ext
		)

	if args.timeseries_layout == "long":
		if not args.value_columns:
			raise ValueError(
				"Long layout requires --value-columns"
			)

	elif args.timeseries_layout == "wide":
		if not args.value_prefixes:
			raise ValueError(
				"Wide layout requires --value-prefixes"
			)
	

	if args.timeseries_layout == "long":
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

		required_columns = list(
			dict.fromkeys(
				required_columns
			)
		)

		columns = read_csv_columns(
			filename=filename,
			required_columns=required_columns,
		)

		channel_names = (
			args.channel_names
			if args.channel_names is not None
			else args.value_columns
		)

		channels = build_long_channels(
			columns=columns,
			time_column=args.time_column,
			value_columns=args.value_columns,
			error_columns=args.error_columns,
			channel_names=channel_names,
		)

		time_metadata = {
			"mode": (
				"explicit"
				if args.time_column is not None
				else "none"
			),
			"time_column": args.time_column,
		}

	elif args.timeseries_layout == "wide":
		channel_names = (
			args.channel_names
			if args.channel_names is not None
			else args.value_prefixes
		)

		channels = read_wide_timeseries(
			filename=filename,
			value_prefixes=args.value_prefixes,
			error_prefixes=args.error_prefixes,
			time_prefix=args.time_prefix,
			time_start_column=args.time_start_column,
			cadence_column=args.cadence_column,
			channel_names=channel_names,
		)

		if args.time_prefix is not None:
			time_metadata = {
				"mode": "explicit_prefix",
				"time_prefix": args.time_prefix,
			}

		elif (
			args.time_start_column is not None
			and args.cadence_column is not None
		):
			time_metadata = {
				"mode": "start_cadence",
				"time_start_column": (
					args.time_start_column
				),
				"cadence_column": (
					args.cadence_column
				),
			}

		else:
			time_metadata = {
				"mode": "none",
			}

	else:
		raise ValueError(
			"Unsupported time-series layout '%s'"
			% args.timeseries_layout
		)

	(
		feature_names,
		features,
		channel_metadata,
	) = extract_features(
		channels=channels,
	)

	return (
		feature_names,
		features,
		channel_metadata,
		time_metadata,
	)	
	
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


	bad_features = [
		name
		for name, value in zip(
			feature_names,
			feature_values,
		)
		if not np.isfinite(
			value
		)
	]

	if bad_features:
		print(
			"WARNING: Replacing non-finite FATS feature "
			"value(s) with 0: %s"
			% ", ".join(
				bad_features
			)
		)

		feature_values = np.asarray(
			feature_values,
			dtype=np.float64,
		)

		feature_values[
			~np.isfinite(
				feature_values
			)
		] = 0.0

	return (
		feature_names,
		feature_values,
		bad_features,
	)
	

	

def extract_features(
	channels,
):
	all_feature_names = []
	all_feature_values = []
	channel_metadata = []

	for channel in channels:
		channel_name = channel[
			"name"
		]

		values = np.asarray(
			channel[
				"values"
			],
			dtype=np.float64,
		)

		time = channel.get(
			"time"
		)

		errors = channel.get(
			"errors"
		)

		n_input = len(
			values
		)

		(
			time,
			values,
			errors,
		) = select_valid_samples(
			time=time,
			values=values,
			errors=errors,
		)

		(
			time,
			values,
			errors,
		) = sort_by_time(
			time=time,
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
			invalid_features,
		) = extract_channel(
			values=values,
			time=time,
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
			"value_source": channel.get(
				"value_source"
			),
			"error_source": channel.get(
				"error_source"
			),
			"n_input": int(
				n_input
			),
			"n_valid": int(
				n_valid
			),
			"n_features": int(
				len(features)
			),
			"invalid_features": list(
				invalid_features
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
	layout,
	time_metadata,
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
			"layout": layout,
			"time": time_metadata,
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

#################################
###     MAIN
#################################
def main(argv=None):
	parser = build_parser()

	args = parser.parse_args(
		argv
	)

	validate_args(
		args
	)

	ext = os.path.splitext(
		args.inputfile
	)[1].lower()

	print(
		"INFO: Starting FATS extractor"
	)

	print(
		"INFO: input='%s'"
		% args.inputfile
	)

	#################################
	## JSON DATALIST
	#################################

	if ext == ".json":
		print(
			"INFO: Processing JSON datalist key='%s'"
			% args.datalist_key
		)

		payload = extract_json_datalist(
			filename=args.inputfile,
			args=args,
		)

		with open(
			args.outfile,
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

		print(
			"INFO: Output written to '%s'"
			% args.outfile
		)

		return 0

	#################################
	## DIRECT CSV
	#################################

	(
		feature_names,
		features,
		channel_metadata,
		time_metadata,
	) = extract_file(
		args.inputfile,
		args,
	)

	save_output(
		filename=args.outfile,
		features=features,
		feature_names=feature_names,
		channel_metadata=channel_metadata,
		layout=args.timeseries_layout,
		time_metadata=time_metadata,
	)

	print(
		"INFO: FATS extraction completed "
		"channels=%d features=%d"
		% (
			len(channel_metadata),
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
