"""Tests for common time-series model-input sample policies."""

from __future__ import annotations

import numpy as np
import pytest

from fextractor.timeseries import (
	TimeSeries,
	build_input_sample_mask,
)


def make_mixed_provenance_series() -> TimeSeries:
	values = np.asarray(
		[
			[1.0],
			[2.0],
			[3.0],
			[4.0],
			[np.nan],
		],
		dtype=np.float32,
	)

	return TimeSeries(
		values=values,
		times=np.arange(
			5,
			dtype=np.float64,
		),
		observed_mask=np.asarray(
			[
				[True],
				[False],
				[False],
				[True],
				[False],
			],
			dtype=bool,
		),
		interpolated_mask=np.asarray(
			[
				[False],
				[True],
				[False],
				[False],
				[True],
			],
			dtype=bool,
		),
		predicted_mask=np.asarray(
			[
				[False],
				[False],
				[True],
				[False],
				[False],
			],
			dtype=bool,
		),
	)


def test_observed_policy_keeps_only_finite_observed_samples():
	series = make_mixed_provenance_series()

	mask = build_input_sample_mask(
		series,
		policy="observed",
	)

	assert mask[:, 0].tolist() == [
		True,
		False,
		False,
		True,
		False,
	]


def test_completed_policy_keeps_all_finite_prepared_samples():
	series = make_mixed_provenance_series()

	mask = build_input_sample_mask(
		series,
		policy="completed",
	)

	assert mask[:, 0].tolist() == [
		True,
		True,
		True,
		True,
		False,
	]


def test_completed_policy_does_not_change_provenance_masks():
	series = make_mixed_provenance_series()

	observed_before = series.observed_mask.copy()
	interpolated_before = series.interpolated_mask.copy()
	predicted_before = series.predicted_mask.copy()

	build_input_sample_mask(
		series,
		policy="completed",
	)

	np.testing.assert_array_equal(
		series.observed_mask,
		observed_before,
	)
	np.testing.assert_array_equal(
		series.interpolated_mask,
		interpolated_before,
	)
	np.testing.assert_array_equal(
		series.predicted_mask,
		predicted_before,
	)


def test_observed_policy_rejects_gp_regularized_series():
	series = TimeSeries(
		values=np.asarray(
			[
				[1.0],
				[2.0],
				[3.0],
			],
			dtype=np.float32,
		),
		times=np.asarray(
			[
				0.0,
				1.0,
				2.0,
			],
			dtype=np.float64,
		),
		observed_mask=np.zeros(
			(3, 1),
			dtype=bool,
		),
		predicted_mask=np.ones(
			(3, 1),
			dtype=bool,
		),
		metadata={
			"regularized": True,
			"regularization_method": "gp",
		},
	)

	with pytest.raises(
		ValueError,
		match="incompatible with GP regularization",
	):
		build_input_sample_mask(
			series,
			policy="observed",
		)


def test_completed_policy_accepts_gp_regularized_series():
	series = TimeSeries(
		values=np.asarray(
			[
				[1.0],
				[2.0],
				[3.0],
			],
			dtype=np.float32,
		),
		times=np.asarray(
			[
				0.0,
				1.0,
				2.0,
			],
			dtype=np.float64,
		),
		observed_mask=np.zeros(
			(3, 1),
			dtype=bool,
		),
		predicted_mask=np.ones(
			(3, 1),
			dtype=bool,
		),
		metadata={
			"regularized": True,
			"regularization_method": "gp",
		},
	)

	mask = build_input_sample_mask(
		series,
		policy="completed",
	)

	assert np.all(mask)
