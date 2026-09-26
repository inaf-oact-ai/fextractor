import numpy as np
import pytest

from fextractor.timeseries import (
	TokenRepresentation,
	aggregate_token_representation,
)


def test_mean_std_aggregation():
	tokens = np.asarray(
		[
			[
				[1.0, 2.0],
				[3.0, 4.0],
			],
		],
		dtype=np.float32,
	)

	representation = TokenRepresentation(
		context_tokens=tokens,
	)

	features = aggregate_token_representation(
		representation,
		strategy="mean_std",
	)

	assert features.shape == (
		4,
	)

	np.testing.assert_allclose(
		features[:2],
		[
			2.0,
			3.0,
		],
	)


def test_reg_aggregation():
	context = np.zeros(
		(
			2,
			3,
			4,
		),
		dtype=np.float32,
	)

	reg = np.asarray(
		[
			[
				[
					1.0,
					2.0,
					3.0,
					4.0,
				],
			],
			[
				[
					3.0,
					4.0,
					5.0,
					6.0,
				],
			],
		],
		dtype=np.float32,
	)

	representation = TokenRepresentation(
		context_tokens=context,
		special_tokens={
			"reg": reg,
		},
	)

	features = aggregate_token_representation(
		representation,
		strategy="reg",
	)

	np.testing.assert_allclose(
		features,
		[
			2.0,
			3.0,
			4.0,
			5.0,
		],
	)


def test_flatten_aggregation():
	tokens = np.asarray(
		[
			[
				[1.0, 2.0],
				[3.0, 4.0],
			],
		],
		dtype=np.float32,
	)

	representation = TokenRepresentation(
		context_tokens=tokens,
	)

	features = aggregate_token_representation(
		representation,
		strategy="flatten",
	)

	np.testing.assert_allclose(
		features,
		[
			1.0,
			2.0,
			3.0,
			4.0,
		],
	)


def test_unknown_aggregation_is_rejected():
	representation = TokenRepresentation(
		context_tokens=np.zeros(
			(
				1,
				2,
				3,
			),
			dtype=np.float32,
		),
	)

	with pytest.raises(
		ValueError,
		match="Unsupported aggregation",
	):
		aggregate_token_representation(
			representation,
			strategy="unknown",
		)
