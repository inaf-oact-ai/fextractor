"""Aggregation of token-level time-series representations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


SUPPORTED_AGGREGATIONS = (
	"mean",
	"std",
	"max",
	"mean_std",
	"mean_max",
	"mean_std_max",
	"last",
	"reg",
	"flatten",
)


@dataclass
class TokenRepresentation:
	"""Contextual token representations returned by a model."""

	context_tokens: np.ndarray
	special_tokens: dict[str, np.ndarray] = field(
		default_factory=dict
	)


def _flatten_token_axes(
	tokens: np.ndarray,
) -> np.ndarray:
	"""Flatten all token-like axes while keeping embedding dimension."""

	tokens = np.asarray(
		tokens,
		dtype=np.float32,
	)

	if tokens.ndim < 2:
		raise ValueError(
			"Expected token array with at least two dimensions, "
			f"got {tokens.shape}"
		)

	return tokens.reshape(
		-1,
		tokens.shape[-1],
	)


def aggregate_token_representation(
	representation: TokenRepresentation,
	strategy: str = "mean_std",
) -> np.ndarray:
	"""Aggregate contextual tokens into one one-dimensional vector."""

	if strategy not in SUPPORTED_AGGREGATIONS:
		raise ValueError(
			f"Unsupported aggregation '{strategy}'. "
			f"Available: {', '.join(SUPPORTED_AGGREGATIONS)}"
		)

	tokens = _flatten_token_axes(
		representation.context_tokens
	)

	if tokens.shape[0] == 0:
		raise ValueError(
			"Cannot aggregate an empty token representation"
		)

	if strategy == "flatten":
		return tokens.reshape(-1).astype(
			np.float32,
			copy=False,
		)

	if strategy == "mean":
		return tokens.mean(
			axis=0,
			dtype=np.float64,
		).astype(np.float32)

	if strategy == "std":
		return tokens.std(
			axis=0,
			dtype=np.float64,
		).astype(np.float32)

	if strategy == "max":
		return tokens.max(
			axis=0,
		).astype(np.float32)

	if strategy == "last":
		return tokens[-1].astype(
			np.float32,
			copy=False,
		)

	if strategy == "reg":
		try:
			reg = representation.special_tokens[
				"reg"
			]

		except KeyError as exc:
			raise ValueError(
				"This representation does not provide "
				"a 'reg' token"
			) from exc

		reg = _flatten_token_axes(
			reg
		)

		return reg.mean(
			axis=0,
			dtype=np.float64,
		).astype(np.float32)

	mean = tokens.mean(
		axis=0,
		dtype=np.float64,
	).astype(np.float32)

	std = tokens.std(
		axis=0,
		dtype=np.float64,
	).astype(np.float32)

	maximum = tokens.max(
		axis=0,
	).astype(np.float32)

	if strategy == "mean_std":
		return np.concatenate([
			mean,
			std,
		])

	if strategy == "mean_max":
		return np.concatenate([
			mean,
			maximum,
		])

	if strategy == "mean_std_max":
		return np.concatenate([
			mean,
			std,
			maximum,
		])

	raise RuntimeError(
		f"Unhandled aggregation strategy '{strategy}'"
	)
