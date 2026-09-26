from pathlib import Path

import pytest

from fextractor.runner import _get_entry_filepath

def test_get_entry_filepath_singular():
	item = {
		"filepath": "/tmp/series.csv",
	}

	path = _get_entry_filepath(
		item
	)

	assert path == Path(
		"/tmp/series.csv"
	)


def test_get_entry_filepath_plural():
	item = {
		"filepaths": [
			"/tmp/a.csv",
			"/tmp/b.csv",
		],
	}

	path = _get_entry_filepath(
		item,
		filepath_index=1,
	)

	assert path == Path(
		"/tmp/b.csv"
	)


def test_get_entry_filepath_missing():
	with pytest.raises(
		KeyError,
		match="filepath",
	):
		_get_entry_filepath({})
