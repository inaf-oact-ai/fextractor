import json

from fextractor.io import read_datalist, save_datalist_json


def test_datalist_roundtrip(tmp_path):
	entries = [{"sname": "source", "filepaths": ["image.fits"]}]
	path = tmp_path / "data.json"
	save_datalist_json(entries, path, key="data")
	assert read_datalist(path, key="data") == entries


def test_bare_list_is_accepted(tmp_path):
	entries = [{"filepaths": ["image.fits"]}]
	path = tmp_path / "data.json"
	path.write_text(json.dumps(entries), encoding="utf-8")
	assert read_datalist(path) == entries
