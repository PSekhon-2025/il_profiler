from il_rag.json_utils import extract_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_with_prose_around_it():
    assert extract_json('Sure! Here you go: {"a": 1} Hope that helps.') == {"a": 1}


def test_garbage_returns_none():
    assert extract_json("no json here") is None


def test_truncated_json_returns_none():
    assert extract_json('{"a": 1, "b": [1, 2') is None
