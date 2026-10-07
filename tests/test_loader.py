from pathlib import Path

import pandas as pd
import pytest

import config
from src.loader import (
    DataLoadError,
    DatasetNotFound,
    DatasetStore,
    load_bytes,
    load_file,
)
from src.schemas import error_result, ok_result

FIXTURES = Path(__file__).parent / "fixtures"


# ---------- loading ----------

def test_load_csv_bytes():
    df = load_bytes(b"a,b\n1,2\n3,4\n", "x.csv")
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


def test_load_csv_from_path(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    assert load_file(str(path)).shape == (1, 2)


def test_load_excel(tmp_path):
    path = tmp_path / "data.xlsx"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(path, index=False)
    df = load_file(str(path))
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


def test_empty_file_rejected():
    with pytest.raises(DataLoadError, match="empty"):
        load_bytes(b"", "x.csv")


def test_header_only_rejected():
    with pytest.raises(DataLoadError, match="no data rows"):
        load_bytes(b"a,b\n", "x.csv")


def test_whitespace_only_rejected():
    with pytest.raises(DataLoadError):
        load_bytes(b"   \n\n", "x.csv")


def test_unsupported_extension_rejected():
    with pytest.raises(DataLoadError, match="Unsupported"):
        load_bytes(b"hello", "notes.txt")


def test_oversized_file_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_FILE_BYTES", 10)
    with pytest.raises(DataLoadError, match="limit"):
        load_bytes(b"a,b\n1,2\n3,4\n5,6\n", "x.csv")


def test_oversized_file_rejected_before_reading(tmp_path, monkeypatch):
    path = tmp_path / "big.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    monkeypatch.setattr(config, "MAX_FILE_BYTES", 5)
    with pytest.raises(DataLoadError, match="limit"):
        load_file(str(path))


def test_missing_file_rejected():
    with pytest.raises(DataLoadError, match="Cannot open"):
        load_file("does_not_exist.csv")


def test_latin1_encoding_fallback():
    data = "name,city\nJosé,Zürich\n".encode("latin-1")
    df = load_bytes(data, "x.csv")
    assert df.loc[0, "name"] == "José"


def test_utf8_bom_header_is_clean():
    data = "﻿a,b\n1,2\n".encode("utf-8")
    assert list(load_bytes(data, "x.csv").columns) == ["a", "b"]


def test_column_names_are_stripped():
    df = load_bytes(b" a , b \n1,2\n", "x.csv")
    assert list(df.columns) == ["a", "b"]


def test_duplicate_column_names_are_made_unique():
    df = load_bytes(b"a,a\n1,2\n", "x.csv")
    assert len(set(df.columns)) == 2


def test_broken_fixture_loads_with_expected_problems():
    df = load_file(str(FIXTURES / "broken.csv"))
    assert df.shape == (6, 5)
    assert not pd.api.types.is_numeric_dtype(df["age"])  # mixed ints and text
    assert df["status"].nunique() == 1       # constant column
    assert df.duplicated().sum() == 1        # one duplicate row
    assert df["country"].isna().sum() == 1


# ---------- DatasetStore ----------

def test_store_add_and_get():
    store = DatasetStore()
    dataset_id = store.add(pd.DataFrame({"a": [1, 2]}), name="demo")
    assert store.get(dataset_id).shape == (2, 1)
    assert store.meta(dataset_id).name == "demo"


def test_store_unknown_id():
    with pytest.raises(DatasetNotFound):
        DatasetStore().get("nope")


def test_store_does_not_alias_the_input_frame():
    df = pd.DataFrame({"a": [1, 2]})
    store = DatasetStore()
    dataset_id = store.add(df)
    df.loc[0, "a"] = 99
    assert store.get(dataset_id).loc[0, "a"] == 1


def test_update_keeps_original_and_reset_restores_it():
    store = DatasetStore()
    dataset_id = store.add(pd.DataFrame({"a": [1, 2, 2]}))
    store.update(dataset_id, store.get(dataset_id).drop_duplicates())
    assert len(store.get(dataset_id)) == 2
    assert len(store.original(dataset_id)) == 3
    store.reset(dataset_id)
    assert len(store.get(dataset_id)) == 3


def test_reset_gives_independent_copy():
    store = DatasetStore()
    dataset_id = store.add(pd.DataFrame({"a": [1, 2]}))
    store.reset(dataset_id)
    store.get(dataset_id).loc[0, "a"] = 99
    assert store.original(dataset_id).loc[0, "a"] == 1


# ---------- ToolResult ----------

def test_ok_result():
    result = ok_result("done", rows=3)
    assert result.ok and result.data == {"rows": 3} and result.error is None


def test_error_result():
    result = error_result("bad column", hint="valid: a, b")
    assert not result.ok and result.hint == "valid: a, b"
