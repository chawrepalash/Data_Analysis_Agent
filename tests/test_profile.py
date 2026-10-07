import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.loader import DatasetStore, load_file
from src.tools.profile import profile_dataset, validate_data

FIXTURES = Path(__file__).parent / "fixtures"


def add(df: pd.DataFrame) -> tuple[DatasetStore, str]:
    store = DatasetStore()
    return store, store.add(df, "test")


def broken() -> tuple[DatasetStore, str]:
    return add(load_file(str(FIXTURES / "broken.csv")))


def issue_keys(result) -> set[tuple]:
    return {(i["type"], i["column"]) for i in result.data["issues"]}


def find_issue(result, type_: str, column: str | None) -> dict:
    return next(i for i in result.data["issues"] if i["type"] == type_ and i["column"] == column)


# ============================ profile_dataset ============================

def test_profile_broken_csv_numbers():
    store, dataset_id = broken()
    result = profile_dataset(store, dataset_id)
    assert result.ok
    assert result.data["rows"] == 6
    assert result.data["columns"] == 5
    assert result.data["duplicate_rows"] == 1
    info = {c["name"]: c for c in result.data["column_info"]}
    assert info["age"]["missing"] == 1
    assert info["country"]["missing"] == 1
    assert info["status"]["unique"] == 1
    assert info["id"]["kind"] == "numeric"
    assert info["age"]["kind"] == "text"  # mixed ints and text
    assert len(result.data["sample_rows"]) == 5


def test_profile_missing_percent():
    store, dataset_id = add(pd.DataFrame({"a": [1, None, 3, None]}))
    info = profile_dataset(store, dataset_id).data["column_info"][0]
    assert info["missing"] == 2
    assert info["missing_pct"] == 50.0


def test_profile_numeric_min_max():
    store, dataset_id = add(pd.DataFrame({"a": [3, 1, 2], "b": ["x", "y", "z"]}))
    info = {c["name"]: c for c in profile_dataset(store, dataset_id).data["column_info"]}
    assert info["a"]["min"] == 1.0 and info["a"]["max"] == 3.0
    assert "min" not in info["b"]


def test_profile_all_null_numeric_column_has_no_min_max():
    store, dataset_id = add(pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]}))
    info = {c["name"]: c for c in profile_dataset(store, dataset_id).data["column_info"]}
    assert "min" not in info["b"]


def test_profile_sample_rows_are_json_safe():
    store, dataset_id = add(pd.DataFrame({"a": [1.0, np.nan], "d": pd.to_datetime(["2025-01-01", None])}))
    result = profile_dataset(store, dataset_id)
    json.dumps(result.data)  # raises if anything is not JSON-safe
    assert result.data["sample_rows"][1]["a"] is None


def test_profile_caps_wide_frames():
    df = pd.DataFrame(np.zeros((2, 120)), columns=[f"c{i}" for i in range(120)])
    store, dataset_id = add(df)
    data = profile_dataset(store, dataset_id).data
    assert data["columns"] == 120
    assert len(data["column_info"]) == 100
    assert data["columns_truncated"] is True


def test_profile_unknown_dataset():
    result = profile_dataset(DatasetStore(), "nope")
    assert not result.ok
    assert "nope" in result.error


# ============================= validate_data =============================

def test_validate_broken_csv_finds_every_planted_problem():
    store, dataset_id = broken()
    result = validate_data(store, dataset_id)
    assert result.ok
    keys = issue_keys(result)
    assert ("mixed_types", "age") in keys
    assert ("constant_column", "status") in keys
    assert ("duplicate_rows", None) in keys
    assert ("date_like_text", "signup_date") in keys
    assert ("missing_values", "age") in keys
    assert ("missing_values", "country") in keys


def test_validate_broken_csv_details():
    store, dataset_id = broken()
    result = validate_data(store, dataset_id)
    assert "1 value(s) could not be parsed" in find_issue(result, "date_like_text", "signup_date")["detail"]
    assert find_issue(result, "mixed_types", "age")["severity"] == "high"
    assert find_issue(result, "duplicate_rows", None)["severity"] == "medium"


def test_validate_sorts_most_severe_first():
    store, dataset_id = broken()
    severities = [i["severity"] for i in validate_data(store, dataset_id).data["issues"]]
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    assert severities == sorted(severities, key=order.get)
    assert severities[0] == "high"


def test_validate_counts_match_issues():
    store, dataset_id = broken()
    result = validate_data(store, dataset_id)
    assert sum(result.data["counts"].values()) == len(result.data["issues"])


def test_validate_clean_frame_has_no_issues():
    store, dataset_id = add(pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]}))
    result = validate_data(store, dataset_id)
    assert result.ok and result.data["issues"] == []
    assert "No data quality issues" in result.summary


def test_validate_empty_column():
    store, dataset_id = add(pd.DataFrame({"a": [1, 2], "b": [None, None]}))
    result = validate_data(store, dataset_id)
    assert find_issue(result, "empty_column", "b")["severity"] == "high"


def test_validate_negative_values_only_in_non_negative_columns():
    store, dataset_id = add(pd.DataFrame({"units": [1, -2, 3], "temperature": [-5, 3, 4]}))
    keys = issue_keys(validate_data(store, dataset_id))
    assert ("negative_values", "units") in keys
    assert ("negative_values", "temperature") not in keys


def test_validate_numbers_stored_as_text():
    store, dataset_id = add(pd.DataFrame({"zip": ["110001", "560001", "400001"]}))
    result = validate_data(store, dataset_id)
    assert ("numbers_as_text", "zip") in issue_keys(result)
    assert ("date_like_text", "zip") not in issue_keys(result)


def test_validate_clean_dates_have_no_unparseable_note():
    store, dataset_id = add(pd.DataFrame({"d": ["2025-01-01", "2025-01-02", "2025-01-03"]}))
    issue = find_issue(validate_data(store, dataset_id), "date_like_text", "d")
    assert "could not" not in issue["detail"]


def test_validate_names_are_not_mistaken_for_dates():
    store, dataset_id = add(pd.DataFrame({"country": ["India", "USA", "UK", "Brazil"]}))
    assert validate_data(store, dataset_id).data["issues"] == []


def test_validate_high_cardinality_text():
    ids = [f"user_{i:03d}" for i in range(30)]
    store, dataset_id = add(pd.DataFrame({"user": ids}))
    issue = find_issue(validate_data(store, dataset_id), "high_cardinality_text", "user")
    assert issue["severity"] == "info"


@pytest.mark.parametrize(
    "n_rows, n_missing, expected",
    [(20, 10, "high"), (20, 2, "medium"), (200, 1, "low")],
)
def test_validate_missing_value_severity(n_rows, n_missing, expected):
    values = list(range(n_rows))
    for i in range(n_missing):
        values[i] = None
    store, dataset_id = add(pd.DataFrame({"a": pd.Series(values, dtype="float")}))
    assert find_issue(validate_data(store, dataset_id), "missing_values", "a")["severity"] == expected


def test_validate_small_duplicate_share_is_low():
    store, dataset_id = add(pd.DataFrame({"a": list(range(199)) + [0]}))
    assert find_issue(validate_data(store, dataset_id), "duplicate_rows", None)["severity"] == "low"


def test_validate_single_row_is_not_constant():
    store, dataset_id = add(pd.DataFrame({"a": [1]}))
    assert ("constant_column", "a") not in issue_keys(validate_data(store, dataset_id))


def test_validate_sales_like_frame():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "order_date": pd.date_range("2025-01-01", periods=n).strftime("%Y-%m-%d"),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "units": rng.integers(1, 20, n),
        }
    )
    df.loc[rng.choice(n, 20, replace=False), "region"] = None
    store, dataset_id = add(df)
    result = validate_data(store, dataset_id)
    assert find_issue(result, "missing_values", "region")["severity"] == "low"  # 4%
    assert ("date_like_text", "order_date") in issue_keys(result)
    assert not any(i["column"] == "units" for i in result.data["issues"])


def test_validate_unknown_dataset():
    assert not validate_data(DatasetStore(), "nope").ok


# =========================== read-only guarantee ===========================

@pytest.mark.parametrize("tool", [profile_dataset, validate_data])
def test_tools_do_not_modify_the_dataframe(tool):
    store, dataset_id = broken()
    before = store.get(dataset_id).copy()
    tool(store, dataset_id)
    pd.testing.assert_frame_equal(store.get(dataset_id), before)
