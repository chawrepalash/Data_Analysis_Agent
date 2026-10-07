"""Tests for propose_cleaning and apply_cleaning. No LLM, no API calls."""

import numpy as np
import pandas as pd
import pytest

from src.loader import DatasetStore, load_bytes
from src.tools.cleaning import apply_cleaning, propose_cleaning


def make_store(df: pd.DataFrame) -> tuple[DatasetStore, str]:
    """The one place tests add data to the store."""
    store = DatasetStore()
    return store, store.add(df, name="test")


@pytest.fixture
def broken_df() -> pd.DataFrame:
    """Mixed-type column, constant column, duplicate row, dates as text, gaps."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 8, 9],
            "score": ["10", "20", "x", "40", "50", "60", "70", "80", "80", "90"],
            "signup": [
                "2024-01-05", "2024-02-10", "2024-03-15", "2024-04-20", "2024-05-25",
                "2024-06-30", "2024-07-04", "2024-08-08", "2024-08-08", "2024-09-09",
            ],
            "source": ["web"] * 10,
            "age": [22, np.nan, 35, 41, 29, 33, 38, 27, 27, 45],
            "nickname": ["a", "b", None, "d", "e", "f", "g", "h", "h", "j"],
        }
    )


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [10.5, 20.5, 30.5, 40.5], "z": ["a", "b", "c", "d"]})


def fix_ids(result) -> set[str]:
    return {f["id"] for f in result.data["fixes"]}


# ---------------------------------------------------------------- propose

def test_propose_finds_expected_fixes(broken_df):
    store, ds = make_store(broken_df)
    result = propose_cleaning(store, ds)
    assert result.ok
    assert fix_ids(result) == {
        "drop_column:source",
        "to_numeric:score",
        "to_datetime:signup",
        "fill_median:age",
        "fill_unknown:nickname",
        "drop_duplicates",
    }
    for fix in result.data["fixes"]:
        assert set(fix) == {"id", "column", "action", "reason"}
        assert fix["reason"]


def test_propose_applies_nothing(broken_df):
    store, ds = make_store(broken_df)
    before = store.get(ds).copy()
    propose_cleaning(store, ds)
    pd.testing.assert_frame_equal(store.get(ds), before)


def test_propose_no_issues(clean_df):
    store, ds = make_store(clean_df)
    result = propose_cleaning(store, ds)
    assert result.ok
    assert result.data["fixes"] == []


def test_propose_unknown_dataset():
    result = propose_cleaning(DatasetStore(), "nope")
    assert not result.ok and result.hint


def test_propose_drops_very_sparse_column():
    df = pd.DataFrame({"a": range(10), "sparse": [1.0] + [np.nan] * 9})
    store, ds = make_store(df)
    assert "drop_column:sparse" in fix_ids(propose_cleaning(store, ds))


def test_propose_all_null_column():
    df = pd.DataFrame({"a": [1, 2, 3], "empty": [np.nan, np.nan, np.nan]})
    store, ds = make_store(df)
    assert "drop_column:empty" in fix_ids(propose_cleaning(store, ds))


def test_propose_empty_dataframe():
    store, ds = make_store(pd.DataFrame())
    result = propose_cleaning(store, ds)
    assert result.ok and result.data["fixes"] == []


def test_propose_zero_rows():
    store, ds = make_store(pd.DataFrame({"a": pd.Series([], dtype=float)}))
    result = propose_cleaning(store, ds)
    assert result.ok and result.data["fixes"] == []


def test_propose_single_column():
    store, ds = make_store(pd.DataFrame({"a": [1, 2, 3]}))
    assert propose_cleaning(store, ds).ok


def test_duplicate_column_names_return_error():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    store, ds = make_store(df)
    result = propose_cleaning(store, ds)
    assert not result.ok and "Duplicate column" in result.error


# ------------------------------------------------------------------ apply

def test_apply_all_fixes(broken_df):
    store, ds = make_store(broken_df)
    ids = list(fix_ids(propose_cleaning(store, ds)))
    result = apply_cleaning(store, ds, ids)
    assert result.ok, result.error

    assert result.data["dataset_id"] == ds
    cleaned = store.get(ds)

    # shapes: 1 duplicate row and 1 constant column removed
    assert result.data["shape_before"] == [10, 6]
    assert result.data["shape_after"] == [9, 5] == list(cleaned.shape)
    assert "source" not in cleaned.columns
    assert not cleaned.duplicated().any()

    # numbers computed by hand with plain pandas
    expected_score = pd.to_numeric(
        broken_df["score"], errors="coerce").drop_duplicates()
    assert cleaned["score"].isna().sum() == 1
    assert cleaned["score"].sum() == expected_score.sum()
    assert pd.api.types.is_datetime64_any_dtype(cleaned["signup"])

    # fills
    assert cleaned["age"].isna().sum() == 0
    assert cleaned.loc[1, "age"] == broken_df["age"].median()
    assert cleaned["nickname"].isna().sum() == 0
    assert (cleaned["nickname"] == "Unknown").sum() == 1


def test_apply_never_mutates_original(broken_df):
    store, ds = make_store(broken_df)
    snapshot = broken_df.copy(deep=True)
    ids = list(fix_ids(propose_cleaning(store, ds)))
    apply_cleaning(store, ds, ids)
    pd.testing.assert_frame_equal(store.original(ds), snapshot)
    pd.testing.assert_frame_equal(broken_df, snapshot)


def test_reset_restores_original_after_cleaning(broken_df):
    store, ds = make_store(broken_df)
    snapshot = broken_df.copy(deep=True)
    ids = list(fix_ids(propose_cleaning(store, ds)))
    apply_cleaning(store, ds, ids)
    assert store.get(ds).shape != snapshot.shape
    store.reset(ds)
    pd.testing.assert_frame_equal(store.get(ds), snapshot)


def test_failed_apply_leaves_dataset_unchanged(broken_df):
    store, ds = make_store(broken_df)
    snapshot = broken_df.copy(deep=True)
    apply_cleaning(store, ds, ["drop_duplicates", "bogus"])
    apply_cleaning(store, ds, [])
    pd.testing.assert_frame_equal(store.get(ds), snapshot)


def test_applied_fix_ids_are_no_longer_valid(broken_df):
    store, ds = make_store(broken_df)
    assert apply_cleaning(store, ds, ["drop_duplicates"]).ok
    again = apply_cleaning(store, ds, ["drop_duplicates"])
    assert not again.ok and "Unknown fix id" in again.error


def test_end_to_end_with_real_csv_loader():
    csv = (
        "id,score,signup,source,age\n"
        "1,10,2024-01-05,web,22\n2,20,2024-02-10,web,\n3,x,2024-03-15,web,35\n"
        "4,40,2024-04-20,web,41\n5,50,2024-05-25,web,29\n5,50,2024-05-25,web,29\n"
    ).encode()
    store = DatasetStore()
    ds = store.add(load_bytes(csv, "broken.csv"), name="broken")
    proposed = propose_cleaning(store, ds)
    assert proposed.ok and proposed.data["fixes"]
    result = apply_cleaning(store, ds, [f["id"]
                            for f in proposed.data["fixes"]])
    assert result.ok, result.error
    assert store.get(ds).shape[0] == 5  # one duplicate row removed


def test_apply_subset_only(broken_df):
    store, ds = make_store(broken_df)
    result = apply_cleaning(store, ds, ["drop_duplicates"])
    assert result.ok
    assert result.data["shape_after"] == [9, 6]
    assert [c["fix_id"]
            for c in result.data["change_log"]] == ["drop_duplicates"]
    assert result.data["change_log"][0]["rows_affected"] == 1
    cleaned = store.get(ds)
    assert cleaned["score"].dtype == broken_df["score"].dtype  # untouched


def test_apply_change_log_entries(broken_df):
    store, ds = make_store(broken_df)
    result = apply_cleaning(store, ds, ["fill_median:age", "to_numeric:score"])
    log = {c["fix_id"]: c for c in result.data["change_log"]}
    assert log["fill_median:age"]["rows_affected"] == 1
    assert log["to_numeric:score"]["rows_affected"] == 10
    assert "1 value(s) became missing" in log["to_numeric:score"]["change"]


def test_apply_ignores_repeated_ids(broken_df):
    store, ds = make_store(broken_df)
    result = apply_cleaning(store, ds, ["drop_duplicates", "drop_duplicates"])
    assert result.ok and len(result.data["change_log"]) == 1


def test_apply_unknown_fix_id_lists_valid_ids(broken_df):
    store, ds = make_store(broken_df)
    result = apply_cleaning(store, ds, ["bogus"])
    assert not result.ok
    assert "bogus" in result.error
    assert "drop_duplicates" in result.hint and "to_numeric:score" in result.hint


def test_apply_empty_fix_list(broken_df):
    store, ds = make_store(broken_df)
    result = apply_cleaning(store, ds, [])
    assert not result.ok and result.hint


def test_apply_when_nothing_to_clean(clean_df):
    store, ds = make_store(clean_df)
    result = apply_cleaning(store, ds, ["drop_duplicates"])
    assert not result.ok
    assert "none" in result.hint


def test_apply_unknown_dataset():
    result = apply_cleaning(DatasetStore(), "nope", ["drop_duplicates"])
    assert not result.ok


def test_apply_refuses_to_remove_every_column():
    df = pd.DataFrame({"a": ["x", "x", "x"], "b": [1, 1, 1]})
    store, ds = make_store(df)
    ids = ["drop_column:a", "drop_column:b"]
    assert set(ids) <= fix_ids(propose_cleaning(store, ds))
    result = apply_cleaning(store, ds, ids)
    assert not result.ok and "every column" in result.error


def test_cleaned_data_can_be_cleaned_again(broken_df):
    store, ds = make_store(broken_df)
    ids = list(fix_ids(propose_cleaning(store, ds)))
    apply_cleaning(store, ds, ids)
    again = propose_cleaning(store, ds)
    assert again.ok
    assert "drop_duplicates" not in fix_ids(again)
    assert "drop_column:source" not in fix_ids(again)


def test_code_like_mixed_column_is_not_coerced_to_numbers():
    """Like Titanic Ticket: many non-numeric codes, so converting would destroy data."""
    values = ["113803", "A/5 21171", "373450", "PC 17599", "330877", "STON/O2 3101282",
              "17463", "349909", "347742", "237736"]
    df = pd.DataFrame({"ticket": values})
    store, ds = make_store(df)
    assert "to_numeric:ticket" not in fix_ids(propose_cleaning(store, ds))
    assert apply_cleaning(store, ds, ["to_numeric:ticket"]).ok is False
    pd.testing.assert_frame_equal(store.get(ds), df)
