"""Tests for describe_stats, correlation_analysis and detect_outliers. No LLM."""

import json

import numpy as np
import pandas as pd
import pytest

from src.loader import DatasetStore
from src.tools.stats import correlation_analysis, describe_stats, detect_outliers


def make_store(df: pd.DataFrame) -> tuple[DatasetStore, str]:
    store = DatasetStore()
    return store, store.add(df, name="test")


def assert_json_safe(result) -> None:
    """No NaN or inf may leak into results."""
    json.dumps(result.data, allow_nan=False)


@pytest.fixture
def people() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [22.0, 35.0, np.nan, 41.0, 29.0, 33.0, 38.0, 27.0],
            "fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.07],
            "city": ["A", "B", "A", "C", "A", "B", "A", None],
            "alive": [True, False, True, True, False, True, True, False],
            "joined": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", None,
                    "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01"]
            ),
        }
    )


# ============================================================ describe_stats

def test_describe_numeric_matches_pandas(people):
    store, ds = make_store(people)
    result = describe_stats(store, ds, ["age"])
    assert result.ok
    s = result.data["columns"][0]
    age = people["age"].dropna()
    assert s["kind"] == "numeric"
    assert s["count"] == 7 and s["missing"] == 1
    assert s["mean"] == round(age.mean(), 4)
    assert s["median"] == age.median()
    assert s["std"] == round(age.std(), 4)
    assert s["min"] == 22.0 and s["max"] == 41.0
    assert s["q1"] == round(age.quantile(0.25), 4)
    assert s["q3"] == round(age.quantile(0.75), 4)


def test_describe_categorical_top_values(people):
    store, ds = make_store(people)
    s = describe_stats(store, ds, ["city"]).data["columns"][0]
    assert s["kind"] == "text"
    assert s["unique"] == 3
    assert s["top_values"][0] == {"value": "A", "count": 4}
    assert [t["count"] for t in s["top_values"]] == [4, 2, 1]


def test_describe_boolean_is_categorical(people):
    store, ds = make_store(people)
    s = describe_stats(store, ds, ["alive"]).data["columns"][0]
    assert s["kind"] == "boolean"
    assert s["top_values"][0] == {"value": "True", "count": 5}


def test_describe_datetime_min_max(people):
    store, ds = make_store(people)
    s = describe_stats(store, ds, ["joined"]).data["columns"][0]
    assert s["kind"] == "datetime"
    assert s["min"].startswith(
        "2024-01-01") and s["max"].startswith("2024-08-01")


def test_describe_all_columns_by_default_and_json_safe(people):
    store, ds = make_store(people)
    result = describe_stats(store, ds)
    assert [c["column"]
            for c in result.data["columns"]] == list(people.columns)
    assert_json_safe(result)
    assert describe_stats(
        store, ds, []).data["columns"] == result.data["columns"]


def test_describe_unknown_column_lists_valid_columns(people):
    store, ds = make_store(people)
    result = describe_stats(store, ds, ["age", "nope"])
    assert not result.ok
    assert "nope" in result.error
    assert "age" in result.hint and "city" in result.hint


def test_describe_all_null_numeric_column():
    store, ds = make_store(pd.DataFrame(
        {"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]}))
    result = describe_stats(store, ds, ["a"])
    s = result.data["columns"][0]
    assert result.ok and s["mean"] is None and s["max"] is None and s["count"] == 0
    assert_json_safe(result)


def test_describe_single_row_has_no_std():
    store, ds = make_store(pd.DataFrame({"a": [5.0]}))
    result = describe_stats(store, ds)
    assert result.data["columns"][0]["std"] is None
    assert result.data["columns"][0]["mean"] == 5.0
    assert_json_safe(result)


def test_describe_infinite_values_do_not_leak():
    store, ds = make_store(pd.DataFrame({"a": [1.0, 2.0, np.inf, 3.0]}))
    result = describe_stats(store, ds)
    assert result.data["columns"][0]["max"] == 3.0
    assert_json_safe(result)


def test_describe_caps_columns():
    df = pd.DataFrame({f"c{i}": range(3) for i in range(40)})
    store, ds = make_store(df)
    result = describe_stats(store, ds)
    assert len(result.data["columns"]
               ) == 30 and result.data["truncated"] is True


def test_describe_duplicate_column_names():
    store, ds = make_store(pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"]))
    result = describe_stats(store, ds)
    assert not result.ok and "Duplicate column" in result.error


def test_describe_unknown_dataset():
    assert not describe_stats(DatasetStore(), "nope").ok


def test_describe_does_not_mutate(people):
    store, ds = make_store(people)
    before = store.get(ds).copy()
    describe_stats(store, ds)
    pd.testing.assert_frame_equal(store.get(ds), before)


# ====================================================== correlation_analysis

@pytest.fixture
def corr_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    return pd.DataFrame(
        {
            "x": x,
            "double_x": 2 * x + 1,                      # +1.0 with x
            "neg_x": -x + rng.normal(scale=0.1, size=200),  # about -1.0 with x
            "noise": rng.normal(size=200),
            "label": ["a", "b"] * 100,
            "flag": [True, False] * 100,
        }
    )


def test_corr_matches_pandas(corr_df):
    store, ds = make_store(corr_df)
    result = correlation_analysis(store, ds)
    assert result.ok
    expected = corr_df[["x", "double_x", "neg_x", "noise"]].corr()
    for a in expected.columns:
        for b in expected.columns:
            assert result.data["matrix"][a][b] == round(expected.loc[a, b], 4)
    assert result.data["columns"] == ["x", "double_x",
                                      "neg_x", "noise"]  # text and bool excluded


def test_corr_matrix_symmetric_with_unit_diagonal(corr_df):
    store, ds = make_store(corr_df)
    m = correlation_analysis(store, ds).data["matrix"]
    for a in m:
        assert m[a][a] == 1.0
        for b in m:
            assert m[a][b] == m[b][a]


def test_corr_top_pairs_sorted_unique_and_capped(corr_df):
    store, ds = make_store(corr_df)
    pairs = correlation_analysis(store, ds).data["top_pairs"]
    assert len(pairs) == 5  # 4 columns -> 6 pairs, capped at 5
    strengths = [abs(p["correlation"]) for p in pairs]
    assert strengths == sorted(strengths, reverse=True)
    assert {pairs[0]["a"], pairs[0]["b"]} == {"x", "double_x"}
    assert pairs[0]["correlation"] == 1.0
    keys = [frozenset((p["a"], p["b"])) for p in pairs]
    assert len(set(keys)) == len(keys)


def test_corr_summary_quotes_strongest_pair(corr_df):
    store, ds = make_store(corr_df)
    assert "+1.00" in correlation_analysis(store, ds).summary


def test_corr_spearman_differs_from_pearson_on_nonlinear_monotonic():
    x = np.arange(1, 21, dtype=float)
    store, ds = make_store(pd.DataFrame({"x": x, "y": x**5}))
    pearson = correlation_analysis(
        store, ds, method="pearson").data["top_pairs"][0]["correlation"]
    spearman = correlation_analysis(
        store, ds, method="spearman").data["top_pairs"][0]["correlation"]
    assert spearman == 1.0 and pearson < 1.0


def test_corr_explicit_columns(corr_df):
    store, ds = make_store(corr_df)
    result = correlation_analysis(store, ds, columns=["x", "noise"])
    assert result.data["columns"] == [
        "x", "noise"] and len(result.data["top_pairs"]) == 1


def test_corr_fewer_than_two_numeric_columns():
    store, ds = make_store(pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    result = correlation_analysis(store, ds)
    assert not result.ok
    assert "at least 2" in result.error and "a" in result.hint


def test_corr_no_numeric_columns_at_all():
    store, ds = make_store(pd.DataFrame({"b": ["x", "y"], "c": ["u", "v"]}))
    result = correlation_analysis(store, ds)
    assert not result.ok and "(none)" in result.hint


def test_corr_non_numeric_requested_column(corr_df):
    store, ds = make_store(corr_df)
    result = correlation_analysis(store, ds, columns=["x", "label"])
    assert not result.ok and "label" in result.error and "x" in result.hint


def test_corr_unknown_column_lists_valid(corr_df):
    store, ds = make_store(corr_df)
    result = correlation_analysis(store, ds, columns=["x", "zzz"])
    assert not result.ok and "zzz" in result.error and "x" in result.hint


def test_corr_unknown_method(corr_df):
    store, ds = make_store(corr_df)
    result = correlation_analysis(store, ds, method="magic")
    assert not result.ok and "pearson" in result.hint


def test_corr_constant_column_gives_none_not_nan():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "const": [
                      5.0] * 4, "b": [2.0, 4.0, 6.0, 8.0]})
    store, ds = make_store(df)
    result = correlation_analysis(store, ds)
    assert result.ok
    assert result.data["matrix"]["a"]["const"] is None
    assert all("const" not in (p["a"], p["b"])
               for p in result.data["top_pairs"])
    assert_json_safe(result)


def test_corr_all_pairs_undefined():
    store, ds = make_store(pd.DataFrame({"a": [1.0] * 4, "b": [2.0] * 4}))
    result = correlation_analysis(store, ds)
    assert result.ok and result.data["top_pairs"] == []
    assert "No correlations" in result.summary


def test_corr_caps_columns():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"c{i}": rng.normal(size=30) for i in range(20)})
    store, ds = make_store(df)
    result = correlation_analysis(store, ds)
    assert len(result.data["columns"]
               ) == 15 and result.data["truncated"] is True


def test_corr_with_missing_values():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0], "b": [
                      2.0, 4.0, 6.0, np.nan, 10.0]})
    store, ds = make_store(df)
    result = correlation_analysis(store, ds)
    assert result.ok and result.data["top_pairs"][0]["correlation"] == 1.0
    assert_json_safe(result)


# ========================================================== detect_outliers

@pytest.fixture
def one_outlier() -> pd.DataFrame:
    return pd.DataFrame({"v": [1, 2, 3, 4, 5, 100], "tag": list("abcdef")})


def test_iqr_hand_checked(one_outlier):
    # Q1=2.25, Q3=4.75, IQR=2.5 -> bounds -1.5 and 8.5
    store, ds = make_store(one_outlier)
    result = detect_outliers(store, ds, "v")
    assert result.ok
    d = result.data
    assert d["lower_bound"] == -1.5 and d["upper_bound"] == 8.5
    assert d["outlier_count"] == 1 and d["above_upper"] == 1 and d["below_lower"] == 0
    assert d["examples"] == [{"_row": 5, "v": 100.0, "tag": "f"}]
    assert d["method"] == "iqr" and d["threshold"] == 1.5
    assert "1 outlier" in result.summary
    assert_json_safe(result)


def test_iqr_low_outliers_counted_below():
    store, ds = make_store(pd.DataFrame({"v": [-100, 10, 11, 12, 13, 14, 15]}))
    d = detect_outliers(store, ds, "v").data
    assert d["below_lower"] == 1 and d["above_upper"] == 0 and d["outlier_count"] == 1


def test_zscore_matches_pandas():
    values = list(range(10, 30)) + [1000]
    store, ds = make_store(pd.DataFrame({"v": values}))
    result = detect_outliers(store, ds, "v", method="zscore")
    s = pd.Series(values, dtype=float)
    z = (s - s.mean()) / s.std()
    assert result.data["threshold"] == 3.0
    assert result.data["outlier_count"] == int((z.abs() > 3).sum()) == 1
    assert result.data["upper_bound"] == round(s.mean() + 3 * s.std(), 4)
    assert result.data["examples"][0]["v"] == 1000.0


def test_threshold_changes_the_count(one_outlier):
    store, ds = make_store(one_outlier)
    assert detect_outliers(
        store, ds, "v", threshold=1.5).data["outlier_count"] == 1
    assert detect_outliers(
        store, ds, "v", threshold=100).data["outlier_count"] == 0


def test_no_outliers_in_uniform_data():
    store, ds = make_store(pd.DataFrame({"v": list(range(1, 21))}))
    d = detect_outliers(store, ds, "v").data
    assert d["outlier_count"] == 0 and d["examples"] == []


def test_examples_capped_and_most_extreme_first():
    values = [10] * 30 + [200, 300, 400, 500, 600, 700, 800]
    store, ds = make_store(pd.DataFrame({"v": values}))
    d = detect_outliers(store, ds, "v").data
    assert d["outlier_count"] == 7
    assert [e["v"] for e in d["examples"]] == [
        800.0, 700.0, 600.0, 500.0, 400.0]


def test_missing_values_are_ignored():
    store, ds = make_store(pd.DataFrame(
        {"v": [1, 2, 3, 4, 5, 100, np.nan, np.nan]}))
    d = detect_outliers(store, ds, "v").data
    assert d["n_values"] == 6 and d["outlier_count"] == 1


def test_infinite_values_are_ignored():
    store, ds = make_store(pd.DataFrame({"v": [1, 2, 3, 4, 5, 100, np.inf]}))
    result = detect_outliers(store, ds, "v")
    assert result.data["n_values"] == 6
    assert_json_safe(result)


def test_constant_column_zscore_has_no_outliers():
    store, ds = make_store(pd.DataFrame({"v": [0.1] * 10}))
    result = detect_outliers(store, ds, "v", method="zscore")
    assert result.ok and result.data["outlier_count"] == 0
    assert_json_safe(result)


def test_non_numeric_column_is_an_error_with_hint():
    store, ds = make_store(pd.DataFrame(
        {"v": [1, 2, 3, 4], "name": list("abcd")}))
    result = detect_outliers(store, ds, "name")
    assert not result.ok and "not numeric" in result.error and "v" in result.hint


def test_boolean_column_is_not_numeric():
    store, ds = make_store(pd.DataFrame(
        {"flag": [True, False, True, False], "v": [1, 2, 3, 4]}))
    assert not detect_outliers(store, ds, "flag").ok


def test_unknown_column_lists_valid_columns(one_outlier):
    store, ds = make_store(one_outlier)
    result = detect_outliers(store, ds, "missing")
    assert not result.ok and "v" in result.hint and "tag" in result.hint


def test_too_few_values():
    store, ds = make_store(pd.DataFrame({"v": [1.0, 2.0, np.nan, np.nan]}))
    assert not detect_outliers(store, ds, "v").ok


def test_all_null_column():
    store, ds = make_store(pd.DataFrame({"v": [np.nan] * 6}))
    assert not detect_outliers(store, ds, "v").ok


def test_bad_method_and_threshold(one_outlier):
    store, ds = make_store(one_outlier)
    assert not detect_outliers(store, ds, "v", method="mad").ok
    assert not detect_outliers(store, ds, "v", threshold=0).ok
    assert not detect_outliers(store, ds, "v", threshold=-2).ok
    assert not detect_outliers(store, ds, "v", threshold=float("inf")).ok


def test_example_rows_are_capped_in_width():
    df = pd.DataFrame({f"c{i}": [1, 2, 3, 4, 5, 100] for i in range(30)})
    store, ds = make_store(df)
    example = detect_outliers(store, ds, "c0").data["examples"][0]
    assert len(example) == 1 + 12  # _row plus 12 columns


def test_duplicate_index_labels_are_handled():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 100]}, index=[0, 0, 1, 1, 2, 2])
    store, ds = make_store(df)
    d = detect_outliers(store, ds, "v").data
    assert d["outlier_count"] == 1 and d["examples"][0]["_row"] == 5


def test_outliers_unknown_dataset():
    assert not detect_outliers(DatasetStore(), "nope", "v").ok


def test_outliers_does_not_mutate(one_outlier):
    store, ds = make_store(one_outlier)
    before = store.get(ds).copy()
    detect_outliers(store, ds, "v")
    pd.testing.assert_frame_equal(store.get(ds), before)
