import pytest
import pandas as pd
from src.loader import DatasetStore
from src.tools.query import group_aggregate
from src.tools.charts import create_chart


@pytest.fixture
def populated_store():
    store = DatasetStore()
    df = pd.DataFrame({
        "category": ["A", "A", "B", "B", "C"],
        "sales": [100.0, 150.0, 200.0, 50.0, 300.0],
        "quantity": [1, 2, 2, 1, 3],
    })
    dataset_id = store.add(df, "sales_data")
    return store, dataset_id


def test_group_aggregate_happy_path(populated_store):
    store, dataset_id = populated_store
    result = group_aggregate(
        store=store,
        dataset_id=dataset_id,
        group_by=["category"],
        metrics=[{"column": "sales", "agg": "sum"}],
        sort_by="sales_sum",
        ascending=False,
    )
    assert result.ok is True
    assert result.data["total_groups"] == 3
    assert result.data["rows"][0]["category"] == "C"
    assert result.data["rows"][0]["sales_sum"] == 300.0


def test_group_aggregate_with_filter(populated_store):
    store, dataset_id = populated_store
    result = group_aggregate(
        store=store,
        dataset_id=dataset_id,
        group_by=["category"],
        metrics=[{"column": "sales", "agg": "mean"}],
        filters=[{"column": "quantity", "op": ">", "value": 1}],
    )
    assert result.ok is True
    assert result.data["total_groups"] == 3  # categories A, B, C each have row with quantity > 1


def test_group_aggregate_unknown_column(populated_store):
    store, dataset_id = populated_store
    result = group_aggregate(
        store=store,
        dataset_id=dataset_id,
        group_by=["non_existent"],
        metrics=[{"column": "sales", "agg": "sum"}],
    )
    assert result.ok is False
    assert "Unknown column" in result.error
    assert "Valid columns" in result.hint


def test_create_chart_bar_success(populated_store):
    store, dataset_id = populated_store
    result = create_chart(
        store=store,
        dataset_id=dataset_id,
        chart_type="bar",
        x="category",
        y="sales",
        aggregation="sum",
    )
    assert result.ok is True
    assert "chart_json" in result.data
    assert result.data["chart_type"] == "bar"


def test_create_chart_missing_required_y(populated_store):
    store, dataset_id = populated_store
    result = create_chart(
        store=store,
        dataset_id=dataset_id,
        chart_type="scatter",
        x="category",
        y=None,
    )
    assert result.ok is False
    assert "requires both 'x' and 'y'" in result.error