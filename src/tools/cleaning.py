"""Tools 3 and 4: propose_cleaning and apply_cleaning.

propose_cleaning only reads. apply_cleaning works on a copy and saves the result
with store.update(), which replaces the CURRENT version only. The original
upload stays in the store, so store.reset() brings it back.

Fix ids are built from the data (for example "to_numeric:score"), so
apply_cleaning can rebuild the same proposals from the dataset itself and no
proposal state has to be stored between the two calls.
"""

import warnings

import pandas as pd

from src.loader import DatasetStore
from src.schemas import ToolResult, error_result, ok_result
from src.tools.common import MAX_HINT_COLUMNS, column_kind, fetch
from src.tools.profile import validate_data

MISSING_DROP_PCT = 40.0  # columns missing at least this much are proposed for dropping
MAX_FIXES = 50
# A mixed column is only coerced to numbers when a small share of values is not
# numeric (typos, "N/A"). Above this it is more likely a code or identifier
# (e.g. Titanic Ticket: "A/5 21171", "113803"), and coercing would destroy data.
MAX_COERCE_SHARE = 0.15

# Apply order: remove columns, convert types, fill gaps, drop duplicates last.
ACTION_ORDER = {
    "drop_column": 0,
    "to_numeric": 1,
    "to_datetime": 1,
    "fill_median": 2,
    "fill_unknown": 2,
    "drop_duplicates": 3,
}


# --------------------------------------------------------------------------
# Building proposals
# --------------------------------------------------------------------------

def _fix(action: str, column: str | None, reason: str) -> dict:
    fix_id = action if column is None else f"{action}:{column}"
    return {"id": fix_id, "column": column, "action": action, "reason": reason}


def _real_column(df: pd.DataFrame, label: str):
    """Map the string label used in fix ids back to the dataframe's column."""
    for col in df.columns:
        if str(col) == label:
            return col
    return None


def _non_numeric_count(series: pd.Series) -> int:
    text = series.dropna().astype(str).str.strip()
    return int(pd.to_numeric(text, errors="coerce").isna().sum())


def _build_fixes(df: pd.DataFrame, issues: list[dict]) -> list[dict]:
    fixes: list[dict] = []
    handled: set[str] = set()  # columns that already have a structural or type fix
    by_column: dict[str, dict[str, dict]] = {}
    for issue in issues:
        if issue["column"] is not None:
            by_column.setdefault(issue["column"], {})[issue["type"]] = issue

    for label, found in by_column.items():
        if "empty_column" in found:
            fixes.append(_fix("drop_column", label, "Every value is missing, so the column carries no information."))
            handled.add(label)
        elif "constant_column" in found:
            fixes.append(_fix("drop_column", label, "Every value is identical, so the column cannot explain anything."))
            handled.add(label)

    for label, found in by_column.items():
        if label in handled:
            continue
        real = _real_column(df, label)
        if "numbers_as_text" in found:
            fixes.append(_fix("to_numeric", label, "All values are numbers stored as text."))
            handled.add(label)
        elif "mixed_types" in found:
            n = _non_numeric_count(df[real])
            share = n / max(int(df[real].notna().sum()), 1)
            if share <= MAX_COERCE_SHARE:
                fixes.append(
                    _fix("to_numeric", label, f"Mostly numbers; {n} non-numeric value(s) would become missing.")
                )
                handled.add(label)
            # otherwise: likely a code/identifier column, leave it as text
        elif "date_like_text" in found:
            fixes.append(_fix("to_datetime", label, "Dates are stored as text; unparseable values would become missing."))
            handled.add(label)

    # Missing values: only for columns with no other fix (a converted column
    # should be re-validated after conversion, not filled blindly).
    for label, found in by_column.items():
        if label in handled or "missing_values" not in found:
            continue
        series = df[_real_column(df, label)]
        kind = column_kind(series)
        pct = 100 * series.isna().mean()
        if pct >= MISSING_DROP_PCT:
            fixes.append(_fix("drop_column", label, f"{pct:.1f}% of values are missing; too sparse to fill reliably."))
        elif kind == "numeric":
            fixes.append(_fix("fill_median", label, f"Fill {pct:.1f}% missing values with the column median."))
        elif kind == "text":
            fixes.append(_fix("fill_unknown", label, f"Fill {pct:.1f}% missing values with the label 'Unknown'."))
        # booleans and datetimes are left alone: no safe default

    if any(i["type"] == "duplicate_rows" for i in issues):
        n = int(df.duplicated().sum())
        fixes.append(_fix("drop_duplicates", None, f"Remove {n} exact duplicate row(s), keeping the first."))

    fixes.sort(key=lambda f: (ACTION_ORDER[f["action"]], f["column"] or ""))
    return fixes


def _proposals(store: DatasetStore, dataset_id: str, df: pd.DataFrame) -> tuple[list[dict] | None, ToolResult | None]:
    """Return (fixes, None) or (None, error result)."""
    if not df.columns.is_unique:
        return None, error_result(
            "Duplicate column names make cleaning ambiguous.",
            hint="Rename the duplicated columns in the source file and reload it.",
        )
    if df.empty:
        return [], None
    validation = validate_data(store, dataset_id)
    if not validation.ok:
        return None, validation
    return _build_fixes(df, validation.data["issues"]), None


# --------------------------------------------------------------------------
# Tool 3: propose_cleaning
# --------------------------------------------------------------------------

def propose_cleaning(store: DatasetStore, dataset_id: str) -> ToolResult:
    """Suggest fixes for the problems validate_data finds. Applies nothing."""
    df, error = fetch(store, dataset_id)
    if error:
        return error
    fixes, error = _proposals(store, dataset_id, df)
    if error:
        return error
    if not fixes:
        return ok_result("No cleaning needed.", fixes=[], truncated=False)
    shown = fixes[:MAX_FIXES]
    return ok_result(
        f"{len(fixes)} cleaning fix(es) proposed; none applied yet.",
        fixes=shown,
        truncated=len(fixes) > MAX_FIXES,
    )


# --------------------------------------------------------------------------
# Tool 4: apply_cleaning
# --------------------------------------------------------------------------

def _apply_one(df: pd.DataFrame, fix: dict) -> tuple[pd.DataFrame, int, str]:
    """Apply one fix to df. Returns (new df, rows affected, change description)."""
    action, label = fix["action"], fix["column"]

    if action == "drop_duplicates":
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        return df, removed, f"Removed {removed} duplicate row(s)."

    real = _real_column(df, label)
    series = df[real]

    if action == "drop_column":
        return df.drop(columns=[real]), 0, f"Dropped column '{label}'."

    if action == "to_numeric":
        text = series.where(series.isna(), series.astype(str).str.strip())
        converted = pd.to_numeric(text, errors="coerce")
        lost = int((converted.isna() & series.notna()).sum())
        df[real] = converted
        return df, int(series.notna().sum()), f"Converted '{label}' to numbers; {lost} value(s) became missing."

    if action == "to_datetime":
        text = series.where(series.isna(), series.astype(str).str.strip())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(text, errors="coerce", format="mixed")
        lost = int((converted.isna() & series.notna()).sum())
        df[real] = converted
        return df, int(series.notna().sum()), f"Converted '{label}' to dates; {lost} value(s) became missing."

    if action == "fill_median":
        n = int(series.isna().sum())
        df[real] = series.fillna(series.median())
        return df, n, f"Filled {n} missing value(s) in '{label}' with the median."

    if action == "fill_unknown":
        n = int(series.isna().sum())
        df[real] = series.fillna("Unknown")
        return df, n, f"Filled {n} missing value(s) in '{label}' with 'Unknown'."

    raise ValueError(f"Unhandled action: {action}")  # programming error, not user input


def apply_cleaning(store: DatasetStore, dataset_id: str, fix_ids: list[str]) -> ToolResult:
    """Apply the chosen fixes to a copy and save it as the dataset's current version."""
    df, error = fetch(store, dataset_id)
    if error:
        return error
    if not fix_ids:
        return error_result("No fix ids given.", hint="Call propose_cleaning and pass the ids to apply.")

    proposals, error = _proposals(store, dataset_id, df)
    if error:
        return error
    by_id = {f["id"]: f for f in proposals}

    unknown = [fid for fid in fix_ids if fid not in by_id]
    if unknown:
        valid = list(by_id)[:MAX_HINT_COLUMNS]
        return error_result(
            f"Unknown fix id(s): {', '.join(unknown)}.",
            hint=f"Valid fix ids: {', '.join(valid) if valid else '(none, the data needs no cleaning)'}",
        )

    chosen = sorted({by_id[fid]["id"] for fid in fix_ids}, key=lambda fid: (ACTION_ORDER[by_id[fid]["action"]], fid))
    working = df.copy()  # the original is never modified
    shape_before = list(df.shape)
    change_log = []
    for fid in chosen:
        fix = by_id[fid]
        working, affected, description = _apply_one(working, fix)
        change_log.append({"fix_id": fid, "rows_affected": affected, "change": description})

    if working.shape[1] == 0:
        return error_result("These fixes would remove every column.", hint="Deselect some of the drop_column fixes.")

    store.update(dataset_id, working)  # original upload is kept by the store
    return ok_result(
        f"Applied {len(chosen)} fix(es): shape {shape_before[0]}x{shape_before[1]} -> "
        f"{working.shape[0]}x{working.shape[1]}. The original upload is kept.",
        dataset_id=dataset_id,
        shape_before=shape_before,
        shape_after=list(working.shape),
        change_log=change_log,
    )
