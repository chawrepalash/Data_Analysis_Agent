"""Reading uploaded files and holding dataframes in memory.

The LLM never sees a dataframe. Only a dataset id travels through the agent
state, and tools look the dataframe up in a DatasetStore.
"""

import io
import os
import uuid
from dataclasses import dataclass

import pandas as pd

import config


class DataLoadError(Exception):
    """The file could not be turned into a usable dataframe."""


class DatasetNotFound(KeyError):
    """No dataset is stored under this id."""


def _check_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise DataLoadError(f"Unsupported file type '{ext or filename}'. Use one of: {allowed}.")
    return ext


def _check_size(size_bytes: int) -> None:
    if size_bytes == 0:
        raise DataLoadError("The file is empty.")
    if size_bytes > config.MAX_FILE_BYTES:
        mb = size_bytes / (1024 * 1024)
        raise DataLoadError(
            f"The file is {mb:.1f} MB, over the {config.MAX_FILE_MB} MB limit."
        )


def _read_csv(data: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in config.CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError:
            raise DataLoadError("The file has no columns to read.")
        except pd.errors.ParserError as exc:
            raise DataLoadError(f"The CSV could not be parsed: {exc}")
    raise DataLoadError(f"Could not decode the file text: {last_error}")


def _read_excel(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")
    except Exception as exc:  # openpyxl raises several unrelated types
        raise DataLoadError(f"The Excel file could not be read: {exc}")


def load_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded bytes (for example from Streamlit) into a dataframe."""
    ext = _check_extension(filename)
    _check_size(len(data))
    df = _read_csv(data) if ext == ".csv" else _read_excel(data)

    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] == 0:
        raise DataLoadError("The file has no columns.")
    if df.shape[0] == 0:
        raise DataLoadError("The file has a header but no data rows.")
    return df


def load_file(path: str) -> pd.DataFrame:
    """Read a file from disk. Checks size before reading it into memory."""
    _check_extension(path)
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise DataLoadError(f"Cannot open the file: {exc}")
    _check_size(size)
    with open(path, "rb") as handle:
        return load_bytes(handle.read(), os.path.basename(path))


@dataclass
class DatasetMeta:
    dataset_id: str
    name: str


class DatasetStore:
    """In-memory dataframes keyed by dataset id.

    Each dataset keeps its original upload untouched plus a current version
    that cleaning can replace. Reset brings the original back.
    """

    def __init__(self) -> None:
        self._original: dict[str, pd.DataFrame] = {}
        self._current: dict[str, pd.DataFrame] = {}
        self._meta: dict[str, DatasetMeta] = {}

    def add(self, df: pd.DataFrame, name: str = "dataset") -> str:
        dataset_id = uuid.uuid4().hex[:8]
        self._original[dataset_id] = df.copy()
        self._current[dataset_id] = df.copy()
        self._meta[dataset_id] = DatasetMeta(dataset_id, name)
        return dataset_id

    def _require(self, dataset_id: str) -> None:
        if dataset_id not in self._current:
            raise DatasetNotFound(f"No dataset with id '{dataset_id}'.")

    def get(self, dataset_id: str) -> pd.DataFrame:
        """Current version. Tools must treat it as read-only."""
        self._require(dataset_id)
        return self._current[dataset_id]

    def original(self, dataset_id: str) -> pd.DataFrame:
        self._require(dataset_id)
        return self._original[dataset_id]

    def update(self, dataset_id: str, df: pd.DataFrame) -> None:
        """Replace the current version (used by cleaning). Original is kept."""
        self._require(dataset_id)
        self._current[dataset_id] = df

    def reset(self, dataset_id: str) -> None:
        self._require(dataset_id)
        self._current[dataset_id] = self._original[dataset_id].copy()

    def meta(self, dataset_id: str) -> DatasetMeta:
        self._require(dataset_id)
        return self._meta[dataset_id]
