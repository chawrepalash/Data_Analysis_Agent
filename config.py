"""Central settings. Change values here, not inside other modules."""

MAX_FILE_MB = 50
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# .xls is skipped on purpose: it needs an extra library (xlrd) for little gain.
SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

# Tried in order when reading CSV text.
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")

# Model name lives here so changing it is a one-line edit.
# Check Google's Gemini API docs for the current Flash model name before use.
GEMINI_MODEL = "gemini-2.5-flash"
