import inspect
from src.tools.stats import describe_stats, correlation_analysis, detect_outliers
from src.tools.query import group_aggregate
from src.tools.charts import create_chart

for fn in (describe_stats, correlation_analysis, detect_outliers, group_aggregate, create_chart):
    print(fn.__name__, inspect.signature(fn))
    print((fn.__doc__ or "").strip())
    print()
