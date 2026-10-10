"""The eval questions. Expected numbers are computed from the data with pandas at run
time, so they are always right for the file being tested. Written for the Titanic CSV."""

from eval.harness import Case

GA = ("group_aggregate",)


def _rate(df, mask=None):
    sub = df if mask is None else df[mask]
    return float(sub["Survived"].mean())


CASES = [
    Case("rows", "How many passengers are in the dataset?", lambda df: [len(df)]),
    Case("overall_rate", "What is the overall survival rate?",
         lambda df: [_rate(df)], ("describe_stats", "group_aggregate")),
    Case("rate_by_sex", "What was the survival rate for men and for women?",
         lambda df: [_rate(df, df.Sex == "male"), _rate(df, df.Sex == "female")], GA),
    Case("rate_by_class", "What was the survival rate in each passenger class?",
         lambda df: [_rate(df, df.Pclass == c) for c in (1, 2, 3)], GA),
    Case("women_first", "What was the survival rate for women in first class, and how many were there?",
         lambda df: [_rate(df, (df.Sex == "female") & (df.Pclass == 1)),
                     int(((df.Sex == "female") & (df.Pclass == 1)).sum())], GA),
    Case("men_third", "How many men travelled in third class and what share survived?",
         lambda df: [int(((df.Sex == "male") & (df.Pclass == 3)).sum()),
                     _rate(df, (df.Sex == "male") & (df.Pclass == 3))], GA),
    Case("fare_by_class", "What was the average fare in each class?",
         lambda df: [float(df[df.Pclass == c].Fare.mean()) for c in (1, 2, 3)], GA),
    Case("mean_age", "What is the average age of passengers?",
         lambda df: [float(df.Age.mean())], ("describe_stats", "group_aggregate")),
    Case("median_fare", "What is the median fare?",
         lambda df: [float(df.Fare.median())], ("describe_stats",)),
    Case("max_fare", "What was the highest fare paid?",
         lambda df: [float(df.Fare.max())], ("describe_stats", "group_aggregate")),
    Case("fare_outliers", "How many fare outliers are there?",
         lambda df: [int(((df.Fare < df.Fare.quantile(.25) - 1.5 * (df.Fare.quantile(.75) - df.Fare.quantile(.25)))
                          | (df.Fare > df.Fare.quantile(.75) + 1.5 * (df.Fare.quantile(.75) - df.Fare.quantile(.25)))).sum())],
         ("detect_outliers",)),
    Case("corr_class_fare", "How strongly are passenger class and fare correlated?",
         lambda df: [float(df.Pclass.corr(df.Fare))], ("correlation_analysis",)),
    Case("rate_by_port", "What was the survival rate by port of embarkation?",
         lambda df: [_rate(df, df.Embarked == p) for p in ("S", "C", "Q")], GA),
    Case("port_most", "How many passengers boarded at the busiest port?",
         lambda df: [int(df.Embarked.value_counts().iloc[0])], GA),
    Case("age_survivors", "What was the average age of survivors compared with those who died?",
         lambda df: [float(df[df.Survived == 1].Age.mean()), float(df[df.Survived == 0].Age.mean())], GA),
    Case("children", "What was the survival rate of children under 10?",
         lambda df: [_rate(df, df.Age < 10)], GA),
    Case("chart_rate_class", "Show the survival rate by passenger class as a chart.",
         any_tool=("create_chart",), needs_chart=True),
    Case("refuse_mars", "What is the average ticket price on Mars?", refusal=True),
    Case("refuse_salary", "What is the average salary of the passengers?", refusal=True),
]
