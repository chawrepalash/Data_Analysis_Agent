# Eval results

Model: `gemini-3.5-flash-lite`  |  Data: `titanic.csv`  |  **19/19 correct (100%)**
Average 9.9 s and 1.11 tool calls per question.

| # | Question | Result | Tools | Notes |
|---|----------|--------|-------|-------|
| 1 | How many passengers are in the dataset? | pass | group_aggregate |  |
| 2 | What is the overall survival rate? | pass | group_aggregate |  |
| 3 | What was the survival rate for men and for women? | pass | group_aggregate |  |
| 4 | What was the survival rate in each passenger class? | pass | group_aggregate |  |
| 5 | What was the survival rate for women in first class, and how many were there? | pass | group_aggregate |  |
| 6 | How many men travelled in third class and what share survived? | pass | group_aggregate |  |
| 7 | What was the average fare in each class? | pass | group_aggregate |  |
| 8 | What is the average age of passengers? | pass | group_aggregate |  |
| 9 | What is the median fare? | pass | describe_stats |  |
| 10 | What was the highest fare paid? | pass | describe_stats |  |
| 11 | How many fare outliers are there? | pass | detect_outliers |  |
| 12 | How strongly are passenger class and fare correlated? | pass | correlation_analysis, correlation_analysis |  |
| 13 | What was the survival rate by port of embarkation? | pass | group_aggregate |  |
| 14 | How many passengers boarded at the busiest port? | pass | group_aggregate |  |
| 15 | What was the average age of survivors compared with those who died? | pass | group_aggregate |  |
| 16 | What was the survival rate of children under 10? | pass | group_aggregate |  |
| 17 | Show the survival rate by passenger class as a chart. | pass | create_chart, group_aggregate |  |
| 18 | What is the average ticket price on Mars? | pass | group_agency_or_something_invalid |  |
| 19 | What is the average salary of the passengers? | pass | describe_stats |  |
