### test - surge

Filters: all targets, regular districts, horizons 1-4 pooled; lgbm alert = yhat >= 1.25 x norm and norm >= 5; rule_baseline alert = same rule on recent_average

| model | precision | recall | f1 | n_actual | n_alerts |
|---|---|---|---|---|---|
| lgbm | 0.6654 | 0.6697 | 0.6675 | 6076 | 6115 |
| rule_baseline | 0.6038 | 0.6392 | 0.6210 | 6076 | 6433 |
