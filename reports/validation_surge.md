### validation - surge

Filters: all targets, regular districts, horizons 1-4 pooled; lgbm alert = yhat >= 1.25 x norm and norm >= 5; rule_baseline alert = same rule on recent_average

| model | precision | recall | f1 | n_actual | n_alerts |
|---|---|---|---|---|---|
| lgbm | 0.6793 | 0.6786 | 0.6789 | 6254 | 6248 |
| rule_baseline | 0.6159 | 0.6212 | 0.6185 | 6254 | 6308 |
