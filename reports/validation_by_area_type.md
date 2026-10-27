### validation - by_area_type

Filters: target = total, regular vs special areas, horizons 1-4 pooled

| area_type | model | wape | mase | coverage | width |
|---|---|---|---|---|---|
| regular | seasonal_naive | 0.1548 | 1.0000 | nan | nan |
| regular | recent_average | 0.1350 | 0.8722 | nan | nan |
| regular | auto_ets | 0.1333 | 0.8614 | 0.8752 | 559.2823 |
| regular | mstl | 0.1361 | 0.8793 | 0.7966 | 412.5577 |
| regular | lgbm | 0.1223 | 0.7902 | 0.7875 | 378.8221 |
| special | seasonal_naive | 0.3848 | 1.0000 | nan | nan |
| special | recent_average | 0.3010 | 0.7820 | nan | nan |
| special | auto_ets | 0.2965 | 0.7704 | 0.8109 | 23.3486 |
| special | mstl | 0.2841 | 0.7383 | 0.7420 | 18.1489 |
| special | lgbm | 0.2691 | 0.6993 | 0.8061 | 23.5893 |
