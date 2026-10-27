### test - by_area_type

Filters: target = total, regular vs special areas, horizons 1-4 pooled

| area_type | model | wape | mase | coverage | width |
|---|---|---|---|---|---|
| regular | seasonal_naive | 0.1889 | 1.0000 | nan | nan |
| regular | recent_average | 0.1325 | 0.7013 | nan | nan |
| regular | auto_ets | 0.1415 | 0.7488 | 0.8418 | 601.9543 |
| regular | mstl | 0.1282 | 0.6786 | 0.7835 | 465.3751 |
| regular | lgbm | 0.1177 | 0.6231 | 0.8086 | 418.6240 |
| special | seasonal_naive | 0.3840 | 1.0000 | nan | nan |
| special | recent_average | 0.2848 | 0.7416 | nan | nan |
| special | auto_ets | 0.2824 | 0.7355 | 0.8212 | 22.6116 |
| special | mstl | 0.2669 | 0.6951 | 0.7378 | 17.9141 |
| special | lgbm | 0.2663 | 0.6934 | 0.7569 | 20.2270 |
