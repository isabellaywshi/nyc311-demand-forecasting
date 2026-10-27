# Test-period summary

Headline slice: target = total, regular districts (59), horizons 1-4 pooled (headline).

| model | WAPE | MASE | coverage | width |
|---|---|---|---|---|
| seasonal_naive | 0.1889 | 1.0000 | nan | nan |
| recent_average | 0.1325 | 0.7013 | nan | nan |
| auto_ets | 0.1415 | 0.7488 | 0.8418 | 601.95 |
| mstl | 0.1282 | 0.6786 | 0.7835 | 465.38 |
| lgbm | 0.1177 | 0.6231 | 0.8086 | 418.62 |

## Success criteria (spec section 1)

- PASS: LightGBM WAPE 0.1177 < seasonal_naive 0.1889
- PASS: LightGBM MASE 0.6231 < seasonal_naive 1.0000
- PASS: LightGBM WAPE 0.1177 < recent_average 0.1325
- PASS: LightGBM MASE 0.6231 < recent_average 0.7013
- PASS: P10-P90 coverage 80.9% within 70-90%

Overall: ALL PASS
