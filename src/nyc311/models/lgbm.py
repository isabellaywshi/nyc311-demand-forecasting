"""Global direct LightGBM forecaster with quantile intervals (spec section 3)."""

from __future__ import annotations

from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from nyc311.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, build_features
from nyc311.models.base import FORECAST_COLUMNS, forecast_frame

_RANDOM_STATE = 0
_QUANTILES = {"p10": 0.1, "p90": 0.9}


class LGBMForecaster:
    name = "lgbm"

    def __init__(self, params: dict, train_origin_step: int = 1):
        self.params = dict(params)
        self.train_origin_step = max(1, int(train_origin_step))
        self.models_: dict[tuple[int, str], lgb.LGBMRegressor] = {}
        self.train_frame_: pd.DataFrame = pd.DataFrame()
        self._horizons: list[int] = []
        self._categories: dict[str, pd.Index] = {}

    def _base_params(self) -> dict:
        return {**self.params, "verbose": -1, "random_state": _RANDOM_STATE, "n_jobs": -1}

    def _set_categories(self, panel: pd.DataFrame) -> None:
        self._categories = {c: pd.Index(sorted(panel[c].unique())) for c in CATEGORICAL_COLUMNS}

    def _cast(self, feats: pd.DataFrame) -> pd.DataFrame:
        feats = feats.copy()
        for c in CATEGORICAL_COLUMNS:
            feats[c] = pd.Categorical(feats[c], categories=self._categories[c])
        return feats

    def fit(
        self, panel: pd.DataFrame, last_origin: date, horizons: list[int] | None = None
    ) -> None:
        """Fit 3 models per horizon on rows whose labels are observed by last_origin."""
        horizons = list(horizons) if horizons is not None else [1, 2, 3, 4]
        self._set_categories(panel)
        sliced = panel[panel["week_start"] <= last_origin]
        weeks = sorted(sliced["week_start"].unique())
        origins = weeks[::-1][:: self.train_origin_step]  # keep the most recent origin
        feats = build_features(sliced, origins, horizons, with_label=True)
        feats = self._cast(feats)
        self.train_frame_ = feats
        self._horizons = horizons
        self.models_ = {}
        base = self._base_params()
        point = {**base, "objective": "tweedie"}
        quant = {k: v for k, v in base.items() if k != "tweedie_variance_power"}
        for h in horizons:
            fh = feats[feats["horizon"] == h]
            x, y = fh[FEATURE_COLUMNS], fh["label"].astype(float)
            m = lgb.LGBMRegressor(**point)
            m.fit(x, y, categorical_feature=CATEGORICAL_COLUMNS)
            self.models_[(h, "point")] = m
            for key, alpha in _QUANTILES.items():
                q = lgb.LGBMRegressor(**quant, objective="quantile", alpha=alpha)
                q.fit(x, y, categorical_feature=CATEGORICAL_COLUMNS)
                self.models_[(h, key)] = q

    def predict(self, panel: pd.DataFrame, origin: date) -> pd.DataFrame:
        sliced = panel[panel["week_start"] <= origin]
        frames = []
        for h in self._horizons:
            feats = build_features(sliced, [origin], [h], with_label=False)
            if feats.empty:
                continue
            feats = self._cast(feats)
            x = feats[FEATURE_COLUMNS]
            raw = np.column_stack(
                [self.models_[(h, k)].predict(x) for k in ("p10", "point", "p90")]
            )
            vals = np.sort(np.clip(raw * feats["scale"].to_numpy()[:, None], 0, None), axis=1)
            # Tweedie's log link never reaches exactly 0: a series silent for a full
            # year (roll_mean_52 == 0) is forecast as 0 outright.
            vals[feats["roll_mean_52"].to_numpy() == 0] = 0.0
            frames.append(
                forecast_frame(
                    feats["borocd"].to_numpy(),
                    feats["target"].to_numpy(),
                    origin,
                    h,
                    origin + timedelta(weeks=h),
                    self.name,
                    vals[:, 1],
                    vals[:, 0],
                    vals[:, 2],
                )
            )
        if not frames:
            return pd.DataFrame(columns=FORECAST_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def forecast(self, panel: pd.DataFrame, origin: date, horizons: list[int]) -> pd.DataFrame:
        self.fit(panel, origin, horizons)
        return self.predict(panel, origin)
