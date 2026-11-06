"""Forecasting models. Every model maps day-level features to 24 normalised
hourly values; ``fit`` receives training days, ``predict`` returns (n, 24)."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from features import to_long

warnings.filterwarnings("ignore")


class SeasonalNaive:
    """Same hour one week earlier (D-7), rescaled: the normalised lag itself."""
    name = "SN7"
    needs_fit = False

    def fit(self, X, Y):
        return self

    def predict(self, X):
        return X[[f"l7_h{h}" for h in range(24)]].values


class SeasonalNaiveDayType:
    """Most recent available day of the same type: D-2 for Wed-Fri (when D-2 is a
    regular working day), D-7 otherwise."""
    name = "SN-DT"
    needs_fit = False

    def fit(self, X, Y):
        return self

    def predict(self, X):
        l2 = X[[f"l2_h{h}" for h in range(24)]].values
        l7 = X[[f"l7_h{h}" for h in range(24)]].values
        use2 = X["day_type"].isin([2, 3, 4]).values & (X["off_l2"].values == 0) & (X["holiday"].values == 0)
        return np.where(use2[:, None], l2, l7)


def _linear_cols(X):
    return [c for c in X.columns if c != "day_type"]


class ARX:
    """Multi-output ridge regression = 24 hour-specific linear ARX models
    (in the spirit of LEAR for electricity prices)."""
    name = "ARX"
    needs_fit = True

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Y):
        self.cols = _linear_cols(X)
        Xv = X[self.cols].fillna(0).values
        self.sc = StandardScaler().fit(Xv)
        self.m = Ridge(alpha=self.alpha).fit(self.sc.transform(Xv), Y.values)
        return self

    def predict(self, X):
        return self.m.predict(self.sc.transform(X[self.cols].fillna(0).values))


class MLP:
    """Feed-forward network with 24 outputs; average of ``n_seeds`` networks."""
    name = "MLP"
    needs_fit = True

    def __init__(self, hidden=(128, 64), alpha=1e-3, n_seeds=3, max_iter=400):
        self.hidden, self.alpha, self.n_seeds, self.max_iter = hidden, alpha, n_seeds, max_iter

    def fit(self, X, Y):
        self.cols = _linear_cols(X)
        Xv = X[self.cols].fillna(0).values
        self.sc = StandardScaler().fit(Xv)
        self.mu, self.sd = Y.values.mean(0), Y.values.std(0) + 1e-9
        Yt = (Y.values - self.mu) / self.sd
        self.ms = [MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha, max_iter=self.max_iter,
                                early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                learning_rate_init=1e-3, random_state=k).fit(self.sc.transform(Xv), Yt)
                   for k in range(self.n_seeds)]
        return self

    def predict(self, X):
        Z = self.sc.transform(X[self.cols].fillna(0).values)
        return np.mean([m.predict(Z) for m in self.ms], axis=0) * self.sd + self.mu


class GBM:
    """Global LightGBM model over (day, hour) pairs; hour is a feature."""
    name = "GBM"
    needs_fit = True

    def __init__(self, objective="regression", alpha=None, n_estimators=600, learning_rate=0.03,
                 num_leaves=31, min_child_samples=40, subsample=0.8, colsample=0.8):
        self.params = dict(objective=objective, n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples, subsample=subsample,
                           subsample_freq=1, colsample_bytree=colsample, verbose=-1, n_jobs=1, random_state=0)
        if alpha is not None:
            self.params["alpha"] = alpha

    def fit(self, X, Y):
        L, y = to_long(X, Y)
        ok = y.notna()
        self.m = lgb.LGBMRegressor(**self.params).fit(L[ok], y[ok], categorical_feature=["day_type", "hour"])
        return self

    def predict(self, X):
        L, _ = to_long(X, None)
        p = pd.Series(self.m.predict(L), index=L.index)
        return p.unstack("hour").reindex(X.index).values


class Ensemble:
    name = "ENS"
    needs_fit = True

    def __init__(self, members):
        self.members = members

    def fit(self, X, Y):
        for m in self.members:
            m.fit(X, Y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)


class Masked:
    """Ablation wrapper: overwrite selected input columns with a constant so the
    wrapped model receives no information from them."""

    def __init__(self, model, prefixes, value=0.0):
        self.model, self.prefixes, self.value = model, tuple(prefixes), value
        self.name = model.name + "-masked"

    def _mask(self, X):
        X = X.copy()
        cols = [c for c in X.columns if c.startswith(self.prefixes)]
        X[cols] = self.value
        if "day_type" in X and any(p in ("holiday", "carnival", "dt") for p in self.prefixes):
            X["day_type"] = X.index.dayofweek  # treat holidays as ordinary days
            for k in range(8):
                X[f"dt{k}"] = (X["day_type"] == k).astype(int)
        return X

    def fit(self, X, Y):
        self.model.fit(self._mask(X), Y)
        return self

    def predict(self, X):
        return self.model.predict(self._mask(X))


class Hybrid:
    """Linear ARX backbone + LightGBM on its (in-sample) residuals: the trees only
    need to learn non-linear corrections (e.g. holiday x hour interactions)."""
    name = "HYB"
    needs_fit = True

    def __init__(self, alpha=10.0, **gbm_kw):
        self.arx = ARX(alpha)
        self.gbm = GBM(**gbm_kw)

    def fit(self, X, Y):
        self.arx.fit(X, Y)
        R = Y - self.arx.predict(X)
        self.gbm.fit(X, pd.DataFrame(R, index=Y.index, columns=Y.columns))
        return self

    def predict(self, X):
        return self.arx.predict(X) + self.gbm.predict(X)
