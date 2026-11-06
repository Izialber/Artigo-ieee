"""Probabilistic day-ahead forecasts built on top of point forecasts.

* EQR  - rolling empirical quantiles of relative errors, per hour of day
         (split-conformal style, using only outcomes known at issuance time).
* ACI  - adaptive conformal inference (Gibbs & Candes, 2021) applied to EQR
         intervals, with delayed feedback (outcomes of D-2 are the most recent
         ones available when forecasting D).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wide(df, col):
    return df.pivot(index="date", columns="hour", values=col).sort_index()


def eqr_quantiles(df: pd.DataFrame, taus, window=91) -> dict:
    """df: one subsystem with columns date, hour, y, pred. Returns {tau: wide frame}."""
    Y, P = _wide(df, "y"), _wide(df, "pred")
    E = Y / P - 1.0
    out = {t: pd.DataFrame(np.nan, index=P.index, columns=P.columns) for t in taus}
    Ev = E.values
    for i in range(len(P)):
        j1 = i - 1  # exclusive end: rows < i-1 (i.e., up to D-2)
        j0 = j1 - window
        if j0 < 0:
            continue
        hist = Ev[j0:j1]
        for t in taus:
            out[t].iloc[i] = P.values[i] * (1 + np.nanquantile(hist, t, axis=0))
    return out


def aci_interval(df: pd.DataFrame, alpha=0.1, window=91, gamma=0.01):
    Y, P = _wide(df, "y"), _wide(df, "pred")
    E = (Y / P - 1.0).values
    n = len(P)
    lo = np.full(P.shape, np.nan)
    hi = np.full(P.shape, np.nan)
    a_t = alpha
    miss_hist = {}
    for i in range(n):
        # update with the outcome of D-2 (if an interval was issued for it)
        k = i - 2
        if k in miss_hist:
            a_t = a_t + gamma * (alpha - miss_hist[k])
        j1, j0 = i - 1, i - 1 - window
        if j0 < 0:
            continue
        hist = E[j0:j1]
        a_eff = float(np.clip(a_t, 1e-3, 0.999))
        ql = np.nanquantile(hist, a_eff / 2, axis=0)
        qh = np.nanquantile(hist, 1 - a_eff / 2, axis=0)
        lo[i] = P.values[i] * (1 + ql)
        hi[i] = P.values[i] * (1 + qh)
        y = Y.values[i]
        miss_hist[i] = float(np.mean((y < lo[i]) | (y > hi[i])))
    idx = P.index
    return pd.DataFrame(lo, index=idx, columns=P.columns), pd.DataFrame(hi, index=idx, columns=P.columns)
