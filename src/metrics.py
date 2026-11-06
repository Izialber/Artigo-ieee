"""Accuracy metrics and statistical tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def point_metrics(df: pd.DataFrame) -> pd.Series:
    e = df.pred - df.y
    return pd.Series({
        "MAPE": (e.abs() / df.y).mean() * 100,
        "RMSE": np.sqrt((e ** 2).mean()),
        "MAE": e.abs().mean(),
        "Bias%": (e / df.y).mean() * 100,
    })


def daily_loss(df: pd.DataFrame, kind="ape") -> pd.Series:
    """Daily mean loss over the 24 hours (multivariate DM, Ziel & Weron 2018)."""
    if kind == "ape":
        l = (df.pred - df.y).abs() / df.y
    else:
        l = (df.pred - df.y) ** 2
    return l.groupby(df["date"]).mean()


def dm_test(l1: pd.Series, l2: pd.Series, h: int = 1):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample
    correction. H1: model 1 is more accurate than model 2 (one-sided).
    Returns (statistic, p-value)."""
    d = (l1 - l2).dropna().values
    n = len(d)
    dbar = d.mean()
    # autocovariances up to h-1 (h=1 for day-ahead daily losses, plus robustness lag)
    def acov(k):
        return np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n
    var = acov(0) + 2 * sum(acov(k) for k in range(1, h))
    dm = dbar / np.sqrt(var / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * corr
    p = stats.t.cdf(stat, df=n - 1)  # small p => model 1 better
    return stat, p


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def pinball(y, q, tau):
    u = y - q
    return np.maximum(tau * u, (tau - 1) * u)


def interval_scores(y, lo, hi, alpha):
    cover = ((y >= lo) & (y <= hi)).mean()
    width = (hi - lo)
    wink = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return cover, width, wink
