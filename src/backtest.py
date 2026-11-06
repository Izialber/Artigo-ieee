"""Rolling-origin backtesting engine."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from data import ROOT, SUBSYSTEMS, get_global_load, get_load
from features import build_day_matrix

PRED_DIR = os.path.join(ROOT, "results", "preds")
FREQ = {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS", "Y": "YS"}
LOAD_PREFIXES = ("l1_", "l2_", "l7_", "l14_", "d1_morning", "d2_morning", "d8_morning")


def _refit_dates(start, end, freq):
    d = pd.date_range(start, end, freq=FREQ[freq])
    if len(d) == 0 or d[0] != pd.Timestamp(start):
        d = d.insert(0, pd.Timestamp(start))
    return d


def backtest_series(y: pd.Series, make_model, start, end, window_years=3, freq="W", normalise=True):
    X, Y, s = build_day_matrix(y)
    if not normalise:  # ablation: work in MW
        cols = [c for c in X.columns if c.startswith(LOAD_PREFIXES)]
        X = X.copy()
        X[cols] = X[cols].mul(s, axis=0)
        Y = Y.mul(s, axis=0)
    valid = X[["l14_h0", "trend7"]].notna().all(axis=1) & Y.notna().all(axis=1)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    rd = _refit_dates(start, end, freq)
    for i, t0 in enumerate(rd):
        t1 = rd[i + 1] if i + 1 < len(rd) else end + pd.Timedelta(days=1)
        last_train = t0 - pd.Timedelta(days=2)  # D-2 is the latest complete day at issuance for D=t0
        first_train = X.index.min() if window_years is None else t0 - pd.DateOffset(years=window_years)
        tr = valid & (X.index >= first_train) & (X.index <= last_train)
        te = (X.index >= t0) & (X.index < t1) & (X.index <= end)
        if te.sum() == 0:
            continue
        m = make_model()
        m.fit(X[tr], Y[tr])
        P = m.predict(X[te])
        scale = np.ones(te.sum()) if not normalise else s[te].values
        P = P * scale[:, None]
        out.append(pd.DataFrame(P, index=X.index[te]))
    P = pd.concat(out)
    P.columns = range(24)
    pred = P.stack().rename("pred")
    true = pd.Series(y.values, index=pd.MultiIndex.from_arrays([y.index.normalize(), y.index.hour])).rename("y")
    pred.index.names = ["date", "hour"]
    true.index.names = ["date", "hour"]
    return pd.concat([true.reindex(pred.index), pred], axis=1)


def run(tag, make_model, start, end, window_years=3, freq="W", normalise=True, subsystems=SUBSYSTEMS,
        overwrite=False, verbose=True, dataset="curva"):
    os.makedirs(PRED_DIR, exist_ok=True)
    f = os.path.join(PRED_DIR, f"{tag}.parquet")
    if os.path.exists(f) and not overwrite:
        return pd.read_parquet(f)
    load = get_load() if dataset == "curva" else get_global_load()
    res = []
    t = time.time()
    for ss in subsystems:
        r = backtest_series(load[ss], make_model, start, end, window_years, freq, normalise)
        r["subsystem"] = ss
        res.append(r.reset_index())
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(f)
    if verbose:
        e = df.assign(ape=(df.pred - df.y).abs() / df.y)
        print(f"{tag:45s} {time.time()-t:7.1f}s  MAPE% " +
              " ".join(f"{k}={v*100:.2f}" for k, v in e.groupby("subsystem").ape.mean().items()), flush=True)
    return df
