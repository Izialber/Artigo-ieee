"""Day-ahead feature construction with an operationally realistic information set.

Forecasts for every hour of target day D are issued on D-1 at 12:00 local
time. At issuance the most recent available observation is the hour starting
at D-1 10:00 (``LAST_HOUR``); hence horizons range from 14 to 37 hours.

All load-valued inputs and the target are divided by a scale ``s_D``: the mean
load over the seven most recent complete days (D-8 ... D-2). This makes the
learning problem approximately stationary in level, which matters for tree
ensembles that cannot extrapolate beyond the training range.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from dateutil.easter import easter

LAST_HOUR = 10  # last observed hour on D-1


# --------------------------------------------------------------------- calendar
def br_calendar(years) -> pd.DataFrame:
    """National holidays and special days of Brazil relevant to electricity demand."""
    rows = []
    for y in years:
        e = easter(y)
        fixed = {
            date(y, 1, 1): "new_year", date(y, 4, 21): "tiradentes", date(y, 5, 1): "labour",
            date(y, 9, 7): "independence", date(y, 10, 12): "aparecida", date(y, 11, 2): "souls",
            date(y, 11, 15): "republic", date(y, 12, 25): "christmas",
        }
        if y >= 2024:
            fixed[date(y, 11, 20)] = "black_consciousness"  # national since Law 14.759/2023
        for d, n in fixed.items():
            rows.append((d, n, "holiday"))
        rows += [
            (e - timedelta(days=48), "carnival_mon", "carnival"),
            (e - timedelta(days=47), "carnival_tue", "carnival"),
            (e - timedelta(days=46), "ash_wed", "ash_wed"),
            (e - timedelta(days=2), "good_friday", "holiday"),
            (e + timedelta(days=60), "corpus_christi", "holiday"),
            (date(y, 12, 24), "christmas_eve", "eve"),
            (date(y, 12, 31), "new_year_eve", "eve"),
        ]
    cal = pd.DataFrame(rows, columns=["date", "name", "kind"])
    cal["date"] = pd.to_datetime(cal["date"])
    return cal.drop_duplicates("date").set_index("date").sort_index()


def day_calendar(days: pd.DatetimeIndex) -> pd.DataFrame:
    cal = br_calendar(range(days.min().year - 1, days.max().year + 2))
    kind = cal["kind"].reindex(days)
    f = pd.DataFrame(index=days)
    f["dow"] = days.dayofweek
    f["holiday"] = (kind == "holiday").astype(int)
    f["carnival"] = (kind == "carnival").astype(int)
    f["ash_wed"] = (kind == "ash_wed").astype(int)
    f["eve"] = (kind == "eve").astype(int)
    off = (f["holiday"] + f["carnival"]).astype(bool)
    # bridge day: working day sandwiched between a holiday and the weekend
    prev_off = off.shift(1, fill_value=False) | (days - pd.Timedelta(days=1)).isin(cal.index[cal.kind == "holiday"])
    next_off = off.shift(-1, fill_value=False)
    f["bridge"] = (((f.dow == 0) & next_off) | ((f.dow == 4) & prev_off)).astype(int) * (~off).astype(int)
    # year-end holiday season (26 Dec - 6 Jan, excluding holidays themselves)
    md = days.month * 100 + days.day
    f["yearend"] = (((md >= 1226) | (md <= 106)) & ~off & (f.eve == 0)).astype(int)
    f["day_type"] = np.where(off, 7, f["dow"])  # 0..6 weekdays, 7 = holiday/carnival
    doy = days.dayofyear.values
    for k in (1, 2):
        f[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        f[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    f["off"] = off.astype(int)
    return f


# ------------------------------------------------------------------- features
def build_day_matrix(y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build day-level features X (one row per target day), targets Y (24 cols,
    normalised) and scale s. ``y`` is an hourly series with complete days."""
    M = y.values.reshape(-1, 24)
    days = pd.DatetimeIndex(y.index[::24])
    L = pd.DataFrame(M, index=days)
    daily = L.mean(axis=1)
    s = daily.shift(2).rolling(7).mean()  # mean of D-8..D-2
    cal = day_calendar(days)
    X = {}
    for lag in (2, 7, 14):
        sh = L.shift(lag).div(s, axis=0)
        for h in range(24):
            X[f"l{lag}_h{h}"] = sh[h]
    d1 = L.shift(1).div(s, axis=0)
    for h in range(LAST_HOUR + 1):
        X[f"l1_h{h}"] = d1[h]
    X["d1_morning"] = d1.loc[:, :LAST_HOUR].mean(axis=1)
    X["d2_morning"] = L.shift(2).loc[:, :LAST_HOUR].mean(axis=1) / s
    X["d8_morning"] = L.shift(8).loc[:, :LAST_HOUR].mean(axis=1) / s
    X["trend7"] = daily.shift(2).rolling(7).mean() / daily.shift(9).rolling(7).mean()
    X = pd.DataFrame(X, index=days)
    # calendar of target day and of the lagged days (lags on holidays are atypical)
    for c in ["holiday", "carnival", "ash_wed", "eve", "bridge", "yearend", "sin1", "cos1", "sin2", "cos2"]:
        X[c] = cal[c]
    for lag in (1, 2, 7):
        X[f"off_l{lag}"] = cal["off"].shift(lag)
    X["day_type"] = cal["day_type"]
    for k in range(8):
        X[f"dt{k}"] = (cal["day_type"] == k).astype(int)
    Y = L.div(s, axis=0)
    return X, Y, s


DAY_FEATS_LINEAR = None  # determined at runtime (all numeric except day_type)


def to_long(X: pd.DataFrame, Y: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Series | None]:
    """Convert day-level matrices to an hour-level frame for global tree models."""
    parts = []
    common = ["d1_morning", "d2_morning", "d8_morning", "trend7", "holiday", "carnival", "ash_wed", "eve",
              "bridge", "yearend", "sin1", "cos1", "sin2", "cos2", "off_l1", "off_l2", "off_l7", "day_type"]
    for h in range(24):
        f = X[common].copy()
        f["hour"] = h
        f["l2"] = X[f"l2_h{h}"]
        f["l7"] = X[f"l7_h{h}"]
        f["l14"] = X[f"l14_h{h}"]
        f["l1_last"] = X[f"l1_h{LAST_HOUR}"]
        f["l1_same"] = X[f"l1_h{h}"] if h <= LAST_HOUR else np.nan
        f["l2_last"] = X[f"l2_h{LAST_HOUR}"]
        f["date"] = X.index
        parts.append(f)
    long = pd.concat(parts).set_index(["date", "hour"], drop=False).sort_index()
    long = long.drop(columns=["date"])
    yl = None
    if Y is not None:
        yl = Y.stack().rename("y")
        yl.index.names = ["date", "hour"]
        yl = yl.reindex(long.index)
    return long, yl
