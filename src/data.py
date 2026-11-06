"""Download, clean and cache the ONS hourly load data (curva de carga horaria).

Source: ONS open-data portal (S3 bucket `ons-aws-prod-opendata`, dataset
`curva-carga-ho`). Values are hourly average load in MWh/h (= average MW) per
subsystem of the Brazilian National Interconnected System (SIN).
"""
from __future__ import annotations

import glob
import os
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
URL = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho/CURVA_CARGA_{y}.parquet"
SUBSYSTEMS = ["N", "NE", "S", "SE"]
FIRST_YEAR, LAST_YEAR = 2012, 2026


def download(years=range(FIRST_YEAR, LAST_YEAR + 1)) -> None:
    os.makedirs(RAW, exist_ok=True)
    for y in years:
        dst = os.path.join(RAW, f"CURVA_CARGA_{y}.parquet")
        if not os.path.exists(dst):
            urllib.request.urlretrieve(URL.format(y=y), dst)


def load_raw() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW, "CURVA_CARGA_*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["v"] = pd.to_numeric(d["val_cargaenergiahomwmed"], errors="coerce")
    p = d.pivot_table(index="din_instante", columns="id_subsistema", values="v", aggfunc="mean")
    p = p.reindex(pd.date_range(p.index.min().normalize(), p.index.max(), freq="h"))
    p.index.name = "time"
    return p[SUBSYSTEMS]


def clean(p: pd.DataFrame, thr: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace missing values, non-positive values and isolated spikes/dips.

    A value is flagged when it deviates more than ``thr`` (relative) from the
    centred 5-hour rolling median, or is non-positive. Flagged and missing
    hours are filled with the same hour one week earlier, rescaled to the local
    level, falling back to linear interpolation. Returns the cleaned frame and
    a boolean mask of modified entries.
    """
    p = p.copy()
    mask = pd.DataFrame(False, index=p.index, columns=p.columns)
    for c in p.columns:
        s = p[c].where(p[c] > 0)
        med = s.rolling(5, center=True, min_periods=3).median()
        bad = s.isna() | ((s - med).abs() / med > thr)
        mask[c] = bad
        s = s.mask(bad)
        # whole missing days: use previous week's profile scaled to neighbourhood
        wk = s.shift(168)
        lvl = s.rolling(24 * 3, center=True, min_periods=12).mean() / wk.rolling(24 * 3, center=True, min_periods=12).mean()
        fill = wk * lvl
        long_gap = s.isna() & s.isna().rolling(6, center=True, min_periods=1).sum().ge(6)
        s = s.where(~long_gap, fill)
        s = s.interpolate(limit_direction="both")
        p[c] = s
    return p, mask


def get_load(refresh: bool = False) -> pd.DataFrame:
    os.makedirs(PROC, exist_ok=True)
    f = os.path.join(PROC, "load_clean.parquet")
    if os.path.exists(f) and not refresh:
        return pd.read_parquet(f)
    download()
    raw = load_raw()
    cl, mask = clean(raw)
    # keep only complete days
    last_full = cl.index[cl.index.hour == 23].max()
    cl = cl.loc[:last_full]
    cl.to_parquet(f)
    mask.loc[:last_full].to_parquet(os.path.join(PROC, "clean_mask.parquet"))
    return cl


def get_global_load() -> pd.DataFrame:
    """ONS verified global load (incl. estimated behind-the-meter MMGD), hourly,
    2020-2026, from the ONS load API (see ons_api.py), cleaned like the main series."""
    f = os.path.join(PROC, "global_clean.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    g = pd.read_parquet(os.path.join(PROC, "ons_verified_val_cargaglobal.parquet"))
    g = g.loc["2020-01-01":, SUBSYSTEMS]
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
    cl, _ = clean(g)
    last_full = cl.index[cl.index.hour == 23].max()
    cl = cl.loc[:last_full]
    cl.to_parquet(f)
    return cl


if __name__ == "__main__":
    df = get_load(refresh=True)
    m = pd.read_parquet(os.path.join(PROC, "clean_mask.parquet"))
    print(df.shape, df.index.min(), df.index.max())
    print("modified entries per subsystem:\n", m.sum())
    assert not df.isna().any().any()
