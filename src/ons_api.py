"""Download the operator's day-ahead programmed load and verified load (half-hourly)
from the public ONS load API (https://apicarga.ons.org.br)."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

import pandas as pd

from data import PROC, RAW

API = "https://apicarga.ons.org.br/prd/{ep}?dat_inicio={a}&dat_fim={b}&cod_areacarga={area}"
AREAS = {"N": "N", "NE": "NE", "S": "S", "SE": "SECO"}


def _get(url, tries=5):
    for k in range(tries):
        try:
            txt = urllib.request.urlopen(url, timeout=120).read().decode()
            txt = re.sub(r":\s*(,|\})", r": null\1", txt)  # empty values -> null
            txt = re.sub(r"\bNaN\b", "null", txt)
            return json.loads(txt)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"{url}: {err}")


def download(ep: str, start="2020-01-01", end="2026-11-04") -> pd.DataFrame:
    out = os.path.join(RAW, f"api_{ep}.parquet")
    if os.path.exists(out):
        return pd.read_parquet(out)
    rows = []
    months = pd.date_range(start, end, freq="MS")
    for ss, area in AREAS.items():
        for m in months:
            a = m.strftime("%Y-%m-%d")
            b = min(m + pd.offsets.MonthEnd(0), pd.Timestamp(end)).strftime("%Y-%m-%d")
            d = _get(API.format(ep=ep, a=a, b=b, area=area))
            for r in d:
                r["subsystem"] = ss
            rows += d
        print(ep, ss, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    return df


def hourly(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Half-hourly UTC values (end of interval) -> hourly means in Brasilia time
    (UTC-3), indexed by start of hour, to match the ONS hourly load curve."""
    t = pd.to_datetime(df["din_referenciautc"]).dt.tz_convert(None) - pd.Timedelta(hours=3) - pd.Timedelta(minutes=30)
    s = pd.DataFrame({"t": t.dt.floor("h"), "ss": df["subsystem"], "v": pd.to_numeric(df[col], errors="coerce")})
    return s.pivot_table(index="t", columns="ss", values="v", aggfunc="mean")


if __name__ == "__main__":
    os.makedirs(PROC, exist_ok=True)
    prog = download("cargaprogramada")
    ver = download("cargaverificada")
    hp = hourly(prog, "val_cargaglobalprogramada")
    hp.to_parquet(os.path.join(PROC, "ons_programmed_hourly.parquet"))
    for c in ["val_cargaglobal", "val_cargaglobalsmmgd", "val_cargammgd"]:
        hourly(ver, c).to_parquet(os.path.join(PROC, f"ons_verified_{c}.parquet"))
    print(hp.dropna().index.min(), hp.index.max())
