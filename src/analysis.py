"""Produce all tables (LaTeX) and figures of the paper from cached predictions."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import PROC, ROOT, SUBSYSTEMS, get_global_load, get_load  # noqa: E402
from features import day_calendar  # noqa: E402
from metrics import daily_loss, dm_test, holm, interval_scores, pinball, point_metrics  # noqa: E402
from prob import aci_interval, eqr_quantiles  # noqa: E402

PRED = os.path.join(ROOT, "results", "preds")
TAB = os.path.join(ROOT, "paper", "tables")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
T0, T1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-11-04")
SSN = {"N": "North", "NE": "Northeast", "S": "South", "SE": "Southeast/C.-West"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4})
COL = {"N": "#1b9e77", "NE": "#d95f02", "S": "#7570b3", "SE": "#e7298a"}
SUMMARY = {}
NUMBERS = {}


def write_numbers():
    f = os.path.join(TAB, "numbers.tex")
    old = {}
    if os.path.exists(f):
        for line in open(f):
            if line.startswith("\\newcommand"):
                k = line.split("{")[1].split("}")[0]
                old[k] = line
    for k, v in NUMBERS.items():
        old["\\" + k] = "\\newcommand{\\" + k + "}{" + str(v) + "}\n"
    with open(f, "w") as fh:
        fh.writelines(sorted(old.values()))


def load_pred(tag, test_only=True):
    f = os.path.join(PRED, f"{tag}.parquet")
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f)
    d["date"] = pd.to_datetime(d["date"])
    if test_only:
        d = d[(d.date >= T0) & (d.date <= T1)]
    return d


def ensemble(tags, name="ENS", test_only=True):
    ds = [load_pred(t, test_only) for t in tags]
    if any(d is None for d in ds):
        return None
    k = ["subsystem", "date", "hour"]
    base = ds[0][k + ["y"]].copy()
    P = np.mean([d.set_index(k).loc[base.set_index(k).index, "pred"].values for d in ds], axis=0)
    base["pred"] = P
    return base


def mape_by_ss(d):
    return ((d.pred - d.y).abs() / d.y).groupby(d.subsystem).mean() * 100


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def bold_min(vals, nd=2):
    m = np.nanmin(vals)
    return [("\\textbf{" + fmt(v, nd) + "}") if np.isclose(v, m) else fmt(v, nd) for v in vals]


# ------------------------------------------------------------------ data figs
def fig_data():
    load = get_load()
    ann = load.resample("YE").mean()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for ss in SUBSYSTEMS:
        ax[0].plot(ann.index.year, ann[ss] / 1000, marker="o", ms=2.5, lw=1, color=COL[ss], label=SSN[ss])
    ax[0].set_ylabel("Mean load (GW)")
    ax[0].set_yscale("log")
    ax[0].set_yticks([4, 6, 10, 15, 25, 40])
    ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_title("(a) Annual mean load per subsystem")
    ax[0].legend(ncol=2, loc="center left", bbox_to_anchor=(0.0, 0.66), frameon=False)
    mm = pd.read_parquet(os.path.join(PROC, "ons_verified_val_cargammgd.parquet"))
    gl = pd.read_parquet(os.path.join(PROC, "ons_verified_val_cargaglobal.parquet"))
    sh = (mm / gl)
    mid = sh[(sh.index.hour >= 11) & (sh.index.hour <= 14)].resample("MS").mean() * 100
    for ss in SUBSYSTEMS:
        ax[1].plot(mid.index, mid[ss], lw=1, color=COL[ss], label=SSN[ss])
    ax[1].set_ylabel("MMGD share of load (%)")
    ax[1].set_title("(b) Estimated behind-the-meter DG share, 11:00-14:59")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_data.pdf"))
    plt.close(fig)
    SUMMARY["mmgd_mid_share_last12m"] = mid.iloc[-12:].mean().round(1).to_dict()
    SUMMARY["mmgd_mid_share_2021"] = mid.loc["2021"].mean().round(1).to_dict()
    a, b = mid.loc["2021"].mean(), mid.iloc[-12:].mean()
    NUMBERS["MMGDfirst"] = f"{a.min():.0f}--{a.max():.0f}\\%"
    NUMBERS["MMGDlast"] = f"{b.min():.0f}--{b.max():.0f}\\%"
    # daily profiles
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.8), sharey=True)
    for i, ss in enumerate(SUBSYSTEMS):
        for yr, ls in [(2016, ":"), (2021, "--"), (2025, "-")]:
            s = load.loc[str(yr), ss]
            s = s[s.index.dayofweek < 5]
            prof = s.groupby(s.index.hour).mean()
            ax[i].plot(prof.index, prof / prof.mean(), ls=ls, color=COL[ss], lw=1, label=str(yr))
        ax[i].set_title(SSN[ss])
        ax[i].set_xticks([0, 6, 12, 18, 23])
        ax[i].set_xlabel("Hour")
    ax[0].set_ylabel("Normalised load")
    ax[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_profiles.pdf"))
    plt.close(fig)
    # dataset table
    rows = []
    for ss in SUBSYSTEMS:
        s = load[ss]
        g = (s.loc["2025"].mean() / s.loc["2016"].mean() - 1) * 100
        rows.append([SSN[ss], fmt(s.mean() / 1000, 1), fmt(s.min() / 1000, 1), fmt(s.max() / 1000, 1),
                     fmt(s.loc["2025"].mean() / 1000, 1), fmt(g, 1)])
    m = pd.read_parquet(os.path.join(PROC, "clean_mask.parquet"))
    with open(os.path.join(TAB, "tab_data.tex"), "w") as f:
        for r in rows:
            f.write(" & ".join(r) + r" \\" + "\n")
    SUMMARY["n_hours"] = int(len(load))
    SUMMARY["cleaned_entries"] = m.sum().to_dict()
    SUMMARY["cleaned_frac_pct"] = float(m.values.mean() * 100)
    NUMBERS["NHours"] = f"{len(load):,}"
    NUMBERS["CleanedPct"] = f"{m.values.mean() * 100:.2f}"


# ------------------------------------------------------------------- E1
E1 = ["SN7", "SNDT", "ARX", "GBM", "MLP", "ENS"]


def e1():
    D = {m: load_pred(f"E1_{m}") for m in E1[:-1]}
    D["ENS"] = ensemble(["E1_ARX", "E1_GBM", "E1_MLP"])
    rows = {}
    for m, d in D.items():
        if d is None:
            continue
        r = {}
        for ss in SUBSYSTEMS:
            pm = point_metrics(d[d.subsystem == ss])
            r[(ss, "MAPE")] = pm.MAPE
            r[(ss, "RMSE")] = pm.RMSE
        r[("All", "MAPE")] = np.mean([r[(ss, "MAPE")] for ss in SUBSYSTEMS])
        rows[m] = r
    T = pd.DataFrame(rows).T
    T.to_csv(os.path.join(ROOT, "results", "E1.csv"))
    with open(os.path.join(TAB, "tab_e1.tex"), "w") as f:
        cols = list(T.columns)
        best = {c: T[c].min() for c in cols}
        for m in T.index:
            cells = []
            for c in cols:
                v = T.loc[m, c]
                nd = 0 if c[1] == "RMSE" else 2
                s = fmt(v, nd)
                cells.append("\\textbf{" + s + "}" if np.isclose(v, best[c]) else s)
            f.write(f"{m} & " + " & ".join(cells) + r" \\" + "\n")
    SUMMARY["E1"] = T.round(3).to_dict(orient="index")
    SUMMARY["E1"] = {k: {f"{a}_{b}": v for (a, b), v in d.items()} for k, d in SUMMARY["E1"].items()}
    # DM tests (daily MAPE, one-sided), Holm across all comparisons per subsystem
    models = [m for m in E1 if D.get(m) is not None]
    out = []
    for ss in SUBSYSTEMS:
        L = {m: daily_loss(D[m][D[m].subsystem == ss]) for m in models}
        P = {}
        for a in models:
            for b in models:
                if a != b:
                    P[(a, b)] = dm_test(L[a], L[b])[1]
        A = holm(P)
        for (a, b), p in A.items():
            out.append((ss, a, b, p))
    DM = pd.DataFrame(out, columns=["ss", "a", "b", "p"])
    DM.to_csv(os.path.join(ROOT, "results", "E1_dm.csv"), index=False)
    # compact table: p-value of row model better than column model, all subsystems pooled by showing count of significant
    with open(os.path.join(TAB, "tab_dm.tex"), "w") as f:
        for a in models:
            cells = []
            for b in models:
                if a == b:
                    cells.append("--")
                    continue
                sig = DM[(DM.a == a) & (DM.b == b) & (DM.p < 0.05)].ss.tolist()
                cells.append(str(len(sig)) if sig else "0")
            f.write(f"{a} & " + " & ".join(cells) + r" \\" + "\n")
    SUMMARY["DM_ENS_vs_ARX"] = DM[(DM.a == "ENS") & (DM.b == "ARX")].set_index("ss").p.round(4).to_dict()
    SUMMARY["DM_ARX_vs_GBM"] = DM[(DM.a == "ARX") & (DM.b == "GBM")].set_index("ss").p.round(4).to_dict()
    return D


# ---------------------------------------------------------- error analyses
def e1_breakdowns(D):
    best = "ENS"
    load = get_load()
    days = pd.DatetimeIndex(sorted(D[best].date.unique()))
    cal = day_calendar(days)
    typ = pd.Series("Regular weekday", index=days)
    typ[cal.dow >= 5] = "Weekend"
    typ[cal.bridge == 1] = "Bridge day"
    typ[cal.yearend == 1] = "Year-end period"
    typ[cal.eve == 1] = "Christmas/NY eve"
    typ[cal.ash_wed == 1] = "Ash Wednesday"
    typ[cal.holiday == 1] = "National holiday"
    typ[cal.carnival == 1] = "Carnival"
    order = ["Regular weekday", "Weekend", "National holiday", "Carnival", "Ash Wednesday", "Bridge day",
             "Christmas/NY eve", "Year-end period"]
    rows = []
    for t in order:
        dd = days[typ == t]
        r = [t, str(len(dd))]
        for m in ["SN7", "ARX", "GBM", "MLP", "ENS"]:
            d = D[m]
            d = d[d.date.isin(dd)]
            r.append(((d.pred - d.y).abs() / d.y).mean() * 100)
        rows.append(r)
    with open(os.path.join(TAB, "tab_daytype.tex"), "w") as f:
        for r in rows:
            f.write(" & ".join(r[:2] + bold_min(r[2:])) + r" \\" + "\n")
    SUMMARY["daytype"] = {r[0]: [round(x, 2) for x in r[2:]] for r in rows}
    # by hour of day and monthly (fig)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for m, c, ls in [("SN7", "0.6", ":"), ("GBM", "#1f78b4", "--"), ("MLP", "#33a02c", "-."), ("ARX", "#e31a1c", "-"),
                     ("ENS", "k", "-")]:
        d = D[m]
        ape = (d.pred - d.y).abs() / d.y * 100
        ax[0].plot(range(24), ape.groupby(d.hour).mean(), color=c, ls=ls, lw=1, label=m)
        mo = ape.groupby([d.date.dt.to_period("M")]).mean()
        ax[1].plot(mo.index.to_timestamp(), mo.values, color=c, ls=ls, lw=1, label=m)
    ax[0].set_xlabel("Hour of day")
    ax[0].set_ylabel("MAPE (%)")
    ax[0].set_title("(a) By hour of day (all subsystems)")
    ax[0].set_xticks([0, 6, 12, 18, 23])
    ax[1].set_ylabel("MAPE (%)")
    ax[1].set_title("(b) Monthly MAPE (all subsystems)")
    ax[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_err_hour_month.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------- E2
def e2():
    res = {}
    for m, freqs in [("ARX", ["D", "W", "M", "Q", "Y"]), ("GBM", ["W", "M", "Q", "Y"])]:
        T = pd.DataFrame(index=["1", "2", "3", "5", "8", "exp"], columns=freqs, dtype=float)
        for w in T.index:
            for fr in freqs:
                d = load_pred(f"E2_{m}_w{w}_{fr}")
                if d is not None:
                    T.loc[w, fr] = mape_by_ss(d).mean()
        res[m] = T
    # window averaging (Hubicka et al.) for ARX weekly and GBM monthly
    avg = {}
    for m, fr in [("ARX", "W"), ("GBM", "M")]:
        e = ensemble([f"E2_{m}_w{w}_{fr}" for w in ["1", "2", "3", "5", "8", "exp"]])
        if e is not None:
            avg[m] = mape_by_ss(e).mean()
            for ws, nm in [(["1", "3", "exp"], "avg_1_3_exp"), (["2", "exp"], "avg_2_exp")]:
                e2_ = ensemble([f"E2_{m}_w{w}_{fr}" for w in ws])
                avg[f"{m}_{nm}"] = mape_by_ss(e2_).mean()
    SUMMARY["E2"] = {m: T.round(3).to_dict() for m, T in res.items()}
    SUMMARY["E2_window_avg"] = {k: round(v, 3) for k, v in avg.items()}
    lab = {"1": "1 yr", "2": "2 yr", "3": "3 yr", "5": "5 yr", "8": "8 yr", "exp": "Expanding"}
    fn = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly", "Y": "Yearly"}
    with open(os.path.join(TAB, "tab_e2.tex"), "w") as f:
        allv = pd.concat([res["ARX"].stack(), res["GBM"].stack()])
        for w in res["ARX"].index:
            cells = [lab[w]]
            for m in ["ARX", "GBM"]:
                T = res[m]
                for fr in ["D", "W", "M", "Q", "Y"]:
                    if fr not in T.columns:
                        continue
                    v = T.loc[w, fr]
                    s = "--" if np.isnan(v) else fmt(v)
                    if not np.isnan(v) and np.isclose(v, T.min().min()):
                        s = "\\textbf{" + s + "}"
                    cells.append(s)
            f.write(" & ".join(cells) + r" \\" + "\n")
    return res


# ------------------------------------------------------------------- E3/E4
def e3e4():
    rows = []
    base = {"ARX": "E1_ARX", "GBM": "E1_GBM", "MLP": "E1_MLP"}
    variants = [("Proposed setting (normalised, full information)", base),
                ("Raw target and inputs (no level normalisation)", {m: f"E3_{m}_raw" for m in base}),
                ("Without holiday/special-day features", {"ARX": "E4_ARX_nohol", "GBM": "E4_GBM_nohol"}),
                ("Without D-1 morning observations", {"ARX": "E4_ARX_nod1", "GBM": "E4_GBM_nod1"})]
    out = {}
    for name, tags in variants:
        r = [name]
        for m in ["ARX", "GBM", "MLP"]:
            d = load_pred(tags[m]) if m in tags else None
            r.append("--" if d is None else fmt(mape_by_ss(d).mean()))
            if d is not None:
                out[(name, m)] = mape_by_ss(d).round(3).to_dict()
        rows.append(r)
    d = load_pred("E3_GBM_raw_wexp")
    if d is not None:
        out[("raw_expanding", "GBM")] = mape_by_ss(d).round(3).to_dict()
    with open(os.path.join(TAB, "tab_ablation.tex"), "w") as f:
        for r in rows:
            f.write(" & ".join(r) + r" \\" + "\n")
    SUMMARY["ablation"] = {f"{a} | {b}": v for (a, b), v in out.items()}


# ------------------------------------------------------------------- E5
def e5(D):
    taus = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    ens_full = ensemble(["E1_ARX", "E1_GBM", "E1_MLP"], test_only=False)
    # select EQR window and ACI step on 2023-07..2023-12 (validation)
    V0, V1 = pd.Timestamp("2023-07-01"), pd.Timestamp("2023-12-31")
    sel = {}
    for W in [28, 56, 91, 182]:
        sc = []
        for ss in SUBSYSTEMS:
            d = ens_full[ens_full.subsystem == ss]
            Q = eqr_quantiles(d, taus, W)
            Y = d.pivot(index="date", columns="hour", values="y")
            msk = (Y.index >= V0) & (Y.index <= V1)
            sc.append(np.mean([pinball(Y[msk].values, Q[t][msk].values, t).mean() / Y[msk].values.mean() for t in taus]))
        sel[W] = np.mean(sc)
    Wb = min(sel, key=sel.get)
    gsel = {}
    for gam in [0.002, 0.005, 0.01, 0.02, 0.05]:
        sc = []
        for ss in SUBSYSTEMS:
            d = ens_full[ens_full.subsystem == ss]
            lo, hi = aci_interval(d, 0.1, Wb, gam)
            Y = d.pivot(index="date", columns="hour", values="y")
            msk = (Y.index >= V0) & (Y.index <= V1)
            _, _, wk = interval_scores(Y[msk].values, lo[msk].values, hi[msk].values, 0.1)
            sc.append(np.nanmean(wk) / Y[msk].values.mean())
        gsel[gam] = np.mean(sc)
    gb = min(gsel, key=gsel.get)
    SUMMARY["E5_select"] = {"eqr_window": Wb, "eqr_scores": sel, "aci_gamma": gb, "aci_scores": gsel}
    # evaluate on test
    qg = {t: load_pred(f"E5_GBMq{t}", test_only=False) for t in taus}
    rows = []
    cov_ts = {}
    for name in ["QGBM", "EQR", "ACI"]:
        acc = {"cov80": [], "cov90": [], "w80": [], "w90": [], "wk80": [], "wk90": [], "pin": []}
        for ss in SUBSYSTEMS:
            d = ens_full[ens_full.subsystem == ss]
            Y = d.pivot(index="date", columns="hour", values="y")
            msk = (Y.index >= T0) & (Y.index <= T1)
            if name == "QGBM":
                if any(v is None for v in qg.values()):
                    break
                Q = {t: qg[t][qg[t].subsystem == ss].pivot(index="date", columns="hour", values="pred").reindex(Y.index)
                     for t in taus}
                arr = np.sort(np.stack([Q[t].values for t in taus]), axis=0)  # fix crossing
                Q = {t: pd.DataFrame(arr[i], index=Y.index, columns=Y.columns) for i, t in enumerate(taus)}
            else:
                Q = eqr_quantiles(d, taus, Wb)
            yv = Y[msk].values
            scale = yv.mean()
            acc["pin"].append(np.mean([pinball(yv, Q[t][msk].values, t).mean() for t in taus]) / scale * 100)
            for a, lo_t, hi_t in [(0.2, 0.1, 0.9), (0.1, 0.05, 0.95)]:
                if name == "ACI":
                    lo, hi = aci_interval(d, a, Wb, gb)
                    lo, hi = lo[msk].values, hi[msk].values
                else:
                    lo, hi = Q[lo_t][msk].values, Q[hi_t][msk].values
                c, w, wk = interval_scores(yv, lo, hi, a)
                k = "80" if a == 0.2 else "90"
                acc["cov" + k].append(c * 100)
                acc["w" + k].append(np.nanmean(w) / scale * 100)
                acc["wk" + k].append(np.nanmean(wk) / scale * 100)
                if k == "90":
                    cov_ts.setdefault(name, []).append(pd.Series(((yv >= lo) & (yv <= hi)).mean(axis=1),
                                                                 index=Y.index[msk]))
        if not acc["cov80"] or len(acc["cov80"]) < 4:
            continue
        rows.append([name] + [np.mean(acc[k]) for k in ["cov80", "w80", "wk80", "cov90", "w90", "wk90"]] +
                    ([np.mean(acc["pin"])] if name != "ACI" else [np.nan]))
    with open(os.path.join(TAB, "tab_prob.tex"), "w") as f:
        for r in rows:
            cells = [r[0]] + [fmt(v, 1) if i in (0, 3) else fmt(v) for i, v in enumerate(r[1:7])]
            cells.append("--" if np.isnan(r[7]) else fmt(r[7], 3))
            f.write(" & ".join(cells) + r" \\" + "\n")
    SUMMARY["E5"] = {r[0]: [None if np.isnan(x) else round(float(x), 3) for x in r[1:]] for r in rows}
    # coverage over time
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    for name, c in [("QGBM", "#1f78b4"), ("EQR", "#ff7f00"), ("ACI", "k")]:
        if name in cov_ts:
            s = pd.concat(cov_ts[name], axis=1).mean(axis=1).rolling(30).mean() * 100
            ax.plot(s.index, s.values, color=c, lw=1, label=name)
    ax.axhline(90, color="r", lw=0.8, ls="--")
    ax.set_ylabel("90% PI coverage (%)\n(30-day rolling)")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_coverage.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------- E6
def e6():
    prog = pd.read_parquet(os.path.join(PROC, "ons_programmed_hourly.parquet"))
    D = {m: load_pred(f"E6_{m}") for m in ["SN7", "ARX", "GBM", "MLP"]}
    if any(v is None for v in D.values()):
        return
    D["ENS"] = ensemble(["E6_ARX", "E6_GBM", "E6_MLP"])
    # ONS official forecast in the same long format
    base = D["ARX"][["subsystem", "date", "hour", "y"]].copy()
    t = base.date + pd.to_timedelta(base.hour, unit="h")
    base["pred"] = [prog.at[tt, ss] if tt in prog.index else np.nan for tt, ss in zip(t, base.subsystem)]
    # require complete official forecast on the day
    ok = base.groupby(["subsystem", "date"]).pred.transform(lambda s: s.notna().all())
    keys = base.loc[ok, ["subsystem", "date"]].drop_duplicates()
    D["ONS"] = base[ok]
    k = ["subsystem", "date", "hour"]
    for m in D:
        D[m] = D[m].merge(keys, on=["subsystem", "date"])
    SUMMARY["E6_days_common"] = {ss: int((keys.subsystem == ss).sum()) for ss in SUBSYSTEMS}
    nd = sorted(SUMMARY["E6_days_common"].values())
    NUMBERS["NdaysONS"] = f"{nd[0]:,}" if nd[0] == nd[-1] else f"{nd[0]:,}--{nd[-1]:,}"
    rows = {}
    for m in ["ONS", "SN7", "ARX", "GBM", "MLP", "ENS"]:
        d = D[m]
        r = {}
        for ss in SUBSYSTEMS:
            pm = point_metrics(d[d.subsystem == ss])
            r[(ss, "MAPE")] = pm.MAPE
            r[(ss, "RMSE")] = pm.RMSE
        r[("All", "MAPE")] = np.mean([r[(ss, "MAPE")] for ss in SUBSYSTEMS])
        rows[m] = r
    T = pd.DataFrame(rows).T
    T.to_csv(os.path.join(ROOT, "results", "E6.csv"))
    # DM: model vs ONS, two one-sided p-values
    dm = {}
    for m in ["ARX", "GBM", "MLP", "ENS"]:
        for ss in SUBSYSTEMS:
            la = daily_loss(D[m][D[m].subsystem == ss])
            lb = daily_loss(D["ONS"][D["ONS"].subsystem == ss])
            dm[(m, ss)] = (dm_test(la, lb)[1], dm_test(lb, la)[1])
    SUMMARY["E6"] = {m: {f"{a}_{b}": round(v, 3) for (a, b), v in r.items()} for m, r in T.to_dict(orient="index").items()}
    SUMMARY["E6_dm_model_better_p"] = {f"{m}_{ss}": round(v[0], 4) for (m, ss), v in dm.items()}
    SUMMARY["E6_dm_ons_better_p"] = {f"{m}_{ss}": round(v[1], 4) for (m, ss), v in dm.items()}
    with open(os.path.join(TAB, "tab_e6.tex"), "w") as f:
        cols = list(T.columns)
        best = {c: T[c].min() for c in cols}
        for m in T.index:
            cells = []
            for c in cols:
                v = T.loc[m, c]
                s = fmt(v, 0 if c[1] == "RMSE" else 2)
                if m in dm and c[0] in SUBSYSTEMS and c[1] == "MAPE":
                    p_better, p_worse = dm[(m, c[0])]
                    s += "$^{+}$" if p_better < 0.05 else ("$^{-}$" if p_worse < 0.05 else "")
                cells.append("\\textbf{" + s + "}" if np.isclose(v, best[c]) else s)
            name = "ONS (official)" if m == "ONS" else m
            f.write(f"{name} & " + " & ".join(cells) + r" \\" + "\n")
    # by hour and per year
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=False)
    for i, ss in enumerate(SUBSYSTEMS):
        for m, c, ls in [("ONS", "#e31a1c", "-"), ("ENS", "k", "-"), ("ARX", "#1f78b4", "--")]:
            d = D[m][D[m].subsystem == ss]
            ape = (d.pred - d.y).abs() / d.y * 100
            ax[i].plot(range(24), ape.groupby(d.hour).mean(), color=c, ls=ls, lw=1, label="ONS official" if m == "ONS" else m)
        ax[i].set_title(SSN[ss])
        ax[i].set_xticks([0, 6, 12, 18, 23])
        ax[i].set_xlabel("Hour")
    ax[0].set_ylabel("MAPE (%)")
    ax[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ons_hour.pdf"))
    plt.close(fig)
    yr = {}
    for m in ["ONS", "ENS", "ARX"]:
        d = D[m]
        yr[m] = ((d.pred - d.y).abs() / d.y * 100).groupby([d.date.dt.year, d.subsystem]).mean().unstack().round(2).to_dict(orient="index")
    SUMMARY["E6_by_year"] = {m: {str(k): v for k, v in r.items()} for m, r in yr.items()}
    # bias of ONS by hour (midday)
    d = D["ONS"]
    SUMMARY["E6_ons_bias_by_hour_pct"] = ((d.pred - d.y) / d.y * 100).groupby(d.hour).mean().round(2).to_dict()
    d = D["ENS"]
    SUMMARY["E6_ens_bias_by_hour_pct"] = ((d.pred - d.y) / d.y * 100).groupby(d.hour).mean().round(2).to_dict()
    # midday (10-14 h) bias vs behind-the-meter share, monthly
    mmgd = pd.read_parquet(os.path.join(PROC, "ons_verified_val_cargammgd.parquet"))
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True)
    corr = {}
    for i, ss in enumerate(SUBSYSTEMS):
        res = {}
        for m in ["ONS", "ENS"]:
            d = D[m][(D[m].subsystem == ss) & D[m].hour.between(10, 14)]
            g = ((d.pred - d.y) / d.y * 100).groupby(d.date.dt.to_period("M"))
            res[m] = g.mean()[g.size() >= 20 * 5]  # complete months only
        d = D["ONS"][(D["ONS"].subsystem == ss) & D["ONS"].hour.between(10, 14)]
        t = d.date + pd.to_timedelta(d.hour, unit="h")
        share = pd.Series(mmgd.reindex(t)[ss].values / d.y.values * 100, index=d.index).groupby(d.date.dt.to_period("M")).mean()
        share = share.reindex(res["ONS"].index)
        x = res["ONS"].index.to_timestamp()
        ax[i].axhline(0, color="0.6", lw=0.5)
        ax[i].plot(x, res["ONS"].values, color="#e31a1c", lw=1, label="ONS official")
        ax[i].plot(x, res["ENS"].values, color="k", lw=1, label="ENS")
        a2 = ax[i].twinx()
        a2.plot(x, share.values, color="#ff7f00", lw=0.8, ls=":", label="BTM share")
        a2.set_ylim(0, 60)
        if i < 3:
            a2.set_yticklabels([])
        else:
            a2.set_ylabel("BTM share (%)", fontsize=7)
        a2.tick_params(labelsize=6)
        ax[i].set_title(SSN[ss])
        ax[i].xaxis.set_major_locator(mdates.YearLocator())
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax[i].tick_params(axis="x", labelsize=6)
        corr[ss] = round(float(np.corrcoef(res["ONS"].values, share.values)[0, 1]), 2)
    ax[0].set_ylabel("Midday bias (%)")
    h1, l1 = ax[0].get_legend_handles_labels()
    ax[0].legend(h1 + [a2.lines[0]], l1 + ["BTM share"], frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ons_bias.pdf"))
    plt.close(fig)
    SUMMARY["E6_midday_bias_share_corr"] = corr
    d = D["ONS"][D["ONS"].hour.between(10, 14)]
    SUMMARY["E6_ons_midday_bias_by_year"] = ((d.pred - d.y) / d.y * 100).groupby([d.date.dt.year, d.subsystem]).mean().unstack().round(2).to_dict(orient="index")
    d = D["ENS"][D["ENS"].hour.between(10, 14)]
    SUMMARY["E6_ens_midday_bias_by_year"] = ((d.pred - d.y) / d.y * 100).groupby([d.date.dt.year, d.subsystem]).mean().unstack().round(2).to_dict(orient="index")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["data", "e1", "e2", "e3e4", "e5", "e6"]
    sf = os.path.join(ROOT, "results", "summary.json")
    if os.path.exists(sf):
        SUMMARY.update(json.load(open(sf)))
    D = None
    if "data" in steps:
        fig_data()
    if "e1" in steps or "e5" in steps:
        D = e1()
        e1_breakdowns(D)
    if "e2" in steps:
        e2()
    if "e3e4" in steps:
        e3e4()
    if "e5" in steps:
        e5(D)
    if "e6" in steps:
        e6()
    json.dump(SUMMARY, open(sf, "w"), indent=1, default=str)
    write_numbers()
    print(json.dumps(SUMMARY, indent=1, default=str)[:6000])
