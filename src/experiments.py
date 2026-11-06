"""Run all experiments of the paper. Predictions are cached in results/preds/.

E1  model comparison (3-year window, weekly refits), 2023 warm-up + 2024-2026 test
E2  calibration window x refit frequency grid (ARX, GBM)
E3  level-normalisation ablation
E4  information-set ablations (holiday features, D-1 morning observations)
E5  quantile GBM for probabilistic forecasts
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # avoid BLAS over-subscription with parallel jobs

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from backtest import ROOT, run  # noqa: E402
from models import ARX, GBM, MLP, Masked, SeasonalNaive, SeasonalNaiveDayType  # noqa: E402

W0, T0, T1 = "2023-01-01", "2024-01-01", "2026-11-04"
best = json.load(open(os.path.join(ROOT, "results", "tuning.json")))["best"]


def _parse(tag):
    return {k: float(v) for k, v in re.findall(r"_([a-z]+)([0-9.e-]+)", "_" + tag.split("_", 2)[2])}


g = _parse(best["GBM"]["tag"])
GBM_KW = dict(learning_rate=g["lr"], num_leaves=int(g["nl"]), n_estimators=int(g["ne"]), min_child_samples=int(g["mc"]))
ARX_A = _parse(best["ARX"]["tag"])["a"]
mt = best["MLP"]["tag"]
MLP_H = tuple(int(x) for x in re.search(r"_h([0-9-]+)_", mt).group(1).split("-"))
MLP_A = float(re.search(r"_a([0-9.e-]+)$", mt).group(1))

mk = {
    "SN7": SeasonalNaive,
    "SNDT": SeasonalNaiveDayType,
    "ARX": lambda: ARX(ARX_A),
    "GBM": lambda: GBM(**GBM_KW),
    "MLP": lambda: MLP(hidden=MLP_H, alpha=MLP_A),
}
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def jobs():
    J = []
    # E1
    for m in ["SN7", "SNDT", "ARX", "GBM", "MLP"]:
        J.append((f"E1_{m}", mk[m], W0, T1, 3, "W", True))
    # E2 (test period only)
    for m, freqs in [("ARX", ["D", "W", "M", "Q", "Y"]), ("GBM", ["W", "M", "Q", "Y"])]:
        for w in [1, 2, 3, 5, 8, None]:
            for f in freqs:
                if m == "GBM" and f == "W" and w not in (1, 3):
                    continue  # computational budget: weekly GBM refits only for short windows
                J.append((f"E2_{m}_w{w if w else 'exp'}_{f}", mk[m], T0, T1, w, f, True))
    # E3
    for m in ["ARX", "GBM", "MLP"]:
        J.append((f"E3_{m}_raw", mk[m], T0, T1, 3, "W", False))
    J.append(("E3_GBM_raw_wexp", mk["GBM"], T0, T1, None, "M", False))
    # E4
    hol = ["holiday", "carnival", "ash_wed", "eve", "bridge", "yearend", "off_l"]
    for m in ["ARX", "GBM"]:
        J.append((f"E4_{m}_nohol", lambda m=m: Masked(mk[m](), hol, 0.0), T0, T1, 3, "W", True))
        J.append((f"E4_{m}_nod1", lambda m=m: Masked(mk[m](), ["l1_", "d1_morning"], float("nan")), T0, T1, 3, "W", True))
    # E5
    for q in QUANTILES:
        J.append((f"E5_GBMq{q}", lambda q=q: GBM(objective="quantile", alpha=q, **GBM_KW), W0, T1, 3, "M", True))
    # E6: head-to-head with the operator's official day-ahead forecast (global load, 2020-2026)
    for m in ["SN7", "ARX", "GBM", "MLP"]:
        J.append((f"E6_{m}", mk[m], W0, T1, None, "W", True, "global"))
    return J


def _do(j):
    tag, f, a, b, w, fr, norm = j[:7]
    run(tag, f, a, b, w, fr, norm, dataset=j[7] if len(j) > 7 else "curva")
    return tag


if __name__ == "__main__":
    J = jobs()
    if len(sys.argv) > 1:
        J = [j for j in J if re.match(sys.argv[1], j[0])]
    # heavy jobs first for better load balancing
    def cost(j):
        heavy = ("GBM" in j[0]) * 5 + ("MLP" in j[0]) * 3 + 1
        nref = {"D": 1000, "W": 150, "M": 35, "Q": 12, "Y": 3}[j[5]]
        w = 13 if j[4] is None else j[4]
        return -heavy * nref * w
    J.sort(key=cost)
    Parallel(n_jobs=4, verbose=0)(delayed(_do)(j) for j in J)
