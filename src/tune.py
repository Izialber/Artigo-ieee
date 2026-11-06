"""Hyper-parameter selection on the 2023 validation year (3-year window, monthly refits)."""
import itertools
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from backtest import ROOT, run  # noqa: E402
from models import ARX, GBM, MLP  # noqa: E402

V0, V1 = "2023-01-01", "2023-12-31"


def score(tag, mk):
    df = run(tag, mk, V0, V1, 3, "M", verbose=False)
    return tag, float(((df.pred - df.y).abs() / df.y).groupby(df.subsystem).mean().mean() * 100)


def main():
    jobs = []
    for a in [0.3, 1, 3, 10, 30, 100, 300]:
        jobs.append((f"val_ARX_a{a}", lambda a=a: ARX(a)))
    for lr, nl, ne, mc in itertools.product([0.03, 0.06], [15, 31, 63], [400, 800], [20, 80]):
        jobs.append((f"val_GBM_lr{lr}_nl{nl}_ne{ne}_mc{mc}",
                     lambda lr=lr, nl=nl, ne=ne, mc=mc: GBM(learning_rate=lr, num_leaves=nl, n_estimators=ne, min_child_samples=mc)))
    for hd, al in itertools.product([(64,), (128, 64), (256, 128)], [1e-4, 1e-3, 1e-2, 1e-1]):
        jobs.append((f"val_MLP_h{'-'.join(map(str, hd))}_a{al}", lambda hd=hd, al=al: MLP(hidden=hd, alpha=al)))
    res = dict(Parallel(n_jobs=4)(delayed(score)(t, m) for t, m in jobs))
    best = {}
    for fam in ["ARX", "GBM", "MLP"]:
        r = {k: v for k, v in res.items() if k.startswith(f"val_{fam}_")}
        k = min(r, key=r.get)
        best[fam] = {"tag": k, "mape": r[k]}
        print(fam, "best", k, round(r[k], 3))
    json.dump({"all": res, "best": best}, open(os.path.join(ROOT, "results", "tuning.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
