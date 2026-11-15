"""Test-time adaptation of PointMLP (trained on ModelNet40) on ScanObjectNN.

Test sets: ScanObjectNN OBJ_ONLY, OBJ_BG, PB_T50_RS, 11 classes shared with ModelNet40.

Protocol (as in the earlier DGCNN test-time adaptation study): one pass over a shuffled stream in batches
of at most 64, model reset for every run, the same stream for every method, seeds 0 1 2
(the seed sets the stream order and the 2048 -> 1024 subsample). Labels are only used for scoring.

Output: outputs/results/tta_results.csv, tta_summary.json
Usage:  python scripts/test_tta.py [--seeds 0 1 2]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import CKPT  # noqa: E402
from metrics import summarize  # noqa: E402
from pc_utils import MODELNET40_CLASSES, SCANOBJECTNN_SHARED  # noqa: E402
from pointmlp import load_pointmlp  # noqa: E402
from tta import METHODS, Adapter  # noqa: E402

P = ROOT / "data/processed"
R = ROOT / "outputs/results"
VARIANTS = ["obj_only", "obj_bg", "pb_t50_rs"]
GROUPS = [[MODELNET40_CLASSES.index(m) for m in ms] for ms in SCANOBJECTNN_SHARED.values()]


def load(variant, seed):
    """Fixed 2048 -> 1024 subsample per seed, centred and scaled to the unit sphere."""
    with h5py.File(P / f"scanobjectnn_{variant}.h5", "r") as h:
        X, y = h["data"][:], h["label"][:].ravel()
    rng = np.random.default_rng(seed)
    X = np.stack([x[rng.choice(len(x), 1024, replace=False)] for x in X])
    X = X - X.mean(1, keepdims=True)
    X = X / np.linalg.norm(X, axis=2).max(1)[:, None, None]
    return X.astype(np.float32), y


def run(adapter, X, order):
    preds = np.empty(len(X), dtype=np.int64)
    for idx in np.array_split(order, int(np.ceil(len(order) / 64))):
        x = torch.from_numpy(X[idx]).cuda().permute(0, 2, 1)
        preds[idx] = adapter(x).argmax(1).cpu().numpy()
    return preds


def paired_gain(correct_m, correct_s, n=1000, seed=0):
    d = np.mean(correct_m, 0) - np.mean(correct_s, 0)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    model = load_pointmlp(CKPT)
    print("checkpoint:", CKPT.relative_to(ROOT))
    rows, correct = [], {}
    for name in VARIANTS:
        t0 = time.time()
        for seed in a.seeds:
            X, y = load(name, seed)
            order = np.random.default_rng(seed).permutation(len(X))
            torch.manual_seed(seed)
            for method in METHODS:
                pred = run(Adapter(model, method, GROUPS), X, order)
                r = summarize(y, pred, len(GROUPS))
                rows.append(dict(test_set=name, method=method, seed=seed, OA=r["OA"], mAcc=r["mAcc"], n=r["n"]))
                correct.setdefault((name, method), []).append((pred == y).astype(float))
        oa = {m: np.mean([x["OA"] for x in rows if x["test_set"] == name and x["method"] == m]) for m in METHODS}
        print(f"{name:10s} " + "  ".join(f"{m} {100 * v:5.1f}" for m, v in oa.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    R.mkdir(parents=True, exist_ok=True)
    with open(R / "tta_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for (name, method), c in correct.items():
        rs = [x for x in rows if x["test_set"] == name and x["method"] == method]
        e = dict(OA=float(np.mean([x["OA"] for x in rs])), OA_std=float(np.std([x["OA"] for x in rs])),
                 mAcc=float(np.mean([x["mAcc"] for x in rs])), mAcc_std=float(np.std([x["mAcc"] for x in rs])))
        if method != "source_only":
            e["gain"], e["gain_lo"], e["gain_hi"] = paired_gain(c, correct[(name, "source_only")])
        summary.setdefault(name, {})[method] = e
    (R / "tta_summary.json").write_text(json.dumps(summary, indent=2))
    print("saved ->", (R / "tta_summary.json").relative_to(ROOT))
