"""Evaluation helpers: test-set loading, inference, metrics and confusion matrices."""
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from metrics import confusion, summarize  # noqa: E402
from pc_utils import MODELNET10_CLASSES, MODELNET40_CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
CKPT = ROOT / "checkpoints/pointMLP_baseline_best.pth"
MN10_IN_MN40 = [MODELNET40_CLASSES.index(c) for c in MODELNET10_CLASSES]
PRETTY_SET = {"full_1024": "complete objects, 1024 points", "full_2048": "complete objects, 2048 points",
              "hpr_1024": "single-view HPR, 1024 points", "hpr_2048": "single-view HPR, 2048 points",
              "hpr_raw": "single-view HPR, all visible points"}
PRETTY_SET.update({f"removal_{m}_{r}": f"{m} removal {r}%" for m in ["uniform", "contiguous"]
                   for r in [10, 20, 30, 40, 50]})
REMOVAL_SETS = [f"removal_{m}_{r}" for m in ["uniform", "contiguous"] for r in [10, 20, 30, 40, 50]]
PRETTY_DS = {"modelnet40": "ModelNet40", "modelnet10": "ModelNet10"}


def pretty(ds, s):
    return f"{PRETTY_DS[ds]} — {PRETTY_SET[s]}"


def load_set(ds, s):
    """ds in {modelnet40, modelnet10}; s in {full_1024, full_2048, hpr_1024, hpr_2048, hpr_raw}.
    Returns (X, y): X is an array [N,K,3], or a list of [n_i,3] arrays for hpr_raw (unit-sphere normalised)."""
    with h5py.File(P / f"{ds}_test_{s}.h5", "r") as h:
        y = h["label"][:, 0]
        if s.endswith("raw"):
            pts, off = h["points"][:], h["offsets"][:]
            X = []
            for i in range(len(y)):
                p = pts[off[i]:off[i + 1]]
                p = p - p.mean(0)
                X.append((p / np.linalg.norm(p, axis=1).max()).astype(np.float32))
            return X, y
        return h["data"][:], y


@torch.no_grad()
def predict(model, X, bs=32):
    """PointMLP logits. Fixed-size arrays go in batches; variable-size clouds (lists) one at a time."""
    out = []
    if isinstance(X, list):
        for p in X:
            out.append(model(torch.from_numpy(p).cuda().T[None]).cpu())
    else:
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs]).cuda().permute(0, 2, 1)).cpu())
    return torch.cat(out).numpy()


def run(model, tests, tag):
    """Evaluate `tests` = [(dataset, set), ...]. Returns (results, per_class, predictions)."""
    res, per_class, preds = {}, {}, {}
    for ds, s in tests:
        X, y = load_set(ds, s)
        logits = predict(model, X)
        if ds == "modelnet10":  # use only the outputs of the ten ModelNet10 classes
            logits = logits[:, MN10_IN_MN40]
        n = 10 if ds == "modelnet10" else 40
        pr = logits.argmax(1)
        r = summarize(y, pr, n)
        per_class[f"{ds}/{s}"] = r.pop("per_class")
        res[f"{ds}/{s}"] = r
        preds[f"{ds}_{s}"] = pr
        print(f"[{tag}] {ds:10s} {s:10s} OA {r['OA']:.2%}  mAcc {r['mAcc']:.2%}  (n={r['n']})")
        if s in ("full_1024", "hpr_1024"):
            classes = MODELNET10_CLASSES if ds == "modelnet10" else MODELNET40_CLASSES
            plot_confusion(confusion(y, pr, n), classes, f"PointMLP, {'supervised' if tag == 'supervised' else 'zero-shot'}\n{pretty(ds, s)}",
                           ROOT / f"outputs/figures/confusion_{tag}_{ds}_{s}.png")
    return res, per_class, preds


def save(res, per_class, preds, tag):
    """Save results. Test sets that were not evaluated in this run keep their previous entries."""
    R = ROOT / "outputs/results"
    R.mkdir(parents=True, exist_ok=True)
    jf, nf = R / f"{tag}_results.json", R / f"{tag}_predictions.npz"
    merged = json.loads(jf.read_text()) if jf.exists() else {}
    merged.update({"checkpoint": str(CKPT.relative_to(ROOT)), **res})
    jf.write_text(json.dumps(merged, indent=2))
    old_preds = dict(np.load(nf)) if nf.exists() else {}
    np.savez(nf, **{**old_preds, **preds})
    cf = R / f"{tag}_per_class.csv"
    cols = {}
    if cf.exists():
        rows = list(csv.DictReader(open(cf)))
        cols = {k: [r[k] for r in rows] for k in rows[0] if k != "class"}
    cols.update({k.split("/")[1]: [f"{v:.4f}" for v in per_class[k]] for k in per_class if k.startswith("modelnet40/")})
    if cols:
        with open(cf, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class"] + list(cols))
            for c, name in enumerate(MODELNET40_CLASSES):
                w.writerow([name] + [cols[k][c] for k in cols])
    print("saved ->", jf.relative_to(ROOT))


def plot_confusion(cm, classes, title, out):
    """Row-normalised confusion matrix. Classes without test objects are left out of the rows."""
    support = cm.sum(1)
    rows = np.flatnonzero(support > 0)
    missing = [classes[i].replace("_", " ") for i in np.flatnonzero(support == 0)]
    cmn = cm[rows] / support[rows, None]
    big = len(classes) > 15
    fig, ax = plt.subplots(figsize=(11, 10) if big else (6.5, 5.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="auto" if missing else "equal")
    names = [c.replace("_", " ") for c in classes]
    ax.set_xticks(range(len(classes)), names, rotation=90, fontsize=7 if big else 9)
    ax.set_yticks(range(len(rows)), [names[i] for i in rows], fontsize=7 if big else 9)
    ax.set_xlabel("predicted" + (f"\n(no test objects for: {', '.join(missing)})" if missing else ""))
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="fraction of true class")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
