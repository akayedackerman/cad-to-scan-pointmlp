"""Zero-shot classification with CLIP on the same test sets.

Backbone (ViT-B/32 or ViT-L/14@336px) and background colour are chosen on 400 ModelNet40 training objects.

Usage: python scripts/test_zeroshot_clip.py [--datasets modelnet40 modelnet10] [--sets removal]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import REMOVAL_SETS, load_set, plot_confusion, pretty  # noqa: E402
from metrics import confusion, summarize  # noqa: E402
from pc_utils import MODELNET10_CLASSES, MODELNET40_CLASSES  # noqa: E402
from zeroshot import CLIPZeroShot  # noqa: E402

P = ROOT / "data/processed"
R = ROOT / "outputs/results"
F = ROOT / "outputs/figures"


@torch.no_grad()
def predict(zs, X, bs=16, desc=None):
    if isinstance(X, list):
        return np.concatenate([zs.logits(torch.from_numpy(p)[None]).cpu().numpy() for p in X])
    steps = range(0, len(X), bs)
    if desc:
        steps = tqdm(steps, desc=desc, ncols=100, leave=False)
    return np.concatenate([zs.logits(torch.from_numpy(X[i:i + bs])).cpu().numpy() for i in steps])


def train_subset(per_class=10, n_points=1024, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(P / "modelnet40_train_2048.h5", "r") as h:
        X, y = h["data"][:], h["label"][:, 0]
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(40)])
    pts = np.stack([X[i][rng.choice(X.shape[1], n_points, replace=False)] for i in idx])
    return pts, y[idx]


def show_views(zs, pts, title, out):
    imgs = zs.images(torch.from_numpy(pts)[None]).cpu().numpy()[:, 0]
    fig, axes = plt.subplots(1, len(imgs), figsize=(2 * len(imgs), 2.3))
    for ax, im, t in zip(axes, imgs, [f"view {k + 1}" for k in range(len(imgs) - 1)] + ["top"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def evaluate_sets(tests):
    """Evaluate (dataset, set) pairs with the backbone and background already selected; other results are kept."""
    res = json.loads((R / "clip_results.json").read_text())
    cfg = res["config"]
    print(f"-> using {cfg['clip']}, background={cfg['background']} (selected on training objects)")
    zs = CLIPZeroShot(cfg["clip"], classes=MODELNET40_CLASSES, background=cfg["background"])
    preds = dict(np.load(R / "clip_predictions.npz"))
    for ds, s in tests:
        classes = MODELNET10_CLASSES if ds == "modelnet10" else MODELNET40_CLASSES
        if zs.classes != classes:
            zs.set_classes(classes)
        X, y = load_set(ds, s)
        pr = predict(zs, X, desc=f"{ds} {s}").argmax(1)
        r = summarize(y, pr, len(classes))
        r.pop("per_class")
        res[f"{ds}/{s}"], preds[f"{ds}_{s}"] = r, pr
        print(f"{'ModelNet40' if ds == 'modelnet40' else 'ModelNet10'} {s:22s} OA {r['OA']:.2%}  mAcc {r['mAcc']:.2%}  "
              f"(n={r['n']})", flush=True)
    (R / "clip_results.json").write_text(json.dumps(res, indent=2))
    np.savez(R / "clip_predictions.npz", **preds)
    print("saved -> outputs/results/clip_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["modelnet40", "modelnet10"])
    ap.add_argument("--sets", nargs="+", default=None, choices=["complete", "removal"],
                    help="evaluate only these sets with the selected backbone; other results are kept")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.sets or a.datasets == ["modelnet10"]:
        tests = []
        if a.sets is None or "complete" in a.sets:
            tests += [("modelnet10", "full_1024"), ("modelnet10", "hpr_1024")] if "modelnet10" in a.datasets else []
        if a.sets and "removal" in a.sets:
            tests += [(ds, s) for ds in a.datasets for s in REMOVAL_SETS]
        evaluate_sets(tests)
        sys.exit(0)
    # choose backbone and background on training objects
    Xv, yv = train_subset()
    best, search = None, {}
    for arch in ["ViT-B/32", "ViT-L/14@336px"]:
        zs = CLIPZeroShot(arch, classes=MODELNET40_CLASSES)
        for bg in ["white", "black"]:
            zs.background = bg
            acc = float((predict(zs, Xv).argmax(1) == yv).mean())
            search[f"{arch} | {bg}"] = acc
            print(f"[selection on 400 train objects] {arch:15s} bg={bg:5s}  acc {acc:.2%}")
            if best is None or acc > best[0]:
                best = (acc, arch, bg)
        del zs
        torch.cuda.empty_cache()
    _, arch, bg = best
    print(f"-> using {arch}, background={bg}")

    # test sets
    zs = CLIPZeroShot(arch, classes=MODELNET40_CLASSES, background=bg)
    res = {"config": {"clip": arch, "background": bg, "views": int(zs.mats.shape[0]), "selection_on_train": search}}
    per_class, preds = {}, {}
    for s in ["full_1024", "full_2048", "hpr_1024", "hpr_2048", "hpr_raw"]:
        X, y = load_set("modelnet40", s)
        pr = predict(zs, X).argmax(1)
        r = summarize(y, pr, 40)
        per_class[s] = r.pop("per_class")
        res[f"modelnet40/{s}"], preds[f"modelnet40_{s}"] = r, pr
        print(f"ModelNet40 {s:10s} OA {r['OA']:.2%}  mAcc {r['mAcc']:.2%}")
        if s in ("full_1024", "hpr_1024"):
            plot_confusion(confusion(y, pr, 40), MODELNET40_CLASSES, f"CLIP {arch}, zero-shot\n{pretty('modelnet40', s)}",
                           F / f"confusion_clip_modelnet40_{s}.png")
            i = int(np.flatnonzero(y == MODELNET40_CLASSES.index("chair"))[0])
            show_views(zs, X[i], f"Depth images rendered for CLIP — chair, {pretty('modelnet40', s)}", F / f"clip_views_chair_{s}.png")

    zs.set_classes(MODELNET10_CLASSES)
    for s in ["full_1024", "hpr_1024"]:
        X, y = load_set("modelnet10", s)
        pr = predict(zs, X).argmax(1)
        r = summarize(y, pr, 10)
        r.pop("per_class")
        res[f"modelnet10/{s}"], preds[f"modelnet10_{s}"] = r, pr
        print(f"ModelNet10 {s:10s} OA {r['OA']:.2%}  mAcc {r['mAcc']:.2%}")

    R.mkdir(parents=True, exist_ok=True)
    (R / "clip_results.json").write_text(json.dumps(res, indent=2))
    with open(R / "clip_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"] + list(per_class))
        for c, name in enumerate(MODELNET40_CLASSES):
            w.writerow([name] + [f"{per_class[s][c]:.4f}" for s in per_class])
    np.savez(R / "clip_predictions.npz", **preds)
    print("saved ->", R)
