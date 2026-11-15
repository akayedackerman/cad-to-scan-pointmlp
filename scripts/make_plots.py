"""Result plots (outputs/figures/plots/).

  accuracy_overview, pointmlp_vs_clip, per_class_complete_vs_hpr, robustness,
  tsne_features, predictions_supervised, predictions_zeroshot, lr_schedule,
  training_curves (when outputs/training/<run>/log.csv exists)
  removal_curves (PointMLP and CLIP on the random point removal sets)
  tta_scanobjectnn, tta_gains   (test-time adaptation on ScanObjectNN; also outputs/results/tta_table.md)

Usage: python scripts/make_plots.py [--run pointmlp_modelnet40_300ep]
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
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import CKPT  # noqa: E402
from pc_utils import MODELNET40_CLASSES  # noqa: E402
from pointmlp import load_pointmlp  # noqa: E402
from viz import Camera, draw_points  # noqa: E402

# colours
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SUP, ZS = BLUE, ORANGE
PMLP, CLIPC = BLUE, AQUA

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

P = ROOT / "data/processed"
R = ROOT / "outputs/results"
OUT = ROOT / "outputs/figures/plots"
SETS = [  # (key, label, setting)
    ("modelnet40/full_1024", "MN40\ncomplete\n1024", "supervised"),
    ("modelnet40/full_2048", "MN40\ncomplete\n2048", "supervised"),
    ("modelnet40/hpr_1024", "MN40\nHPR\n1024", "zeroshot"),
    ("modelnet40/hpr_2048", "MN40\nHPR\n2048", "zeroshot"),
    ("modelnet40/hpr_raw", "MN40\nHPR\nall visible", "zeroshot"),
    ("modelnet10/full_1024", "MN10\ncomplete\n1024", "zeroshot"),
    ("modelnet10/hpr_1024", "MN10\nHPR\n1024", "zeroshot"),
]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def bar_labels(ax, bars, vals):
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9, color=INK)


def training_curves(run):
    log = ROOT / "outputs/training" / run / "log.csv"
    if not log.exists():
        return
    rows = list(csv.DictReader(open(log)))
    ep = np.array([int(r["epoch"]) for r in rows])
    g = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.plot(ep, g("train_loss"), color=BLUE, label="train")
    a.plot(ep, g("test_loss"), color=ORANGE, label="test")
    a.set(xlabel="epoch", ylabel="loss (label-smoothed CE)", title="Loss")
    a.legend()
    b.plot(ep, 100 * g("train_acc"), color=BLUE, label="train OA")
    b.plot(ep, 100 * g("test_acc"), color=ORANGE, label="test OA")
    b.plot(ep, 100 * g("test_macc"), color=AQUA, label="test mAcc")
    b.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy", ylim=(0, 100))
    b.legend(loc="lower right")
    fig.suptitle(f"PointMLP training on ModelNet40 — {run} ({len(rows)} epochs)", fontweight="bold")
    fig.tight_layout()
    save(fig, "training_curves.png")


def lr_schedule():
    cfg = yaml.safe_load(open(ROOT / "configs/pointmlp_modelnet40.yaml"))
    n, lr, lo = cfg["epochs"], cfg["learning_rate"], cfg["min_lr"]
    e = np.arange(n)
    lrs = lo + (lr - lo) * (1 + np.cos(np.pi * e / n)) / 2
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(e + 1, lrs, color=BLUE)
    ax.set(xlabel="epoch", ylabel="learning rate", title=f"Cosine LR schedule ({n} epochs)", xlim=(1, n))
    ax.text(n * 0.98, lrs[-1] + 0.004, f"min {lo}", ha="right", color=INK2, fontsize=9)
    ax.text(3, lr, f"start {lr}", va="bottom", color=INK2, fontsize=9)
    save(fig, "lr_schedule.png")


def accuracy_overview(res):
    x = np.arange(len(SETS))
    v = [100 * res[k]["OA"] for k, _, _ in SETS]
    cols = [SUP if s == "supervised" else ZS for _, _, s in SETS]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(x, v, 0.62, color=cols)
    bar_labels(ax, bars, v)
    ax.set_xticks(x, [lab for _, lab, _ in SETS], fontsize=9.5)
    ax.set(ylabel="overall accuracy (%)", ylim=(0, 105),
           title="PointMLP — supervised and zero-shot accuracy")
    ax.grid(axis="x", visible=False)
    ax.axvline(1.5, color=MUTED, lw=1, ls=":")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SUP, label="supervised: complete ModelNet40 objects"),
                       Patch(color=ZS, label="zero-shot: single-view HPR and ModelNet10")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9.5)
    save(fig, "accuracy_overview.png")


def pointmlp_vs_clip(res, clip):
    keys = [s for s in SETS if s[0] in clip]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for off, r, col, name in [(-w / 2, res, PMLP, "PointMLP"),
                              (w / 2, clip, CLIPC, f"CLIP {clip['config']['clip']}")]:
        v = [100 * r[k]["OA"] for k, _, _ in keys]
        bar_labels(ax, ax.bar(x + off, v, w - 0.03, color=col, label=name), v)
    ax.set_xticks(x, [lab for _, lab, _ in keys], fontsize=9.5)
    ax.set(ylabel="overall accuracy (%)", ylim=(0, 105), title="PointMLP and CLIP accuracy")
    ax.grid(axis="x", visible=False)
    ax.axvline(1.5, color=MUTED, lw=1, ls=":")
    ax.text(0.5, 101, "supervised", ha="center", color=INK2, fontsize=9.5)
    ax.text(4, 101, "zero-shot", ha="center", color=INK2, fontsize=9.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9.5)
    save(fig, "pointmlp_vs_clip.png")


def per_class():
    sup = {r["class"]: r for r in csv.DictReader(open(R / "supervised_per_class.csv"))}
    zs = {r["class"]: r for r in csv.DictReader(open(R / "zeroshot_per_class.csv"))}
    names = MODELNET40_CLASSES
    full = np.array([100 * float(sup[c]["full_1024"]) for c in names])
    hpr = np.array([100 * float(zs[c]["hpr_1024"]) for c in names])
    order = np.argsort(full - hpr)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 11))
    ax.hlines(y, full[order], hpr[order], color=GRID, lw=2.5, zorder=1)
    ax.scatter(full[order], y, s=42, color=SUP, edgecolor="white", linewidth=1.5, zorder=2,
               label="supervised: complete (1024)")
    ax.scatter(hpr[order], y, s=42, color=ZS, edgecolor="white", linewidth=1.5, zorder=3,
               label="zero-shot: HPR (1024)")
    ax.set_yticks(y, [names[i].replace("_", " ") for i in order], fontsize=9)
    ax.set(xlabel="per-class accuracy (%)", xlim=(-2, 102),
           title="PointMLP per-class accuracy: complete vs single-view HPR\n(sorted by drop; largest drop at the bottom)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, fontsize=9.5)
    save(fig, "per_class_complete_vs_hpr.png")


def robustness(pred, clip_pred=None):
    with h5py.File(P / "modelnet40_test_hpr_1024.h5", "r") as h:
        y, nvis, el, dense = h["label"][:, 0], h["n_visible"][:], h["elevation"][:], h.attrs["dense_points"]
    frac = 100 * nvis / dense
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.hist(frac, bins=30, color=BLUE, edgecolor="white", linewidth=1)
    ax.set(xlabel="visible points after HPR (% of 32,768)", ylabel="objects", title="How much of each object is visible")
    ax.axvline(np.median(frac), color=INK2, lw=1, ls="--")
    ax.text(np.median(frac) + 1, ax.get_ylim()[1] * 0.92, f"median {np.median(frac):.0f}%", color=INK2, fontsize=9)
    series = [(pred, PMLP, "PointMLP", "o")]
    if clip_pred is not None:
        series.append((clip_pred, CLIPC, "CLIP", "s"))

    def binned(ax, v, n_bins, xlabel, title):
        """Equal-count bins, plotted at the bin median."""
        edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-6
        groups = [(v >= lo) & (v < hi) for lo, hi in zip(edges[:-1], edges[1:])]
        mids = [np.median(v[g]) for g in groups]
        for pr, col, name, m in series:
            acc = [100 * np.mean(pr[g] == y[g]) for g in groups]
            ax.plot(mids, acc, color=col, marker=m, ms=8, markeredgecolor="white", markeredgewidth=1.5, label=name)
        ax.set(xlabel=xlabel, ylabel="accuracy on HPR 1024 (%)", ylim=(0, 100), title=title)
        ax.legend(loc="lower right")

    binned(axes[1], frac, 8, "visible points (%), 8 equal-size bins", "Accuracy vs visibility")
    binned(axes[2], el, 5, "camera elevation (degrees), 5 equal-size bins", "Accuracy vs camera elevation")
    fig.tight_layout()
    save(fig, "robustness.png")


@torch.no_grad()
def tsne(per_class_n=25):
    from sklearn.manifold import TSNE
    model = load_pointmlp(CKPT)
    rng = np.random.default_rng(0)
    with h5py.File(P / "modelnet40_test_full_1024.h5", "r") as h:
        y = h["label"][:, 0]
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), min(per_class_n, (y == c).sum()), replace=False)
                                  for c in range(40)]))
    feats = []
    for kind in ["full", "hpr"]:
        with h5py.File(P / f"modelnet40_test_{kind}_1024.h5", "r") as h:
            X = h["data"][:][idx]
        feats.append(torch.cat([model.forward_features(torch.from_numpy(X[i:i + 64]).cuda().permute(0, 2, 1)).cpu()
                                for i in range(0, len(X), 64)]).numpy())
    F_ = np.concatenate(feats)
    Y, K = np.concatenate([y[idx], y[idx]]), np.repeat([0, 1], len(idx))
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
        F_ / np.linalg.norm(F_, axis=1, keepdims=True))
    highlight = ["airplane", "chair", "car", "guitar", "person", "lamp", "bottle", "bed"]
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for ax, k, title in [(axes[0], 0, "Supervised: complete objects"), (axes[1], 1, "Zero-shot: single-view HPR")]:
        m = K == k
        ax.scatter(Z[m, 0], Z[m, 1], s=9, color="#cfcec9", linewidths=0, label="other 32 classes")
        for c, col in zip(highlight, cols):
            s = m & (Y == MODELNET40_CLASSES.index(c))
            ax.scatter(Z[s, 0], Z[s, 1], s=26, color=col, edgecolor="white", linewidth=0.8, label=c)
        ax.set_title(f"{title} ({m.sum()} objects)")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=1.3)
    fig.suptitle("t-SNE of PointMLP global features (one shared embedding; 25 objects per class)", fontweight="bold")
    fig.tight_layout()
    save(fig, "tsne_features.png")


def predictions(pred, set_name, title, out, n=24, seed=1):
    with h5py.File(P / f"modelnet40_test_{set_name}.h5", "r") as h:
        X, y = h["data"][:], h["label"][:, 0]
        center, scale = h["center"][:], h["scale"][:]
    with h5py.File(P / "modelnet40_test_hpr_1024.h5", "r") as h:
        cam = h["camera"][:]
    idx = np.sort(np.random.default_rng(seed).choice(len(y), n, replace=False))
    fig, axes = plt.subplots(4, 6, figsize=(15, 10.5))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.08, hspace=0.32)
    for ax, i in zip(axes.flat, idx):
        p = X[i] * scale[i] + center[i]
        draw_points(ax, p, Camera(cam[i]), p.min(0), np.ptp(p, 0), size=2.5)
        ok = pred[i] == y[i]
        t, pr = MODELNET40_CLASSES[y[i]].replace("_", " "), MODELNET40_CLASSES[pred[i]].replace("_", " ")
        ax.set_title(f"{'✓' if ok else '✗'} true: {t}\npred: {pr}", fontsize=10, color=INK if ok else RED,
                     fontweight="normal")
    fig.suptitle(f"{title} — {np.sum(pred[idx] == y[idx])}/{n} correct on this random sample "
                 f"(whole test set: {100 * np.mean(pred == y):.1f}%)", fontweight="bold", y=0.985)
    save(fig, out)


def removal_curves(res, clip):
    ratios = [0, 10, 20, 30, 40, 50]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, ds in zip(axes, ["modelnet40", "modelnet10"]):
        for model, r_, col in [("PointMLP", res, PMLP), (f"CLIP {clip['config']['clip']}", clip, CLIPC)]:
            for removal, ls, mk in [("uniform", "-", "o"), ("contiguous", "--", "s")]:
                keys = [f"{ds}/full_1024"] + [f"{ds}/removal_{removal}_{r}" for r in ratios[1:]]
                if not all(k in r_ for k in keys):
                    continue
                ax.plot(ratios, [100 * r_[k]["OA"] for k in keys], color=col, ls=ls, marker=mk, ms=6,
                        markeredgecolor="white", markeredgewidth=1.2, label=f"{model}, {removal}")
        ax.set_title("ModelNet40" if ds == "modelnet40" else "ModelNet10")
        ax.set_xticks(ratios, [f"{r}%" for r in ratios])
        ax.set_xlabel("points removed")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("overall accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Random point removal: uniform and contiguous", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "removal_curves.png")


TTA_METHODS = ["source_only", "bn_adapt", "tent", "shot_im", "t3a", "bn_t3a", "sar"]
TTA_LABEL = {"source_only": "Source only", "bn_adapt": "BN-adapt", "tent": "TENT", "shot_im": "SHOT-IM",
             "t3a": "T3A", "bn_t3a": "BN+T3A", "sar": "SAR"}
TTA_COLOR = {"source_only": "#5f5e5a", "bn_adapt": BLUE, "tent": ORANGE, "shot_im": AQUA, "t3a": YELLOW,
             "bn_t3a": MAGENTA, "sar": GREEN}
SONN = {"obj_only": "OBJ_ONLY", "obj_bg": "OBJ_BG", "pb_t50_rs": "PB_T50_RS"}
def tta_scanobjectnn(t):
    x = np.arange(len(SONN))
    w = 0.12
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for k, m in enumerate(TTA_METHODS):
        v = [100 * t[s][m]["OA"] for s in SONN]
        e = [100 * t[s][m]["OA_std"] for s in SONN]
        bars = ax.bar(x + (k - 3) * w, v, w - 0.012, yerr=e, color=TTA_COLOR[m], label=TTA_LABEL[m],
                      error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        for b_, val in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, val + 1.6, f"{val:.1f}", ha="center", fontsize=7, color=INK,
                    rotation=90)
    ax.set_xticks(x, list(SONN.values()))
    ax.set(ylabel="overall accuracy (%)", ylim=(0, 72),
           title="ModelNet40 → ScanObjectNN (11 shared classes): test-time adaptation")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=7, fontsize=9.5)
    save(fig, "tta_scanobjectnn.png")


def tta_gains(t):
    methods = TTA_METHODS[1:]
    G = np.array([[100 * t[s][m]["gain"] for m in methods] for s in SONN])
    sig = np.array([[t[s][m]["gain_lo"] > 0 or t[s][m]["gain_hi"] < 0 for m in methods] for s in SONN])
    lim = np.abs(G).max()
    fig, ax = plt.subplots(figsize=(9, 2.9))
    im = ax.imshow(G, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            ax.text(j, i, f"{G[i, j]:+.1f}{'*' if sig[i, j] else ''}", ha="center", va="center", fontsize=10,
                    color="white" if abs(G[i, j]) > 0.6 * lim else INK)
    ax.set_xticks(range(len(methods)), [TTA_LABEL[m] for m in methods])
    ax.set_yticks(range(len(SONN)), list(SONN.values()))
    ax.xaxis.tick_top()
    ax.grid(False)
    fig.colorbar(im, fraction=0.04, pad=0.02, label="gain (points)")
    ax.set_title("ScanObjectNN: accuracy gain over source only (* = 95% CI excludes 0)", pad=28)
    save(fig, "tta_gains.png")


def tta_table(t):
    lines = ["| ScanObjectNN | " + " | ".join(TTA_LABEL[m] for m in TTA_METHODS) + " |",
             "|---|" + "---|" * len(TTA_METHODS)]
    for s, label in SONN.items():
        cells = []
        for m in TTA_METHODS:
            e = t[s][m]
            c = f"{100 * e['OA']:.1f} ± {100 * e['OA_std']:.1f}"
            if m != "source_only":
                c += f" ({100 * e['gain']:+.1f}{'*' if e['gain_lo'] > 0 or e['gain_hi'] < 0 else ''})"
            cells.append(c)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    text = ("Overall accuracy (%), mean ± std over 3 seeds. In brackets: gain over source only; "
            "* = 95% paired bootstrap CI excludes 0.\n\n" + "\n".join(lines) + "\n")
    (R / "tta_table.md").write_text(text)
    print("saved", R / "tta_table.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pointmlp_modelnet40_300ep", help="training run folder in outputs/training")
    a = ap.parse_args()
    sup = json.loads((R / "supervised_results.json").read_text())
    zs = json.loads((R / "zeroshot_results.json").read_text())
    res = {**sup, **zs}
    sp, zp = np.load(R / "supervised_predictions.npz"), np.load(R / "zeroshot_predictions.npz")
    clip_file = R / "clip_results.json"
    clip = json.loads(clip_file.read_text()) if clip_file.exists() else None
    cp = np.load(R / "clip_predictions.npz") if clip else None

    training_curves(a.run)
    lr_schedule()
    accuracy_overview(res)
    if clip:
        pointmlp_vs_clip(res, clip)
    per_class()
    robustness(zp["modelnet40_hpr_1024"], cp["modelnet40_hpr_1024"] if clip else None)
    tsne()
    predictions(sp["modelnet40_full_1024"], "full_1024", "Supervised: PointMLP on complete objects",
                "predictions_supervised.png")
    predictions(zp["modelnet40_hpr_1024"], "hpr_1024", "Zero-shot: PointMLP on single-view HPR",
                "predictions_zeroshot.png")
    if clip:
        removal_curves(res, clip)
    tta_file = R / "tta_summary.json"
    if tta_file.exists():
        t = json.loads(tta_file.read_text())
        tta_scanobjectnn(t)
        tta_gains(t)
        tta_table(t)
