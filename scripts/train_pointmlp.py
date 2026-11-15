"""Train PointMLP on ModelNet40.

Settings are read from configs/train.yaml; any key can be overridden on the
command line (e.g. --epochs 50 --batch_size 16 --run_name test).
Output in outputs/training/<run_name>/: config.yaml, log.csv, curves.png, best.pth, last.pth.

Usage:
  python scripts/train_pointmlp.py [--config configs/train.yaml] [--resume]
  bash train.sh
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pointmlp as models  # noqa: E402
from metrics import summarize  # noqa: E402


class H5Points(Dataset):
    """Point clouds from an HDF5 file (keys data/label), as in the official ModelNet40 dataset class."""

    def __init__(self, path, cfg, train=True):
        with h5py.File(path, "r") as h:
            self.X, self.y = h["data"][:], h["label"][:, 0]
        self.cfg, self.train = cfg, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        c = self.cfg
        p = self.X[i]
        if self.train:
            p = p[np.random.choice(len(p), c["num_points"], replace=False)]
            p = p * np.random.uniform(c["scale_low"], c["scale_high"], size=3) \
                + np.random.uniform(-c["shift_range"], c["shift_range"], size=3)
            np.random.shuffle(p)
        else:
            p = p[: c["num_points"]]
        return p.astype(np.float32), self.y[i]


def smooth_ce(logits, y, eps):
    """Label-smoothing cross-entropy (official `cal_loss`)."""
    n = logits.size(1)
    one_hot = F.one_hot(y, n).float()
    target = one_hot * (1 - eps) + (1 - one_hot) * eps / (n - 1)
    return -(target * F.log_softmax(logits, 1)).sum(1).mean()


@torch.no_grad()
def evaluate(model, loader, n_classes, eps):
    model.eval()
    ys, ps, loss = [], [], 0.0
    for x, y in loader:
        logits = model(x.cuda().permute(0, 2, 1))
        loss += smooth_ce(logits, y.cuda(), eps).item() * len(y)
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    r = summarize(np.concatenate(ys), np.concatenate(ps), n_classes)
    return loss / len(loader.dataset), r["OA"], r["mAcc"]


def plot_curves(rows, out, title):
    ep = [r["epoch"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    a.plot(ep, [r["train_loss"] for r in rows], color="#2a78d6", label="train")
    a.plot(ep, [r["test_loss"] for r in rows], color="#eb6834", label="test")
    a.set(xlabel="epoch", ylabel="loss (label-smoothed CE)", title="Loss")
    b.plot(ep, [100 * r["train_acc"] for r in rows], color="#2a78d6", label="train OA")
    b.plot(ep, [100 * r["test_acc"] for r in rows], color="#eb6834", label="test OA")
    b.plot(ep, [100 * r["test_macc"] for r in rows], color="#1baf7a", label="test mAcc")
    b.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy", ylim=(0, 100))
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def load_config():
    ap = argparse.ArgumentParser(description="Train PointMLP on ModelNet40")
    ap.add_argument("--config", default=str(ROOT / "configs/train.yaml"))
    ap.add_argument("--resume", action="store_true", help="continue from outputs/training/<run_name>/last.pth")
    known, rest = ap.parse_known_args()
    cfg = yaml.safe_load(open(known.config))
    # command-line overrides: --key value
    for k, v in zip(rest[::2], rest[1::2]):
        key = k.lstrip("-")
        if key not in cfg:
            ap.error(f"unknown config key: {key}")
        cfg[key] = yaml.safe_load(v)
    return cfg, known


if __name__ == "__main__":
    cfg, args = load_config()
    out = ROOT / "outputs/training" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)
    print(yaml.safe_dump(cfg, sort_keys=False))

    if cfg["seed"] is None:
        torch.backends.cudnn.benchmark = True
    else:
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.backends.cudnn.deterministic = True

    train_set = H5Points(ROOT / cfg["train_file"], cfg, train=True)
    test_set = H5Points(ROOT / cfg["test_file"], cfg, train=False)
    tr = DataLoader(train_set, cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["workers"],
                    persistent_workers=cfg["workers"] > 0)
    te = DataLoader(test_set, cfg["batch_size"], num_workers=cfg["workers"])
    print(f"train: {len(train_set)} objects | test: {len(test_set)} objects | {len(tr)} iterations / epoch")

    model = getattr(models, cfg["model"])(num_classes=cfg["num_classes"]).cuda()
    print(f"{cfg['model']}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    opt = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"], eta_min=cfg["min_lr"])
    start, best, rows = 0, 0.0, []
    if args.resume:
        ck = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, rows = ck["epoch"], ck["best"], ck["rows"]
        print(f"resumed after epoch {start} (best test OA {best:.2%})")

    eps = cfg["label_smoothing"]
    for ep in range(start, cfg["epochs"]):
        model.train()
        t0, tot, seen, ys, ps = time.time(), 0.0, 0, [], []
        bar = tqdm(tr, desc=f"epoch {ep + 1}/{cfg['epochs']}", leave=False, dynamic_ncols=True)
        for x, y in bar:
            x, y = x.cuda(non_blocking=True).permute(0, 2, 1), y.cuda(non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = smooth_ce(logits, y, eps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tot += loss.item() * len(y)
            seen += len(y)
            ys.append(y.cpu().numpy())
            ps.append(logits.argmax(1).cpu().numpy())
            bar.set_postfix(loss=f"{tot / seen:.3f}", acc=f"{np.mean(np.concatenate(ps) == np.concatenate(ys)):.1%}")
        lr = opt.param_groups[0]["lr"]
        sched.step()
        tr_r = summarize(np.concatenate(ys), np.concatenate(ps), cfg["num_classes"])
        if (ep + 1) % cfg["eval_every"] == 0 or ep + 1 == cfg["epochs"]:
            tl, oa, macc = evaluate(model, te, cfg["num_classes"], eps)
        else:
            tl, oa, macc = float("nan"), float("nan"), float("nan")
        rows.append(dict(epoch=ep + 1, lr=lr, train_loss=tot / seen, train_acc=tr_r["OA"], train_macc=tr_r["mAcc"],
                         test_loss=tl, test_acc=oa, test_macc=macc, seconds=round(time.time() - t0, 1)))
        if oa > best:
            best = oa
            torch.save(model.state_dict(), out / "best.pth")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=ep + 1, best=best, rows=rows), out / "last.pth")
        with open(out / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        plot_curves(rows, out / "curves.png", f"PointMLP on ModelNet40, {cfg['run_name']}")
        r = rows[-1]
        print(f"epoch {ep + 1:3d}/{cfg['epochs']} | lr {lr:.4f} | train loss {r['train_loss']:.3f} "
              f"OA {r['train_acc']:.2%} | test loss {tl:.3f} OA {oa:.2%} mAcc {macc:.2%} | best {best:.2%} | "
              f"{r['seconds']:.0f}s", flush=True)
    print(f"done -> {out}  (best test OA {best:.2%})")
