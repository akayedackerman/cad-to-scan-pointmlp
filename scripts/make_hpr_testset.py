"""Build the single-view test sets with Hidden Point Removal (HPR).

Camera: distance 4, azimuth 45/135/225/315 +- 15 deg, elevation 15-30 deg, +Z up.
HPR radius: max|p - cam| * 10**gamma, gamma = 3.

Output (data/processed/<dataset>_test_*.h5):
  hpr_raw            all visible points (points + offsets)
  hpr_1024/2048      visible points, FPS, normalised to the unit sphere
  full_1024/2048     complete objects, FPS, normalised to the unit sphere
Fixed-size files also store camera, azimuth, elevation, n_visible, center and scale.

Usage: python scripts/make_hpr_testset.py [--gamma 3] [--seed 0] [--datasets modelnet40 modelnet10]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pc_utils import hidden_point_removal, three_quarter_camera  # noqa: E402
from pointmlp import furthest_point_sample  # noqa: E402

SIZES = (1024, 2048)


def _hpr_job(args):
    i, pts, seed, gamma = args
    cam, az, el = three_quarter_camera(np.random.default_rng([seed, i]))
    return hidden_point_removal(pts, cam, gamma=gamma), cam, az, el


def batched_fps(clouds, k, device="cuda", bs=64):
    """Batched FPS for clouds of different sizes. Shorter clouds are padded with their first point."""
    out = []
    for s in range(0, len(clouds), bs):
        chunk = clouds[s:s + bs]
        m = max(len(c) for c in chunk)
        pad = np.stack([np.concatenate([c, np.repeat(c[:1], m - len(c), 0)]) for c in chunk])
        x = torch.from_numpy(pad).to(device)
        idx = furthest_point_sample(x, k)
        out.append(torch.gather(x, 1, idx[..., None].expand(-1, -1, 3)).cpu().numpy())
    return np.concatenate(out)


def unit_sphere(batch):
    """Normalise each cloud to the unit sphere. Returns (points, center, scale)."""
    center = batch.mean(1, keepdims=True)
    scale = np.linalg.norm(batch - center, axis=2).max(1)
    return ((batch - center) / scale[:, None, None]).astype(np.float32), center[:, 0].astype(np.float32), scale.astype(np.float32)


def build(ds, seed, gamma, workers):
    P = ROOT / "data/processed"
    with h5py.File(P / f"{ds}_test_32768.h5", "r") as h:
        dense, label, name, classes = h["data"][:], h["label"][:], h["name"][:], h.attrs["classes"]
    with Pool(workers) as pool:
        res = pool.map(_hpr_job, [(i, dense[i], seed, gamma) for i in range(len(dense))], chunksize=8)
    vis = [dense[i][r[0]] for i, r in enumerate(res)]
    cam = np.stack([r[1] for r in res]).astype(np.float32)
    az = np.array([r[2] for r in res], np.float32)
    el = np.array([r[3] for r in res], np.float32)
    nvis = np.array([len(v) for v in vis])
    meta = dict(label=label, name=name, camera=cam, azimuth=az, elevation=el, n_visible=nvis)
    attrs = dict(classes=classes, hpr_gamma=gamma, camera_distance=4.0, seed=seed,
                 dense_points=dense.shape[1], downsampling="fps(start=0) then unit-sphere normalisation")

    def write(fname, **arrays):
        with h5py.File(P / fname, "w") as h:
            for k, v in {**meta, **arrays}.items():
                h[k] = v
            h.attrs.update(attrs)

    offsets = np.concatenate([[0], np.cumsum(nvis)])
    write(f"{ds}_test_hpr_raw.h5", points=np.concatenate(vis).astype(np.float32), offsets=offsets)
    for k in SIZES:
        for kind, clouds in [("hpr", vis), ("full", list(dense))]:
            data, center, scale = unit_sphere(batched_fps(clouds, k))
            write(f"{ds}_test_{kind}_{k}.h5", data=data, center=center, scale=scale)
    frac = nvis / dense.shape[1]
    print(f"{ds}: {len(vis)} objects | visible {nvis.mean():.0f} / {dense.shape[1]} points on average "
          f"({frac.mean():.1%}; min {nvis.min()}, max {nvis.max()}) | "
          f"objects with < 2048 visible points: {(nvis < 2048).sum()}, < 1024: {(nvis < 1024).sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=3.0, help="HPR radius exponent (same default as modelnet_hpr_downsample.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", nargs="+", default=["modelnet40", "modelnet10"])
    a = ap.parse_args()
    for ds in a.datasets:
        build(ds, a.seed, a.gamma, a.workers)
