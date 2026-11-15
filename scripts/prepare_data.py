"""Sample point clouds from the ModelNet meshes.

Points are sampled uniformly on the mesh surface and normalised to the unit sphere.
  modelnet40_train_2048.h5   9,843 x 2048
  modelnet40_test_32768.h5   2,468 x 32768
  modelnet10_test_32768.h5     908 x 32768
Files that already exist are skipped unless --force is given.

Usage: python scripts/prepare_data.py [--workers 16] [--force]
"""
import argparse
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pc_utils import MODELNET10_CLASSES, MODELNET40_CLASSES, normalize_unit_sphere, read_off, sample_surface  # noqa: E402


def _job(args):
    path, label, n = args
    rng = np.random.default_rng(zlib.crc32(path.name.encode()))  # deterministic per file
    v, f = read_off(path)
    return normalize_unit_sphere(sample_surface(v, f, n, rng)), label, path.stem


def build(root, classes, split, n, out, workers, force=False):
    if out.exists() and not force:
        print(f"{out.name:32s} exists -> skipped (delete it to rebuild)")
        return
    jobs = [(p, ci, n) for ci, c in enumerate(classes) for p in sorted((root / c / split).glob("*.off"))]
    t = time.time()
    with Pool(workers) as pool:
        res = pool.map(_job, jobs, chunksize=8)
    data = np.stack([r[0] for r in res]).astype(np.float32)
    label = np.array([r[1] for r in res], dtype=np.int64)[:, None]
    names = np.array([r[2] for r in res], dtype="S")
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h:
        h["data"], h["label"], h["name"] = data, label, names
        h.attrs["classes"] = ",".join(classes)
    print(f"{out.name:32s} {data.shape}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="rebuild files that already exist (overwrites them)")
    a = ap.parse_args()
    raw, proc = ROOT / "data/raw", ROOT / "data/processed"
    build(raw / "ModelNet40", MODELNET40_CLASSES, "test", 32768, proc / "modelnet40_test_32768.h5", a.workers, a.force)
    build(raw / "ModelNet10", MODELNET10_CLASSES, "test", 32768, proc / "modelnet10_test_32768.h5", a.workers, a.force)
    build(raw / "ModelNet40", MODELNET40_CLASSES, "train", 2048, proc / "modelnet40_train_2048.h5", a.workers, a.force)
