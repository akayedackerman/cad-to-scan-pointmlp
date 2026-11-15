"""Random point removal test sets from the complete 1024-point ModelNet test clouds.

  uniform     points chosen at random over the whole object are removed (no point twice);
              the object stays complete but becomes evenly sparser
  contiguous  a random point is chosen and its nearest neighbours are removed until the target
              number is reached; this leaves one ball-shaped hole

Removal ratios 10-50% (at least 512 points remain, the first PointMLP sampling stage).
The remaining points keep the coordinate frame of the complete cloud.

Output: data/processed/<dataset>_test_removal_<method>_<ratio>.h5   data [N, 1024 - removed, 3]

Settings: configs/data.yaml (removal)
Usage:    python scripts/make_removal_sets.py [--ratios 10 20 30 40 50] [--seed 0]
"""
import argparse
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
from pc_utils import remove_points  # noqa: E402

P = ROOT / "data/processed"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--source")
    ap.add_argument("--methods", nargs="+", choices=["uniform", "contiguous"])
    ap.add_argument("--ratios", nargs="+", type=int)
    ap.add_argument("--seed", type=int)
    a = config.load(ap, "configs/data.yaml", "removal")
    for ds in a.datasets:
        with h5py.File(P / f"{ds}_test_{a.source}.h5", "r") as h:
            X, label, name, classes = h["data"][:], h["label"][:], h["name"][:], h.attrs["classes"]
        for method in a.methods:
            for r in a.ratios:
                n_remove = round(X.shape[1] * r / 100)
                out = np.stack([remove_points(x, n_remove, method,
                                              np.random.default_rng([a.seed, r, zlib.crc32(n)]))
                                for x, n in zip(X, name)])
                path = P / f"{ds}_test_removal_{method}_{r}.h5"
                with h5py.File(path, "w") as h:
                    h["data"], h["label"], h["name"] = out, label, name
                    h.attrs.update(classes=classes, method=method, ratio=r, seed=a.seed)
                print(f"{path.name:44s} {out.shape}")
