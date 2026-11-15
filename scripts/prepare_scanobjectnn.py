"""ScanObjectNN test sets with the classes shared with ModelNet40.

Input:  data/raw/ScanObjectNN/h5_files/{main_split_nobg,main_split}/*.h5   (official release)
Output: data/processed/scanobjectnn_{obj_only,obj_bg,pb_t50_rs}.h5   data [N,2048,3], label [N]

Points are rotated from Y-up (ScanObjectNN) to Z-up (ModelNet): (x, y, z) -> (x, -z, y).
Normalisation and the 2048 -> 1024 subsampling are done at test time.

Settings: configs/data.yaml (scanobjectnn)
Usage:    python scripts/prepare_scanobjectnn.py
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
from pc_utils import SCANOBJECTNN_CLASSES, SCANOBJECTNN_SHARED  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir")
    c = config.load(ap, "configs/data.yaml", "scanobjectnn")
    shared = list(SCANOBJECTNN_SHARED)
    for name, rel in c.variants.items():
        with h5py.File(ROOT / c.raw_dir / rel, "r") as h:
            X, y = h["data"][:].astype(np.float32), h["label"][:].astype(int).ravel()
        keep = np.isin(y, [SCANOBJECTNN_CLASSES.index(c) for c in shared])
        X = X[keep]
        X = np.stack([X[..., 0], -X[..., 2], X[..., 1]], -1)
        labels = np.array([shared.index(SCANOBJECTNN_CLASSES[c]) for c in y[keep]], dtype=np.int64)
        out = ROOT / f"data/processed/scanobjectnn_{name}.h5"
        with h5py.File(out, "w") as h:
            h["data"], h["label"] = X, labels
            h.attrs["classes"] = ",".join(shared)
        print(f"{out.name:28s} {X.shape}  classes {np.bincount(labels).tolist()}")
