"""Zero-shot evaluation of PointMLP (trained on complete ModelNet40 objects).

  modelnet40: single-view HPR, 1024 / 2048 / all visible points
  modelnet10: complete and single-view HPR, 1024 points (ModelNet10 classes of the 40-class output)
  modelnet40, modelnet10: random point removal, uniform and contiguous, 10-50%

Settings: configs/test.yaml (checkpoint, zeroshot); removal sets from configs/data.yaml (removal)
Usage:    python scripts/test_zeroshot.py [--datasets modelnet40 modelnet10] [--sets hpr removal]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
from evaluate import REMOVAL_SETS, run, save  # noqa: E402
from pointmlp import load_pointmlp  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--datasets", nargs="+", help="evaluate only these datasets; results of the others are kept")
    ap.add_argument("--sets", nargs="+", choices=["hpr", "removal"])
    ap.add_argument("--batch_size", type=int)
    c = config.load(ap, "configs/test.yaml", "zeroshot")
    model = load_pointmlp(ROOT / c.checkpoint)
    tests = []
    for ds in c.datasets:
        if "hpr" in c.sets:
            tests += [(ds, s) for s in c.hpr_tests.get(ds, [])]
        if "removal" in c.sets:
            tests += [(ds, s) for s in REMOVAL_SETS]
    save(*run(model, tests, "zeroshot", c.batch_size), "zeroshot", ROOT / c.checkpoint)
