"""Zero-shot evaluation of PointMLP (trained on complete ModelNet40 objects).

  modelnet40: single-view HPR, 1024 / 2048 / all visible points
  modelnet10: complete and single-view HPR, 1024 points (ModelNet10 classes of the 40-class output)
  modelnet40, modelnet10: random point removal, uniform and contiguous, 10-50%

Usage: python scripts/test_zeroshot.py [--datasets modelnet40 modelnet10] [--sets hpr removal]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import CKPT, REMOVAL_SETS, run, save  # noqa: E402
from pointmlp import load_pointmlp  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["modelnet40", "modelnet10"],
                    help="evaluate only these datasets; results of the others are kept")
    ap.add_argument("--sets", nargs="+", default=["hpr", "removal"], choices=["hpr", "removal"])
    a = ap.parse_args()
    model = load_pointmlp(CKPT)
    print("checkpoint:", CKPT.relative_to(ROOT))
    tests = [("modelnet40", "hpr_1024"), ("modelnet40", "hpr_2048"), ("modelnet40", "hpr_raw"),
             ("modelnet10", "full_1024"), ("modelnet10", "hpr_1024")]
    tests = [t for t in tests if "hpr" in a.sets]
    if "removal" in a.sets:
        tests += [(ds, s) for ds in ["modelnet40", "modelnet10"] for s in REMOVAL_SETS]
    tests = [t for t in tests if t[0] in a.datasets]
    save(*run(model, tests, "zeroshot"), "zeroshot")
