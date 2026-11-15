"""Supervised evaluation: PointMLP on the complete ModelNet40 test objects (1024 and 2048 points).

Usage: python scripts/test_supervised.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import CKPT, run, save  # noqa: E402
from pointmlp import load_pointmlp  # noqa: E402

if __name__ == "__main__":
    model = load_pointmlp(CKPT)
    print("checkpoint:", CKPT.relative_to(ROOT))
    save(*run(model, [("modelnet40", "full_1024"), ("modelnet40", "full_2048")], "supervised"), "supervised")
