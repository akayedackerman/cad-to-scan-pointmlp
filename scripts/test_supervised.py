"""Supervised evaluation: PointMLP on the complete ModelNet40 test objects (1024 and 2048 points).

Settings: configs/test.yaml (checkpoint, supervised)
Usage:    python scripts/test_supervised.py [--checkpoint path/to.pth]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
from evaluate import run, save  # noqa: E402
from pointmlp import load_pointmlp  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--batch_size", type=int)
    c = config.load(ap, "configs/test.yaml", "supervised")
    model = load_pointmlp(ROOT / c.checkpoint)
    tests = [(ds, s) for ds, sets in c.tests.items() for s in sets]
    save(*run(model, tests, "supervised", c.batch_size), "supervised", ROOT / c.checkpoint)
