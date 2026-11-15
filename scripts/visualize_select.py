"""Interactive figure builder: choose the dataset, objects, figure type, style and camera.

Figure types
  hpr       mesh | full point cloud | HPR | HPR FPS 2048 | HPR FPS 1024
  removal   complete (1024) | uniform removal | contiguous removal, at the chosen ratios

Every question can also be answered with a flag, e.g.
  python scripts/visualize_select.py --dataset modelnet40 --objects chair airplane_0627 \\
      --figure hpr --style both --views auto 40,20 --name my_figure

Usage: python scripts/visualize_select.py
Output: outputs/figures/custom/<name>_<style>.png
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import hpr_figure  # noqa: E402
from pc_utils import MODELNET10_CLASSES, MODELNET40_CLASSES, remove_points  # noqa: E402

RAW = ROOT / "data/raw"
OUT = ROOT / "outputs/figures/custom"
DATASETS = {"modelnet40": ("ModelNet40", MODELNET40_CLASSES), "modelnet10": ("ModelNet10", MODELNET10_CLASSES),
            "stanford": ("stanford", ["armadillo", "bunny"])}


def ask(question, options=None, default=None, multiple=False):
    """Prompt until a valid answer is given. options: list of strings shown as a numbered menu."""
    if options:
        for i, o in enumerate(options, 1):
            print(f"  [{i:>2}] {o}")
    hint = f" [{default}]" if default is not None else ""
    while True:
        answer = input(f"{question}{hint}: ").strip() or (str(default) if default is not None else "")
        if not options:
            if answer:
                return answer
            continue
        picks = []
        for token in answer.replace(",", " ").split():
            if token.isdigit() and 1 <= int(token) <= len(options):
                picks.append(options[int(token) - 1])
            elif token in options:
                picks.append(token)
            else:
                picks = None
                break
        if picks and (multiple or len(picks) == 1):
            return picks if multiple else picks[0]
        print("  please choose from the list (number or name)" + (", several allowed" if multiple else ""))


def mesh_files(dataset, cls):
    folder, _ = DATASETS[dataset]
    if dataset == "stanford":
        return [RAW / folder / f"{cls}.off"]
    return sorted((RAW / folder / cls / "test").glob("*.off"))


def resolve(dataset, spec, interactive):
    """'chair' -> first test object (or ask), 'chair_0891' -> that file, 'chair:3' -> 4th test object."""
    _, classes = DATASETS[dataset]
    name, _, idx = spec.partition(":")
    if name in classes:
        files = mesh_files(dataset, name)
        if idx:
            return files[int(idx)]
        if interactive and len(files) > 1:
            print(f"  {name}: {len(files)} test objects ({files[0].stem} ... {files[-1].stem})")
            choice = ask(f"  object for {name} (index 0-{len(files) - 1} or file name)", default=0)
            if choice.isdigit():
                return files[int(choice)]
            return next(f for f in files if f.stem == choice)
        return files[0]
    cls = name.rsplit("_", 1)[0]
    path = RAW / DATASETS[dataset][0] / cls / "test" / f"{name}.off"
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    return path


def parse_view(v):
    if v in (None, "", "auto"):
        return None
    az, _, el = v.partition(",")
    return float(az), float(el or 20)


def removal_clouds(o, ratios, seed=0):
    """Complete 1024-point cloud (FPS of the dense cloud) and its uniform / contiguous removals."""
    complete = o["points"][hpr_figure.fps_indices(o["points"], 1024)]
    rng = lambda r: np.random.default_rng([seed, r, zlib.crc32(o["name"].encode())])  # noqa: E731
    clouds = [complete]
    for method in ["uniform", "contiguous"]:
        clouds += [remove_points(complete, round(1024 * r / 100), method, rng(r)) for r in ratios]
    return clouds


def main():
    ap = argparse.ArgumentParser(description="Build a figure by choosing the objects and options.")
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--objects", nargs="+", help="class names (first test object), class:index, or file names")
    ap.add_argument("--figure", choices=["hpr", "removal"])
    ap.add_argument("--ratios", nargs="+", type=int, help="removal ratios in %% (removal figure)")
    ap.add_argument("--style", choices=["spatial", "gray", "both"])
    ap.add_argument("--views", nargs="+", help="per object: auto or azimuth,elevation (e.g. 40,20)")
    ap.add_argument("--name", help="output file name (without extension)")
    a = ap.parse_args()

    dataset = a.dataset or ask("Dataset", list(DATASETS), default=1)
    classes = DATASETS[dataset][1]
    objects = a.objects or ask("Objects (numbers or names, e.g. 1 9 31)", classes, multiple=True)
    paths = [resolve(dataset, o, not a.objects) for o in objects]
    figure = a.figure or ask("Figure", ["hpr", "removal"], default=1)
    ratios = a.ratios or ([int(r) for r in ask("Removal ratios in %", default="10 30 50").split()]
                          if figure == "removal" else [])
    style = a.style or ask("Style", ["spatial", "gray", "both"], default=3)
    if a.views:
        views = (a.views * len(paths))[:len(paths)]
    else:
        views = [ask(f"Camera for {p.stem} (auto or azimuth,elevation)", default="auto") for p in paths]
    name = a.name or ask("File name", default=f"{figure}_{dataset}_{'_'.join(p.stem for p in paths)}")

    objs, labels = [], []
    for p, v in zip(paths, views):
        o = hpr_figure.prepare(p, view=parse_view(v))
        o["name"] = p.stem
        objs.append(o)
        labels.append(p.stem.rsplit("_", 1)[0].replace("_", " ").title() if dataset != "stanford" else p.stem.title())
        print(f"  {p.stem}: azimuth {o['az']:.0f}, elevation {o['el']:.0f}, {len(o['visible'])} visible points")

    styles = ["spatial", "gray"] if style == "both" else [style]
    for st in styles:
        out = OUT / f"{name}_{st}.png"
        if figure == "hpr":
            hpr_figure.figure(objs, labels, out, gray=st == "gray")
        else:
            columns = ["Complete (1024)"] + [f"Uniform {r}%" for r in ratios] + [f"Contiguous {r}%" for r in ratios]
            rows = [(lab, o, removal_clouds(o, ratios)) for lab, o in zip(labels, objs)]
            hpr_figure.point_grid(rows, columns, out, gray=st == "gray")


if __name__ == "__main__":
    main()
