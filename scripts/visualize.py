"""Figures of the test data.

  hpr_<dataset>_<a|b>_<spatial|gray>.png   mesh, full point cloud, HPR, HPR FPS 2048, HPR FPS 1024
  hpr_car_armadillo_bunny_piano_<style>.png  the same for car, armadillo, bunny and piano
  removal_<dataset>_<style>.png             complete cloud, uniform and contiguous removal (10/30/50%)
  scanobjectnn_<style>.png                  ScanObjectNN examples: OBJ_ONLY, OBJ_BG, PB_T50_RS
  gallery_<dataset>_<complete|hpr>_<style>.png   one test object per class
  interactive/<object>.html                 3-D views (plotly)

Usage: python scripts/visualize.py
"""
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pc_utils import MODELNET10_CLASSES, MODELNET40_CLASSES  # noqa: E402
from viz import spatial_colors  # noqa: E402
import hpr_figure  # noqa: E402

P = ROOT / "data/processed"
OUT = ROOT / "outputs/figures"
RAW = ROOT / "data/raw"

FIGURES = {
    ("modelnet40", "a"): ["airplane_0627", "car_0198", "chair_0890", "piano_0232", "person_0089"],
    ("modelnet40", "b"): ["guitar_0156", "lamp_0125", "stool_0091", "bottle_0336", "bench_0174"],
    ("modelnet10", "a"): ["bathtub_0107", "bed_0516", "chair_0891", "desk_0201", "dresser_0201"],
    ("modelnet10", "b"): ["monitor_0466", "night_stand_0201", "sofa_0683", "table_0393", "toilet_0346"],
}
# car and piano from the ModelNet40 test set, armadillo and bunny from the Stanford 3D scanning repository
SHOWCASE = [("ModelNet40/car/test/car_0213.off", "Car", (-35.0, 12.0)),
            ("stanford/armadillo.off", "Armadillo", (115.0, 20.0)),
            ("stanford/bunny.off", "Bunny", (-120.0, 20.0)),
            ("ModelNet40/piano/test/piano_0232.off", "Piano", (40.0, 20.0))]
REMOVAL = {"modelnet40": ["airplane_0627", "chair_0890", "piano_0232", "person_0089"],
           "modelnet10": ["bed_0516", "desk_0201", "sofa_0683", "toilet_0346"]}
SCAN_EXAMPLES = ["chair", "table", "sofa", "display", "toilet"]
INTERACTIVE = FIGURES[("modelnet40", "a")] + ["lamp_0125", "stool_0091", "monitor_0466"]


def load(ds="modelnet40"):
    d = {}
    with h5py.File(P / f"{ds}_test_32768.h5", "r") as h:
        d["label"] = h["label"][:, 0]
        d["name"] = [n.decode() for n in h["name"][:]]
        d["dense"] = h["data"][:]
    with h5py.File(P / f"{ds}_test_hpr_raw.h5", "r") as h:
        d["raw"], d["off"], d["camera"] = h["points"][:], h["offsets"][:], h["camera"][:]
        d["az"], d["el"], d["nvis"] = h["azimuth"][:], h["elevation"][:], h["n_visible"][:]
    for kind in ["hpr", "full"]:
        for k in [1024, 2048]:
            with h5py.File(P / f"{ds}_test_{kind}_{k}.h5", "r") as h:
                # object frame, to use the HPR camera
                d[f"{kind}{k}"] = h["data"][:] * h["scale"][:][:, None, None] + h["center"][:][:, None, :]
    d["index"] = {n: i for i, n in enumerate(d["name"])}
    return d


def hpr_figures():
    for (ds, part), names in FIGURES.items():
        folder = RAW / ("ModelNet40" if ds == "modelnet40" else "ModelNet10")
        objects, labels = [], []
        for name in names:
            cls = name.rsplit("_", 1)[0]
            objects.append(hpr_figure.prepare(folder / cls / "test" / f"{name}.off"))
            labels.append(cls.replace("_", " ").title())
        for style in ["spatial", "gray"]:
            hpr_figure.figure(objects, labels, OUT / f"hpr_{ds}_{part}_{style}.png", gray=style == "gray")
    if not all((RAW / path).exists() for path, _, _ in SHOWCASE):
        print("data/raw/stanford/{armadillo,bunny}.off not found: car / armadillo / bunny / piano figure skipped")
        return
    objects = [hpr_figure.prepare(RAW / path, view=view) for path, _, view in SHOWCASE]
    labels = [label for _, label, _ in SHOWCASE]
    for style in ["spatial", "gray"]:
        hpr_figure.figure(objects, labels, OUT / f"hpr_car_armadillo_bunny_piano_{style}.png", gray=style == "gray")


def removal_figures():
    ratios = [10, 30, 50]
    columns = ["Complete (1024)"] + [f"Uniform {r}%" for r in ratios] + [f"Contiguous {r}%" for r in ratios]
    for ds, names in REMOVAL.items():
        sets = {}
        for key in ["full_1024"] + [f"removal_{m}_{r}" for m in ["uniform", "contiguous"] for r in ratios]:
            with h5py.File(P / f"{ds}_test_{key}.h5", "r") as h:
                index = {n.decode(): i for i, n in enumerate(h["name"][:])}
                data = h["data"][:]
                if "scale" in h:
                    data = data * h["scale"][:][:, None, None] + h["center"][:][:, None, :]
                sets[key] = (index, data)
        folder = RAW / ("ModelNet40" if ds == "modelnet40" else "ModelNet10")
        rows = []
        for name in names:
            cls = name.rsplit("_", 1)[0]
            o = hpr_figure.prepare(folder / cls / "test" / f"{name}.off")
            clouds = [sets[k][1][sets[k][0][name]] for k in sets]
            rows.append((cls.replace("_", " ").title(), o, clouds))
        for style in ["spatial", "gray"]:
            hpr_figure.point_grid(rows, columns, OUT / f"removal_{ds}_{style}.png", gray=style == "gray")


def scanobjectnn_figure():
    if not (P / "scanobjectnn_obj_only.h5").exists():
        print("ScanObjectNN not prepared: ScanObjectNN figure skipped")
        return
    from pc_utils import SCANOBJECTNN_SHARED
    classes = list(SCANOBJECTNN_SHARED)
    variants = {"obj_only": "OBJ_ONLY", "obj_bg": "OBJ_BG", "pb_t50_rs": "PB_T50_RS"}
    data = {}
    for v in variants:
        with h5py.File(P / f"scanobjectnn_{v}.h5", "r") as h:
            data[v] = (h["data"][:], h["label"][:])
    clouds, labels = [], []
    for c in SCAN_EXAMPLES:
        k = classes.index(c)
        i = int(np.flatnonzero(data["obj_only"][1] == k)[0])
        j = int(np.flatnonzero(data["pb_t50_rs"][1] == k)[0])
        clouds.append([data["obj_only"][0][i], data["obj_bg"][0][i], data["pb_t50_rs"][0][j]])
        labels.append(c.title())
    for style in ["spatial", "gray"]:
        hpr_figure.scan_grid(clouds, labels, list(variants.values()), OUT / f"scanobjectnn_{style}.png",
                             gray=style == "gray")


def galleries():
    """One test object per class, complete and single-view HPR (1024 points), colour and greyscale."""
    for ds, (folder, classes, ncols) in {"modelnet40": ("ModelNet40", MODELNET40_CLASSES, 8),
                                          "modelnet10": ("ModelNet10", MODELNET10_CLASSES, 5)}.items():
        objects = []
        for cls in classes:
            path = sorted((RAW / folder / cls / "test").glob("*.off"))[0]
            objects.append(hpr_figure.prepare(path))
        titles = [c.replace("_", " ") for c in classes]
        for kind, label in [("complete", "complete point clouds"), ("hpr", "single-view HPR")]:
            for style in ["spatial", "gray"]:
                hpr_figure.gallery(objects, titles, OUT / f"gallery_{ds}_{kind}_{style}.png", kind=kind,
                                   gray=style == "gray", ncols=ncols,
                                   title=f"{folder} test set, {label} (1024 points)")


def interactive(d, names, outdir):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    outdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        i = d["index"][name]
        dense = d["dense"][i]
        lo, ext = dense.min(0), np.ptp(dense, 0)
        fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                            subplot_titles=["Full cloud (1024)", "HPR, all visible points", "HPR, FPS 1024"])
        vis = d["raw"][d["off"][i]:d["off"][i + 1]]
        for c, (p, s) in enumerate([(d["full1024"][i], 2.5), (vis, 1.2), (d["hpr1024"][i], 2.5)], start=1):
            col = ["rgb(%d,%d,%d)" % tuple((255 * x).astype(int)) for x in spatial_colors(p, lo, ext)]
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                       marker=dict(size=s, color=col), showlegend=False), row=1, col=c)
        cam = d["camera"][i] / 4.0 * 1.6
        scene = dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                     camera=dict(eye=dict(x=cam[0], y=cam[1], z=cam[2]), up=dict(x=0, y=0, z=1)))
        fig.update_layout(title=f"{name} — drag to rotate", height=520, margin=dict(l=0, r=0, t=60, b=0),
                          scene=scene, scene2=scene, scene3=scene)
        fig.write_html(outdir / f"{name}.html", include_plotlyjs="cdn")
    print("saved", len(names), "interactive views ->", outdir)


if __name__ == "__main__":
    hpr_figures()
    removal_figures()
    scanobjectnn_figure()
    galleries()
    d = load()
    interactive(d, INTERACTIVE, OUT / "interactive")
