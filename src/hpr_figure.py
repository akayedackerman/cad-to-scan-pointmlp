"""Mesh / point cloud / HPR figures with one perspective camera per object.

Rendering follows pointMLP-pytorch/figures/hpr_armadillo_bunny/generate_figure.py:
camera at distance 4 looking at the origin (+Z up), HPR radius max|p - c| * 10**3,
turbo colours on height with surface-normal shading (or neutral grey with the same shading).
HPR is computed here from the display camera of each object; the test sets are not changed.
"""
import zlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from pc_utils import hidden_point_removal, read_off
from pointmlp import furthest_point_sample

DENSE = 65536
DIST = 4.0
COLUMNS = ["(a) Surface mesh", "(b) Full point cloud", "(c) HPR", "(d) Downsampled HPR (2048)",
           "(e) Downsampled HPR (1024)"]


def sample_with_faces(verts, faces, n, rng):
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    fid = rng.choice(len(faces), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[fid]
    pts = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return pts, fid


def face_normals(verts, faces):
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def camera_basis(az_deg, el_deg):
    az, el = np.radians([az_deg, el_deg])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.cross(direction, right)
    return np.stack([right, up, direction], axis=1), direction


def project(p, basis):
    q = p @ basis
    q[:, :2] *= DIST / (DIST - q[:, 2:3])
    return q


def auto_view(points, elevation=20.0):
    """Azimuth with the largest projected footprint among views oblique to the object's main axis."""
    xy = points[:, :2] - points[:, :2].mean(0)
    w, v = np.linalg.eigh(np.cov(xy.T))
    axis = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
    best, best_area = 0.0, -1.0
    for az in np.arange(0, 360, 5.0):
        rel = (az - axis) % 90
        if not 25 <= rel <= 65:
            continue
        basis, _ = camera_basis(az, elevation)
        q = project(points[::16].copy(), basis)
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if area > best_area:
            best, best_area = az, area
    return float(best)


def fps_indices(points, k):
    x = torch.from_numpy(points.astype(np.float32))[None].cuda()
    return furthest_point_sample(x, k)[0].cpu().numpy()


def prepare(off_path, view=None, elevation=20.0):
    verts, faces = read_off(off_path)
    used, faces = np.unique(faces, return_inverse=True)   # drop vertices that no face uses
    verts, faces = verts[used], faces.reshape(-1, 3)
    rng = np.random.default_rng(zlib.crc32(off_path.name.encode()))
    pts, fid = sample_with_faces(verts, faces, DENSE, rng)
    center = pts.mean(0)
    scale = np.linalg.norm(pts - center, axis=1).max()
    verts = (verts - center) / scale
    pts = ((pts - center) / scale).astype(np.float64)
    normals = face_normals(verts, faces)
    az = auto_view(pts, elevation) if view is None else view[0]
    el = elevation if view is None else view[1]
    basis, direction = camera_basis(az, el)
    vis = hidden_point_removal(pts, DIST * direction, gamma=3.0)
    i2048 = vis[fps_indices(pts[vis], 2048)]
    i1024 = vis[fps_indices(pts[vis], 1024)]
    c1024 = fps_indices(pts, 1024)
    return dict(verts=verts, faces=faces, normals=normals, points=pts, point_normals=normals[fid],
                visible=vis, i2048=i2048, i1024=i1024, c1024=c1024, basis=basis, direction=direction, az=az, el=el)


def colours(p, n, o, gray):
    lo, ext = o["verts"].min(0), np.ptp(o["verts"], 0)
    xyz = (p - lo) / np.maximum(ext, 1e-12)
    c = np.full((len(p), 3), 0.64) if gray else \
        matplotlib.colormaps["turbo"](0.12 + 0.75 * (0.75 * xyz[:, 2] + 0.25 * xyz[:, 0]))[:, :3]
    light = o["direction"] + np.array([-0.2, -0.1, 0.6])
    light /= np.linalg.norm(light)
    shade = 0.52 + 0.48 * np.abs(n @ light)
    return np.clip(c * shade[:, None], 0, 1)


def finish(fig, out):
    """Save the figure to out, or show it in a window when out is None (returns when the window is closed)."""
    if out is None:
        plt.show()
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=250, facecolor="white")
        print("saved", out)
    plt.close(fig)


def figure(objects, labels, out, gray=False, title=None):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    rows = len(objects)
    fig = plt.figure(figsize=(12.6, 2.05 * rows + 0.75), facecolor="white")
    top = 1 - 0.42 / fig.get_figheight()
    bottom = 0.38 / fig.get_figheight()
    gs = fig.add_gridspec(rows, 5, left=0.045, right=0.997, bottom=bottom, top=top, wspace=0.03, hspace=0.05)
    for r, (o, label) in enumerate(zip(objects, labels)):
        basis = o["basis"]
        mesh_q = project(o["verts"].copy(), basis)
        pts_q = project(o["points"].copy(), basis)
        pcol = colours(o["points"], o["point_normals"], o, gray)
        lo, hi = mesh_q[:, :2].min(0), mesh_q[:, :2].max(0)
        mid, span = (lo + hi) / 2, (hi - lo) * 1.08
        for c in range(5):
            ax = fig.add_subplot(gs[r, c])
            ax.set_axis_off()
            if c == 0:
                tri = mesh_q[o["faces"]]
                centers = o["verts"][o["faces"]].mean(1)
                fc = colours(centers, o["normals"], o, gray)
                order = np.argsort(tri[:, :, 2].mean(1))
                ax.add_collection(PolyCollection(tri[order, :, :2], facecolors=fc[order], edgecolors="none"))
            else:
                idx = [np.arange(len(pts_q)), o["visible"], o["i2048"], o["i1024"]][c - 1]
                size = [0.30, 0.42, 1.0, 1.6][c - 1]
                q = pts_q[idx]
                order = np.argsort(q[:, 2])
                ax.scatter(q[order, 0], q[order, 1], s=size, c=pcol[idx][order], linewidths=0, rasterized=True)
            box = ax.get_position()
            cell = (box.width * fig.get_figwidth()) / (box.height * fig.get_figheight())
            w, h = span
            if w / h < cell:
                w = h * cell
            else:
                h = w / cell
            ax.set_xlim(mid[0] - w / 2, mid[0] + w / 2)
            ax.set_ylim(mid[1] - h / 2, mid[1] + h / 2)
            ax.set_aspect("equal")
        box = gs[r, 0].get_position(fig)
        fig.text(0.016, (box.y0 + box.y1) / 2, label, rotation=90, ha="center", va="center",
                 fontweight="bold", fontsize=11)
    for c, t in enumerate(COLUMNS):
        box = gs[0, c].get_position(fig)
        fig.text((box.x0 + box.x1) / 2, 1 - 0.2 / fig.get_figheight(), t, ha="center", va="center", fontsize=10.5)
    handle = Line2D([], [], marker="o", color="#525252" if gray else "#269D81", linestyle="none", markersize=5,
                    label="Spatial points")
    fig.legend(handles=[handle], loc="lower center", bbox_to_anchor=(0.52, 0.0), frameon=False, fontsize=9.5)
    if title:
        fig.suptitle(title)
    finish(fig, out)


def point_grid(rows, columns, out, gray=False, sizes=None):
    """Grid of point clouds. rows: list of (label, object from prepare(), [points per column])."""
    from scipy.spatial import cKDTree
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    n_rows, n_cols = len(rows), len(columns)
    fig = plt.figure(figsize=(2.3 * n_cols + 0.6, 2.05 * n_rows + 0.75), facecolor="white")
    top = 1 - 0.42 / fig.get_figheight()
    bottom = 0.38 / fig.get_figheight()
    gs = fig.add_gridspec(n_rows, n_cols, left=0.6 / fig.get_figwidth(), right=0.997, bottom=bottom, top=top,
                          wspace=0.03, hspace=0.05)
    for r, (label, o, clouds) in enumerate(rows):
        basis = o["basis"]
        mesh_q = project(o["verts"].copy(), basis)
        lo, hi = mesh_q[:, :2].min(0), mesh_q[:, :2].max(0)
        mid, span = (lo + hi) / 2, (hi - lo) * 1.08
        tree = cKDTree(o["points"])
        for c, pts in enumerate(clouds):
            ax = fig.add_subplot(gs[r, c])
            ax.set_axis_off()
            col = colours(pts, o["point_normals"][tree.query(pts)[1]], o, gray)
            q = project(pts.astype(np.float64).copy(), basis)
            order = np.argsort(q[:, 2])
            ax.scatter(q[order, 0], q[order, 1], s=(sizes or [1.6] * n_cols)[c], c=col[order], linewidths=0,
                       rasterized=True)
            box = ax.get_position()
            cell = (box.width * fig.get_figwidth()) / (box.height * fig.get_figheight())
            w, h = span
            if w / h < cell:
                w = h * cell
            else:
                h = w / cell
            ax.set_xlim(mid[0] - w / 2, mid[0] + w / 2)
            ax.set_ylim(mid[1] - h / 2, mid[1] + h / 2)
            ax.set_aspect("equal")
        box = gs[r, 0].get_position(fig)
        fig.text(0.25 / fig.get_figwidth(), (box.y0 + box.y1) / 2, label, rotation=90, ha="center", va="center",
                 fontweight="bold", fontsize=11)
    for c, t in enumerate(columns):
        box = gs[0, c].get_position(fig)
        fig.text((box.x0 + box.x1) / 2, 1 - 0.2 / fig.get_figheight(), t, ha="center", va="center", fontsize=10.5)
    finish(fig, out)


def scan_grid(clouds, labels, columns, out, gray=False, elevation=20.0):
    """Grid of scanned objects (no mesh): rows = objects, columns = variants. clouds[r][c] is [N,3], Z-up."""
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    n_rows, n_cols = len(clouds), len(columns)
    fig = plt.figure(figsize=(2.6 * n_cols + 0.6, 2.2 * n_rows + 0.6), facecolor="white")
    gs = fig.add_gridspec(n_rows, n_cols, left=0.6 / fig.get_figwidth(), right=0.997,
                          bottom=0.1 / fig.get_figheight(), top=1 - 0.42 / fig.get_figheight(), wspace=0.03, hspace=0.05)
    for r, row in enumerate(clouds):
        az = auto_view(np.asarray(row[0], dtype=np.float64), elevation)
        basis, _ = camera_basis(az, elevation)
        for c, pts in enumerate(row):
            pts = np.asarray(pts, dtype=np.float64)
            ax = fig.add_subplot(gs[r, c])
            ax.set_axis_off()
            lo, ext = pts.min(0), np.maximum(np.ptp(pts, 0), 1e-9)
            z = (pts - lo) / ext
            col = np.full((len(pts), 3), 0.45) if gray else \
                matplotlib.colormaps["turbo"](0.12 + 0.75 * (0.75 * z[:, 2] + 0.25 * z[:, 0]))[:, :3]
            q = project(pts.copy(), basis)
            order = np.argsort(q[:, 2])
            ax.scatter(q[order, 0], q[order, 1], s=1.2, c=col[order], linewidths=0, rasterized=True)
            ax.set_aspect("equal")
        box = gs[r, 0].get_position(fig)
        fig.text(0.25 / fig.get_figwidth(), (box.y0 + box.y1) / 2, labels[r], rotation=90, ha="center",
                 va="center", fontweight="bold", fontsize=11)
    for c, t in enumerate(columns):
        box = gs[0, c].get_position(fig)
        fig.text((box.x0 + box.x1) / 2, 1 - 0.2 / fig.get_figheight(), t, ha="center", va="center", fontsize=10.5)
    finish(fig, out)


def gallery(objects, titles, out, kind="complete", gray=False, ncols=8, title=None):
    """One panel per object, each with its own camera. kind: 'complete' (FPS 1024) or 'hpr' (HPR, FPS 1024)."""
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    nrows = int(np.ceil(len(objects) / ncols))
    fig = plt.figure(figsize=(2.0 * ncols, 2.15 * nrows + (0.5 if title else 0.1)), facecolor="white")
    top = 1 - (0.75 if title else 0.35) / fig.get_figheight()
    gs = fig.add_gridspec(nrows, ncols, left=0.005, right=0.995, bottom=0.01, top=top, wspace=0.04, hspace=0.18)
    for k, (o, t) in enumerate(zip(objects, titles)):
        idx = o["i1024"] if kind == "hpr" else o["c1024"]
        pts = o["points"][idx]
        col = colours(pts, o["point_normals"][idx], o, gray)
        q = project(pts.copy(), o["basis"])
        mesh_q = project(o["verts"].copy(), o["basis"])
        lo, hi = mesh_q[:, :2].min(0), mesh_q[:, :2].max(0)
        mid, span = (lo + hi) / 2, (hi - lo) * 1.08
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        ax.set_axis_off()
        order = np.argsort(q[:, 2])
        ax.scatter(q[order, 0], q[order, 1], s=1.4, c=col[order], linewidths=0, rasterized=True)
        box = ax.get_position()
        cell = (box.width * fig.get_figwidth()) / (box.height * fig.get_figheight())
        w, h = span
        if w / h < cell:
            w = h * cell
        else:
            h = w / cell
        ax.set_xlim(mid[0] - w / 2, mid[0] + w / 2)
        ax.set_ylim(mid[1] - h / 2, mid[1] + h / 2)
        ax.set_aspect("equal")
        ax.set_title(t, fontsize=10, pad=2)
    if title:
        fig.suptitle(title, fontsize=13, y=1 - 0.25 / fig.get_figheight())
    finish(fig, out)
