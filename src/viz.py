"""Rendering of meshes and point clouds through a perspective camera.

All panels in a row use the same camera (looking at the origin, +Z up) and the same
height-based colour map, so meshes, complete clouds and HPR clouds can be compared directly.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


CMAP = matplotlib.colormaps["turbo"]


class Camera:
    """Pinhole camera at `pos` looking at the origin, +Z up."""

    def __init__(self, pos, up=(0, 0, 1)):
        self.pos = np.asarray(pos, dtype=np.float64)
        fwd = -self.pos / np.linalg.norm(self.pos)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        self.R = np.stack([right, np.cross(right, fwd), fwd])  # rows: right, up, forward

    def project(self, p):
        """p [...,3] -> screen xy [...,2] and depth [...]."""
        d = (np.asarray(p) - self.pos) @ self.R.T
        return d[..., :2] / d[..., 2:3], d[..., 2]


def spatial_colors(p, lo, ext):
    """Shared colour map: turbo(0.12 + 0.75 * (0.75 z + 0.25 x)) on bbox-normalised coordinates."""
    q = (p - lo) / np.maximum(ext, 1e-9)
    return CMAP(np.clip(0.12 + 0.75 * (0.75 * q[:, 2] + 0.25 * q[:, 0]), 0, 1))[:, :3]


def draw_points(ax, pts, cam, lo, ext, size=1.5, colors=None):
    xy, z = cam.project(pts)
    order = np.argsort(-z)  # draw far points first
    c = spatial_colors(pts, lo, ext) if colors is None else colors
    ax.scatter(xy[order, 0], xy[order, 1], s=size, c=c[order], linewidths=0, rasterized=True)
    _style(ax)


def _style(ax):
    ax.axis("off")
    fill_box(ax)


def fill_box(ax, xlim=None, ylim=None):
    """Equal axis scales, with the limits widened so the panel fills its grid cell.
    Call after the figure layout is fixed."""
    (x0, x1), (y0, y1) = xlim or ax.get_xlim(), ylim or ax.get_ylim()
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    box = (bb.width * fw) / max(bb.height * fh, 1e-9)
    w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if w / h < box:
        w = h * box
    else:
        h = w / box
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)
    ax.set_aspect("equal", adjustable="box")


def save(fig, path, dpi=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)
