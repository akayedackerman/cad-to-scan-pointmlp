"""Point-cloud utilities: mesh I/O, surface sampling, normalisation, Hidden Point Removal."""
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]
MODELNET10_CLASSES = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]

SCANOBJECTNN_CLASSES = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table",
                        "bed", "pillow", "sink", "sofa", "toilet"]
# ScanObjectNN classes shared with ModelNet40 -> ModelNet40 classes
SCANOBJECTNN_SHARED = {"bed": ["bed"], "cabinet": ["dresser", "night_stand", "wardrobe"], "chair": ["chair"],
                       "desk": ["desk"], "display": ["monitor"], "door": ["door"], "shelf": ["bookshelf"],
                       "sink": ["sink"], "sofa": ["sofa"], "table": ["table"], "toilet": ["toilet"]}


# --------------------------------------------------------------------------- #
# Mesh I/O
# --------------------------------------------------------------------------- #
def read_off(path):
    """Read an OFF mesh. Returns vertices [V,3] and triangle faces [F,3].

    Also reads the malformed header found in some ModelNet40 files ('OFF1568 1820 0').
    Polygons with more than three vertices are split into triangles.
    """
    tokens = Path(path).read_text().split()
    if tokens[0] == "OFF":
        i = 1
    elif tokens[0].startswith("OFF"):  # header and counts on one line
        tokens[0] = tokens[0][3:]
        i = 0
    else:
        raise ValueError(f"{path}: not an OFF file")
    nv, nf = int(tokens[i]), int(tokens[i + 1])
    i += 3
    verts = np.asarray(tokens[i:i + 3 * nv], dtype=np.float64).reshape(nv, 3)
    i += 3 * nv
    rest = tokens[i:]
    # fast path: all triangles
    if len(rest) >= 4 * nf and all(rest[k] == "3" for k in range(0, 4 * nf, 4)):
        faces = np.asarray(rest[:4 * nf], dtype=np.int64).reshape(nf, 4)[:, 1:]
        return verts, faces
    faces, k = [], 0
    for _ in range(nf):
        n = int(rest[k])
        idx = [int(t) for t in rest[k + 1:k + 1 + n]]
        k += 1 + n
        for j in range(1, n - 1):
            faces.append((idx[0], idx[j], idx[j + 1]))
    return verts, np.asarray(faces, dtype=np.int64)


def sample_surface(verts, faces, n, rng):
    """Uniform (area-weighted) random sampling of n points on a triangle mesh."""
    tri = verts[faces]  # [F,3,3]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        raise ValueError("degenerate mesh (zero area)")
    fid = rng.choice(len(faces), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[fid]
    return (t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])).astype(np.float32)


def normalize_unit_sphere(pts):
    """Centre on the centroid and scale into the unit sphere (as in modelnet40_ply_hdf5_2048)."""
    pts = pts - pts.mean(axis=0, keepdims=True)
    return (pts / np.max(np.linalg.norm(pts, axis=1))).astype(np.float32)


# ModelNet meshes are Z-up and are used as they are. The checkpoint scores about 88%
# in this frame and about 15% after a conversion to Y-up.
UP_AXIS = 2


# --------------------------------------------------------------------------- #
# Hidden Point Removal  (Katz, Tal & Basri, "Direct Visibility of Point Sets", SIGGRAPH 2007)
# Defaults: camera distance 4, radius = max|p - cam| * 10**gamma with gamma = 3.
# --------------------------------------------------------------------------- #
def hidden_point_removal(pts, camera, gamma=3.0):
    """Indices of the points visible from `camera`.

    1. Move the camera to the origin.
    2. Spherical flip: p' = p + 2(R - |p|) p/|p|, with R = max|p| * 10**gamma.
    3. A point is visible iff its flipped copy is a vertex of the convex hull of
       {flipped points} together with the camera (the origin).
    Equivalent to open3d's PointCloud.hidden_point_removal(camera, R).
    """
    p = pts - camera[None, :]
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    R = norm.max() * 10 ** gamma
    flipped = p + 2 * (R - norm) * p / np.maximum(norm, 1e-12)
    hull = ConvexHull(np.vstack([flipped, np.zeros((1, 3))]))
    vis = hull.vertices[hull.vertices < len(pts)]
    return np.sort(vis)


def camera_position(azimuth_deg, elevation_deg, distance=4.0, up_axis=UP_AXIS):
    """Camera on a sphere around the origin; elevation measured from the horizontal plane (+Z up)."""
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    h = [a for a in range(3) if a != up_axis]
    cam = np.zeros(3)
    cam[h[0]] = distance * np.cos(el) * np.cos(az)
    cam[h[1]] = distance * np.cos(el) * np.sin(az)
    cam[up_axis] = distance * np.sin(el)
    return cam


def three_quarter_camera(rng, distance=4.0, elev_deg=(15.0, 30.0), jitter_deg=15.0):
    """Elevated three-quarter view: azimuth in {45,135,225,315} +/- jitter, elevation 15-30 deg.

    The camera sees the front or back, one side and the top of the object.
    Returns (camera_xyz, azimuth_deg, elevation_deg).
    """
    az = float(rng.choice([45.0, 135.0, 225.0, 315.0]) + rng.uniform(-jitter_deg, jitter_deg))
    el = float(rng.uniform(*elev_deg))
    return camera_position(az, el, distance), az, el


def remove_points(pts, n_remove, method, rng):
    """Remove n_remove points: 'uniform' (random points) or 'contiguous' (nearest neighbours of a random point)."""
    keep = np.ones(len(pts), dtype=bool)
    if method == "uniform":
        keep[rng.choice(len(pts), n_remove, replace=False)] = False
    elif method == "contiguous":
        d = np.sum((pts - pts[rng.integers(len(pts))]) ** 2, axis=1)
        keep[np.argpartition(d, n_remove - 1)[:n_remove]] = False
    else:
        raise ValueError(method)
    return pts[keep]
