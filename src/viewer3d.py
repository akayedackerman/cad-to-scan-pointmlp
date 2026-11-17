"""Live, rotatable 3-D view of the clouds of a figure (Open3D window).

The clouds of one object are placed side by side, one object per row, in the same order as the columns of the
saved figure, coloured the same way and seen from the same camera (each object is turned into its own camera
frame). Mouse: left drag rotates, wheel zooms, right/middle drag pans.
Keys: + / - change the point size, R resets the view, H prints all controls, Q or Esc closes the window.
"""
import numpy as np
import open3d as o3d

from hpr_figure import colours

GAP = 2.4   # distance between neighbouring clouds (objects are normalised to the unit sphere)


def show(rows, columns, gray=False, title="Point clouds", point_size=3.0):
    """rows: list of (label, object from hpr_figure.prepare(), [cloud per column]).
    A cloud is an (N, 3) array, or the string "mesh" for the object's surface mesh."""
    from scipy.spatial import cKDTree
    geoms = []
    for r, (label, o, clouds) in enumerate(rows):
        basis = o["basis"]          # columns: camera right, up, viewing direction (world -> camera: p @ basis)
        tree = cKDTree(o["points"])
        for c, pts in enumerate(clouds):
            shift = np.array([GAP * c, -GAP * r, 0.0])
            if isinstance(pts, str):
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(o["verts"]),
                                                 o3d.utility.Vector3iVector(o["faces"]))
                mesh.compute_vertex_normals()
                vn = np.asarray(mesh.vertex_normals)
                mesh.vertex_colors = o3d.utility.Vector3dVector(colours(o["verts"], vn, o, gray))
                mesh.vertices = o3d.utility.Vector3dVector(o["verts"] @ basis + shift)
                mesh.compute_vertex_normals()
                geoms.append(mesh)
                continue
            pts = np.asarray(pts, dtype=np.float64)
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts @ basis + shift))
            pc.colors = o3d.utility.Vector3dVector(colours(pts, o["point_normals"][tree.query(pts)[1]], o, gray))
            geoms.append(pc)
        print(f"  row {r + 1}: {label}", flush=True)
    print("  columns, left to right: " + " | ".join(columns))
    print("  mouse: drag to rotate, wheel to zoom, right drag to pan; keys: +/- point size, R reset, Q close",
          flush=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1600, height=max(520, 360 * len(rows)))
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.ones(3)
    opt.mesh_show_back_face = True
    centre = np.array([GAP * (max(len(c) for _, _, c in rows) - 1) / 2, -GAP * (len(rows) - 1) / 2, 0.0])
    ctl = vis.get_view_control()
    ctl.set_front([0.0, 0.0, 1.0])     # camera frame: the object's figure camera looks along -z
    ctl.set_up([0.0, 1.0, 0.0])
    ctl.set_lookat(centre)
    ctl.set_zoom(0.36)
    vis.run()
    vis.destroy_window()
