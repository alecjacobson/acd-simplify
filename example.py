"""CoACD -> a replaceable per-hull simplifier -> Polyscope. See README.md."""

import argparse
import colorsys
from pathlib import Path
from time import perf_counter

import coacd
import numpy as np
import pchs
import polyscope as ps
import trimesh


def simplify_hull(vertices, triangles, target_faces):
    """Replace this function with your simplifier.

    Input: (N, 3) float vertices, (M, 3) zero-based integer triangles.
    Output: vertices and a list of polygon index lists (triangles also work).
    PCHS's target counts supporting planes / polygon faces, NOT triangles
    or vertices. PCHS conservatively enlarges each input convex hull.
    """
    out_v, indices, offsets = pchs.simplify_convex_hull(
        vertices, triangles, target_faces, cost_function=pchs.CostFunction.volume
    )
    polygons = [indices[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    return out_v, polygons


def load_shape(path):
    if path is None:
        # A closed concave surface; no assets, downloads, or boolean tools needed.
        mesh = trimesh.creation.torus(
            major_radius=1.0, minor_radius=0.35,
            major_sections=48, minor_sections=16,
        )
    else:
        mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("Input must contain a nonempty triangle mesh")
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    triangles = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    if not np.isfinite(vertices).all() or np.ptp(vertices, axis=0).max() == 0:
        raise ValueError("Input must have finite vertices and nonzero extent")
    return vertices, triangles


def decompose(vertices, triangles, threshold=0.05, preprocess_mode="auto"):
    """Return CoACD's unsimplified hulls as a list of (vertices, triangles)."""
    return coacd.run_coacd(
        coacd.Mesh(vertices, triangles),
        threshold=threshold,
        preprocess_mode=preprocess_mode,
        apx_mode="ch",       # Convex hull output.
        decimate=False,      # Essential: disable CoACD's native simplification.
        extrude=False,       # Do not offset the output hulls either.
        seed=0,
        # max_ch_vertex is intentionally omitted: it only applies if decimate=True.
    )


def display(vertices, triangles, raw_hulls, simplified_hulls, headless, screenshot):
    ps.set_program_name("CoACD + replaceable hull simplifier")
    ps.set_use_prefs_file(False)
    ps.set_window_size(1800, 700)
    ps.init("openGL3_egl" if headless else "auto")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.set_transparency_mode("pretty")

    # Normalize ONLY display copies. Both algorithms use the original coordinates.
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    scale = np.ptp(vertices, axis=0).max()
    normals = trimesh.Trimesh(vertices, triangles, process=False).vertex_normals
    labels = ["Original", "CoACD (decimate=False)", "CoACD + PCHS"]
    for column, label in enumerate(labels):
        offset = np.array([(column - 1) * 1.3, 0, 0])
        # Slightly inset the overlay's original to avoid coincident-surface flicker.
        original_v = vertices if column == 0 else vertices - 0.002 * scale * normals
        ps.register_surface_mesh(
            f"{label}/original", (original_v - center) / scale + offset, triangles,
            color=(0.55, 0.57, 0.60), smooth_shade=True,
        )
        hulls = [[], raw_hulls, simplified_hulls][column]
        for i, (hull_v, hull_f) in enumerate(hulls):
            color = colorsys.hsv_to_rgb((0.08 + i * 0.618034) % 1, 0.65, 0.9)
            ps.register_surface_mesh(
                f"{label}/hull {i:02d}", (hull_v - center) / scale + offset,
                hull_f, color=color, edge_color=(0.12, 0.14, 0.18),
                edge_width=0.7, transparency=0.45, smooth_shade=False,
            )

    ps.look_at((0, -1.1, 2.0), (0, 0, 0))
    print("View, left to right: original | CoACD over original | PCHS over original")
    if headless:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        ps.screenshot(str(screenshot), transparent_bg=False)
        print(f"Saved {screenshot}")
    else:
        ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", nargs="?", type=Path, help="OBJ/PLY/STL/...; default: torus")
    parser.add_argument("--target-faces", type=int, default=12,
                        help="PCHS supporting-plane budget per hull (default: 12)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="CoACD concavity threshold (default: 0.05)")
    parser.add_argument("--preprocess-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--headless", action="store_true", help="Render via EGL and exit")
    parser.add_argument("--screenshot", type=Path, default=Path("render.png"),
                        help="PNG destination with --headless")
    args = parser.parse_args()
    if args.target_faces < 4:
        parser.error("--target-faces must be at least 4")
    if not 0.01 <= args.threshold <= 1:
        parser.error("--threshold must be between 0.01 and 1")

    vertices, triangles = load_shape(args.mesh)
    coacd.set_log_level("warn")
    start = perf_counter()
    raw_hulls = decompose(vertices, triangles, args.threshold, args.preprocess_mode)
    print(f"CoACD: {len(raw_hulls)} hulls in {perf_counter() - start:.2f}s", flush=True)

    # This is the handoff: no re-hulling, decimation, or display offsets applied.
    simplified_hulls = []
    start = perf_counter()
    for i, (hull_v, hull_f) in enumerate(raw_hulls):
        out_v, out_f = simplify_hull(hull_v, hull_f, args.target_faces)
        simplified_hulls.append((out_v, out_f))
        print(f"  hull {i:02d}: {len(hull_v)} vertices / {len(hull_f)} triangles"
              f" -> {len(out_v)} vertices / {len(out_f)} polygon faces", flush=True)
    print(f"PCHS: {perf_counter() - start:.2f}s", flush=True)
    display(vertices, triangles, raw_hulls, simplified_hulls,
            args.headless, args.screenshot)


if __name__ == "__main__":
    main()
