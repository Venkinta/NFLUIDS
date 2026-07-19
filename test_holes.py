"""End-to-end test for multi-loop (holes) meshing.

Builds an outer rectangle (domain) with an inlet on the left, outlet on the
right, and walls top/bottom, plus an inner rectangle (a "wing" hole) tagged
Wall. Runs the full mesher pipeline and validates the resulting mesh dict.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from cfdeditor.point import Point
from cfdeditor.line import Line
from cfdeditor.mesher import Mesher
from cfdeditor.solver import Solver


def rect_lines(x0, y0, x1, y1, bc_map):
    """Build 4 Lines forming a closed rectangle, with per-edge BC types.

    bc_map: dict mapping edge index (0=left,1=top,2=right,3=bottom) -> type.
    """
    bl = Point(x0, y0)
    br = Point(x1, y0)
    tr = Point(x1, y1)
    tl = Point(x0, y1)
    edges = [(bl, br), (br, tr), (tr, tl), (tl, bl)]
    lines = []
    for i, (a, b) in enumerate(edges):
        ln = Line(a, b)
        ln.boundary_type = bc_map[i]
        lines.append(ln)
    return lines


def main():
    # Outer domain 0..200 x 0..100 (world units, mm scale)
    outer = rect_lines(0, 0, 200, 100, {
        0: "Velocity Inlet",   # left
        1: "Wall",             # top
        2: "Pressure Outlet",  # right
        3: "Wall",             # bottom
    })
    # Inner hole (wing) 80..120 x 40..60
    hole = rect_lines(80, 40, 120, 60, {
        0: "Wall", 1: "Wall", 2: "Wall", 3: "Wall"
    })

    lines = outer + hole

    mesher = Mesher(
        lines=lines,
        n_layers=3, growth_factor=1.2, thickness=1.0,
        spacing=5.0, r=4.0, unit_to_meters=0.001,
    )

    print("=== MESH ===")
    mesher.mesh()

    print("\n=== PIPELINE ===")
    data = mesher.solver_data_pipeline()

    Nc = data['Nc']
    Nf = data['Nf']
    tags = data['boundary_tags']
    n_wall = int(np.sum(tags == 0))
    n_inlet = int(np.sum(tags == 1))
    n_outlet = int(np.sum(tags == 2))
    n_internal = int(np.sum(tags == -1))

    print(f"Cells: {Nc}, Faces: {Nf}")
    print(f"  wall={n_wall} inlet={n_inlet} outlet={n_outlet} internal={n_internal}")

    # --- Validation 1: hole interior must contain NO cells ---
    hole_poly = np.array([[80, 40], [120, 40], [120, 60], [80, 60]], float)
    from matplotlib.path import Path
    hole_path = Path(hole_poly)
    cc = data['cell_centers']
    inside_hole = hole_path.contains_points(cc)
    n_inside_hole = int(np.sum(inside_hole))
    print(f"Cells whose centroid falls inside the hole: {n_inside_hole}")

    # --- Validation 2: the hole contributes its own wall boundary faces ---
    # Outer-domain walls sit on x=0, x=200, y=0, y=100.  Any wall face whose
    # midpoint is NOT on one of those four outer edges must belong to the hole.
    Cf = data['Cf']
    on_outer = ((np.abs(Cf[:, 0]) < 1e-6) | (np.abs(Cf[:, 0] - 0.2) < 1e-6) |
                (np.abs(Cf[:, 1]) < 1e-6) | (np.abs(Cf[:, 1] - 0.1) < 1e-6))
    hole_wall_faces = int(np.sum((tags == 0) & ~on_outer))
    print(f"Wall boundary faces belonging to the hole: {hole_wall_faces}")

    # --- Validation 3: connectivity sanity (no orphan cells) ---
    owner = data['owner']
    neighbor = data['neighbor']
    cells_in_faces = set(owner) | set(neighbor[neighbor != -1])
    orphans = len(set(range(Nc)) - cells_in_faces)

    # --- Validation 4: run a few solver iterations to ensure topology is valid ---
    solver = Solver(data, [1.0, 0.0], 0.0, 1.2, 0.002)
    solver.Solve(max_iterations=20, tolerance=1e-3)

    ok = (Nc > 0 and n_inside_hole == 0 and hole_wall_faces > 0
          and orphans == 0 and np.all(np.isfinite(solver.P)))
    print("\n=== RESULT ===")
    print(f"  no cells inside hole : {n_inside_hole == 0}")
    print(f"  hole wall faces > 0  : {hole_wall_faces > 0}")
    print(f"  no orphan cells      : {orphans == 0}")
    print(f"  solver finite        : {np.all(np.isfinite(solver.P))}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()