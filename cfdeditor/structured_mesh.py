"""Structured all-quad mesh of an axis-aligned rectangle.

A second, deliberately tiny mesh generator (NOT a general quad mesher): it
returns the same dict as Mesher.solver_data_pipeline(), so Solver, export_vtu,
meshIO and api.solve_mesh_data() consume it unchanged. Its purpose is
validation: benchmark cases such as the lid-driven cavity are defined on
square domains, and an orthogonal quad grid removes every mesher-induced
error source (triangle/quad seam, non-orthogonality, size jumps) so what is
left is the solver's own discretisation error.

Conventions (identical to the triangle Mesher):
  * edge index 0=bottom, 1=right, 2=top, 3=left  (same as rect_lines)
  * BC tags: -1 internal, 0 Wall, 1 Velocity Inlet, 2 Pressure Outlet,
    3 Symmetry, 4 Moving Wall
  * Sf is the outward-from-owner face vector (length x unit normal) in SI,
    owner < neighbor never assumed by the solver, neighbor = -1 on boundaries.
"""
import numpy as np

_BC_TAGS = {"Wall": 0, "Velocity Inlet": 1, "Pressure Outlet": 2,
            "Symmetry": 3, "Moving Wall": 4}


def _coords(a, b, n, clustering):
    """n+1 grid lines on [a, b]; clustering > 0 packs them toward BOTH ends
    (tanh stretching, ~0 = uniform). Cell size at the wall relative to uniform
    is roughly clustering-dependent: 1.5 is mild, 2.5 strong."""
    xi = np.linspace(0.0, 1.0, n + 1)
    if clustering > 1e-9:
        xi = 0.5 * (1.0 + np.tanh(clustering * (2.0 * xi - 1.0)) / np.tanh(clustering))
    return a + (b - a) * xi


def structured_rect_mesh(x0, y0, x1, y1, nx, ny, bc_map,
                         wall_velocities=None, unit_to_meters=1.0, clustering=0.0):
    """Build the solver mesh dict for an nx-by-ny uniform quad grid.

    x0..y1: corners in world units (scaled to metres by unit_to_meters).
    bc_map: {edge index: boundary_type string}, all four edges required.
    wall_velocities: {edge index: (vx, vy)} in m/s, required for every edge
        whose type is "Moving Wall" (no silent zero default).
    clustering: 0 = uniform grid; > 0 packs cells toward all four walls
        (see _coords). Orthogonality is preserved either way.
    """
    if set(bc_map) != {0, 1, 2, 3}:
        raise ValueError("bc_map must give a boundary type for edges 0,1,2,3")
    wall_velocities = wall_velocities or {}
    edge_tag = {}
    for e, name in bc_map.items():
        if name not in _BC_TAGS:
            raise ValueError(f"unknown boundary type {name!r} on edge {e}")
        edge_tag[e] = _BC_TAGS[name]
        if name == "Moving Wall" and e not in wall_velocities:
            raise ValueError(f"edge {e} is a Moving Wall but has no wall_velocities entry")

    s = float(unit_to_meters)
    xs = _coords(x0, x1, nx, clustering) * s
    ys = _coords(y0, y1, ny, clustering) * s
    dx, dy = np.diff(xs), np.diff(ys)

    cid = lambda i, j: j * nx + i                       # cell index
    Nc = nx * ny
    II, JJ = np.meshgrid(np.arange(nx), np.arange(ny))  # (ny, nx), cell (i,j) -> [j,i]
    cell_centers = np.stack([(xs[II] + xs[II + 1]) / 2,
                             (ys[JJ] + ys[JJ + 1]) / 2], axis=-1).reshape(Nc, 2)
    cell_areas = (dx[II] * dy[JJ]).reshape(Nc)

    owner, neighbor, Sf, Cf, tags = [], [], [], [], []

    # Vertical faces at x = xs[i], spanning row j.  i==0 left, i==nx right.
    for j in range(ny):
        for i in range(nx + 1):
            Cf.append((xs[i], (ys[j] + ys[j + 1]) / 2))
            if i == 0:
                owner.append(cid(0, j));      neighbor.append(-1)
                Sf.append((-dy[j], 0.0));     tags.append(edge_tag[3])
            elif i == nx:
                owner.append(cid(nx - 1, j)); neighbor.append(-1)
                Sf.append((dy[j], 0.0));      tags.append(edge_tag[1])
            else:
                owner.append(cid(i - 1, j));  neighbor.append(cid(i, j))
                Sf.append((dy[j], 0.0));      tags.append(-1)
    # Horizontal faces at y = ys[j], spanning column i.  j==0 bottom, j==ny top.
    for j in range(ny + 1):
        for i in range(nx):
            Cf.append(((xs[i] + xs[i + 1]) / 2, ys[j]))
            if j == 0:
                owner.append(cid(i, 0));      neighbor.append(-1)
                Sf.append((0.0, -dx[i]));     tags.append(edge_tag[0])
            elif j == ny:
                owner.append(cid(i, ny - 1)); neighbor.append(-1)
                Sf.append((0.0, dx[i]));      tags.append(edge_tag[2])
            else:
                owner.append(cid(i, j - 1));  neighbor.append(cid(i, j))
                Sf.append((0.0, dx[i]));      tags.append(-1)

    owner = np.array(owner, dtype=np.int32)
    neighbor = np.array(neighbor, dtype=np.int32)
    Sf = np.array(Sf, dtype=np.float64)
    Cf = np.array(Cf, dtype=np.float64)
    tags = np.array(tags, dtype=np.int64)
    Nf = len(owner)

    df = np.where((neighbor >= 0)[:, None],
                  cell_centers[np.maximum(neighbor, 0)] - cell_centers[owner],
                  Cf - cell_centers[owner])

    # Per-face sliding velocity: edge -> tag lookup by the face's boundary tag
    # is ambiguous if two edges share a tag, so resolve by geometry instead.
    face_wall_velocity = np.zeros((Nf, 2))
    tol = 1e-9 * max(xs[-1] - xs[0], ys[-1] - ys[0])
    edge_test = {0: Cf[:, 1] - ys[0], 1: Cf[:, 0] - xs[-1],
                 2: Cf[:, 1] - ys[-1], 3: Cf[:, 0] - xs[0]}
    for e, v in wall_velocities.items():
        if bc_map[e] == "Moving Wall":
            m = (tags == 4) & (np.abs(edge_test[e]) < tol)
            face_wall_velocity[m] = v

    cell_vertices = np.zeros((Nc, 4, 2))
    for k, (di, dj) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):   # CCW
        cell_vertices[:, k, 0] = xs[II + di].reshape(Nc)
        cell_vertices[:, k, 1] = ys[JJ + dj].reshape(Nc)

    corners = ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0))
    cad_lines = np.array(
        [[*corners[e], _BC_TAGS[bc_map[e]], *wall_velocities.get(e, (0.0, 0.0))]
         for e in range(4)], dtype=np.float64)

    h = (x1 - x0) / nx   # mean spacing (world units); nominal only when clustered
    return {
        'Nc': Nc, 'Nf': Nf,
        'owner': owner, 'neighbor': neighbor,
        'Sf': Sf, 'magSf': np.linalg.norm(Sf, axis=1),
        'Cf': Cf, 'df': df, 'magDf': np.linalg.norm(df, axis=1),
        'cell_centers': cell_centers, 'cell_areas': cell_areas,
        'boundary_tags': tags, 'face_wall_velocity': face_wall_velocity,
        'cell_vertices': cell_vertices,
        'cell_nverts': np.full(Nc, 4, dtype=np.int32),
        'cell_types': np.ones(Nc, dtype=np.int32),
        'cad_lines': cad_lines,
        # meshing parameters (ride along for save/load; solver ignores)
        'n_layers': 0, 'growth_factor': 1.0, 'thickness': 0.0,
        'boundary_spacing': h, 'bc_spacing_map': {}, 'r': h,
        'unit_to_meters': s,
        'refinement_zones': np.array([], dtype=object),
    }
