"""Lid-driven cavity, Re = 100, vs. Ghia, Ghia & Shin (1982) centerline data.

Square cavity (100 mm side, drawn in mm like the channel tests), lid (top)
sliding at U_LID in +x, other three walls stationary.
Closed domain: no outlet, so the solver pins p' = 0 in cell 0 (see
Solver.ASSEMBLE_PRESSURE_CORRECTION) -- nothing to set up here.

Re = rho * U_LID * L / mu = 1 * 1 * 0.1 / 1e-3 = 100.

Geometry is in mm (world units) on purpose: mesher.connect_layers() skips
boundary-layer quads with a hard-coded world-unit area < 1e-4, so a cavity
drawn as a 1x1 *world-unit* square silently loses its boundary layer.

Reference data: read directly from Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982)
387-411 (checked 2026-11-24 against the page images): Table I (p. 398) Re=100
column, Table II (p. 399) Re=100 column, and the vortex-properties table (p. 408)
Re=100 column. All entries below are exactly as printed. (Page 403's "Table III"
in that paper is streamline contour levels, not vortex data.) The paper's vorticity
is positive at the primary vortex (its sign convention); we report |omega|.

Run:  .venv/Scripts/python.exe validation/cavity_ghia.py [spacing_mm] [tri|quad]

`quad` uses cfdeditor.structured_mesh (uniform orthogonal grid, n = L/spacing
cells per side) -- isolates solver error from triangle-mesher error.
"""
import sys
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cfdeditor.api import run_solve, solve_mesh_data
from cfdeditor.structured_mesh import structured_rect_mesh
from cfdeditor.geometry_helpers import rect_lines

L_MM, UNIT = 100.0, 0.001
L = L_MM * UNIT
RHO, U_LID = 1.0, 1.0
MU = RHO * U_LID * L / 100.0   # Re = 100

# Stop rule is OFF by default (see Solver.stop_rule): run a fixed iteration
# count past convergence. The old continuity-only rule stopped this case at
# iteration ~66 with core u = -0.101 instead of the converged -0.196.
MAX_ITERS = int(os.environ.get("CAV_ITERS", 2000))
CLUSTER = float(os.environ.get("CAV_CLUSTER", 0.0))   # quad mesh wall clustering

# u(y) on x = 0.5
GHIA_U = np.array([
    (1.0000, 1.00000), (0.9766, 0.84123), (0.9688, 0.78871), (0.9609, 0.73722),
    (0.9531, 0.68717), (0.8516, 0.23151), (0.7344, 0.00332), (0.6172, -0.13641),
    (0.5000, -0.20581), (0.4531, -0.21090), (0.2813, -0.15662), (0.1719, -0.10150),
    (0.1016, -0.06434), (0.0703, -0.04775), (0.0625, -0.04192), (0.0547, -0.03717),
    (0.0000, 0.00000)])
# v(x) on y = 0.5
GHIA_V = np.array([
    (1.0000, 0.00000), (0.9688, -0.05906), (0.9609, -0.07391), (0.9531, -0.08864),
    (0.9453, -0.10313), (0.9063, -0.16914), (0.8594, -0.22445), (0.8047, -0.24533),
    (0.5000, 0.05454), (0.2344, 0.17527), (0.2266, 0.17507), (0.1563, 0.16077),
    (0.0938, 0.12317), (0.0781, 0.10890), (0.0703, 0.10091), (0.0625, 0.09233),
    (0.0000, 0.00000)])

# Ghia et al. (1982) Re = 100, vortex-properties table (p. 408), exactly as printed.
GHIA_PRIMARY = dict(psi=-0.103423, omega=3.16646, x=0.6172, y=0.7344)
GHIA_SECONDARY = dict(BL=dict(psi=1.74877e-6, x=0.0313, y=0.0391),
                      BR=dict(psi=1.25374e-5, x=0.9453, y=0.0625))
# Extremes of the tabulated centerline profiles above (grid-point values).
GHIA_EXTREMA = dict(u_min=-0.21090, v_max=0.17527, v_min=-0.24533)


def characteristics(md, U, fu, fv):
    """Stream function / vortex quantities on the structured quad grid.
    psi is built by integrating u up each vertical grid line (u = dpsi/dy,
    v = -dpsi/dx, psi = 0 on the bottom wall), so psi(top) = 0 only up to the O(h)
    error of reconstructing face velocities from cell values (it halves with h)."""
    cv = md['cell_vertices'] / L
    xs = np.unique(np.round(cv[:, :, 0], 12)); ys = np.unique(np.round(cv[:, :, 1], 12))
    nx, ny = len(xs) - 1, len(ys) - 1
    u = U[:, 0].reshape(ny, nx); dy = np.diff(ys)
    # u on the vertical grid line x_i, row j: linear interp of the two adjacent cells
    # (walls: u = 0 there is NOT assumed for the lid corner rows, only for i = 0, nx)
    xc = (xs[:-1] + xs[1:]) / 2
    ue = np.zeros((ny, nx + 1))
    for i in range(1, nx):
        w = (xs[i] - xc[i - 1]) / (xc[i] - xc[i - 1])
        ue[:, i] = (1 - w) * u[:, i - 1] + w * u[:, i]
    psi = np.zeros((ny + 1, nx + 1))
    psi[1:, :] = np.cumsum(ue * dy[:, None], axis=0)
    closure = np.abs(psi[-1, :]).max()

    spl = RectBivariateSpline(ys, xs, psi, kx=3, ky=3)
    j0, i0 = np.unravel_index(np.argmin(psi), psi.shape)
    r = minimize(lambda q: spl.ev(q[1], q[0])[()], [xs[i0], ys[j0]],
                 bounds=[(0, 1), (0, 1)], method="L-BFGS-B")
    xv, yv = r.x
    omega = -(spl.ev(yv, xv, dx=2) + spl.ev(yv, xv, dy=2))   # v_x - u_y

    def corner(xlo, xhi):
        sub = psi[1:-1, 1:-1].copy()
        X, Y = np.meshgrid(xs[1:-1], ys[1:-1])
        sub[~((X > xlo) & (X < xhi) & (Y < 0.35))] = -np.inf
        jj, ii = np.unravel_index(np.argmax(sub), sub.shape)
        q = minimize(lambda q: -spl.ev(q[1], q[0])[()], [xs[1:-1][ii], ys[1:-1][jj]],
                     bounds=[(xlo, xhi), (0, 0.35)], method="L-BFGS-B")
        return -q.fun, q.x

    t = np.linspace(0, 1, 2001)
    u_line = fu(np.full_like(t, 0.5), t); v_line = fv(t, np.full_like(t, 0.5))
    return dict(closure=closure, psi_min=float(r.fun), xv=xv, yv=yv, omega=float(np.ravel(omega)[0]),
                BR=corner(0.5, 1.0), BL=corner(0.0, 0.5),
                u_min=np.nanmin(u_line), v_max=np.nanmax(v_line), v_min=np.nanmin(v_line))


def print_characteristics(c):
    g, e = GHIA_PRIMARY, GHIA_EXTREMA
    def row(name, ours, ref):
        d = "" if ref is None else f"  ({100*(ours-ref)/abs(ref):+6.2f}%)"
        print(f"  {name:<28} {ours:12.5f}   Ghia {'' if ref is None else f'{ref:12.5f}'}{d}")
    print("\nCharacteristic quantities (Re = 100)")
    print(f"  psi(lid) closure |psi|max    {c['closure']:12.2e}   (reconstruction error, O(h); not a solver mass error)")
    row("primary vortex psi_min", c['psi_min'], g['psi'])
    row("primary vortex x", c['xv'], g['x'])
    row("primary vortex y", c['yv'], g['y'])
    row("|vorticity| at centre", abs(c['omega']), g['omega'])
    row("u_min on x=0.5", c['u_min'], e['u_min'])
    row("v_max on y=0.5", c['v_max'], e['v_max'])
    row("v_min on y=0.5", c['v_min'], e['v_min'])
    for k, nm in (("BR", "bottom-right"), ("BL", "bottom-left")):
        psi_c, (xc_, yc_) = c[k]
        g = GHIA_SECONDARY[k]
        print(f"  secondary {nm:<12} psi={psi_c:.4e} at ({xc_:.4f}, {yc_:.4f})   "
              f"Ghia {g['psi']:.4e} at ({g['x']:.4f}, {g['y']:.4f})  ({100*(psi_c-g['psi'])/g['psi']:+.1f}%)")


def main(spacing=2.0, kind="tri"):   # spacing in mm
    solver_params = dict(inlet_velocity=[0.0, 0.0], outlet_pressure=0.0,
                         rho=RHO, viscosity=MU, max_iterations=MAX_ITERS)
    if kind == "quad":
        n = int(round(L_MM / spacing))
        mesh = structured_rect_mesh(
            0, 0, L_MM, L_MM, n, n,
            {0: "Wall", 1: "Wall", 2: "Moving Wall", 3: "Wall"},
            wall_velocities={2: (U_LID, 0.0)}, unit_to_meters=UNIT,
            clustering=CLUSTER)
        h = solve_mesh_data(mesh, solver_params, verbose=False)
    else:
        lines = rect_lines(0, 0, L_MM, L_MM, {0: "Wall", 1: "Wall", 2: "Moving Wall", 3: "Wall"})
        lines[2].wall_velocity = (U_LID, 0.0)          # top edge = lid
        h = run_solve(
            lines,
            mesh_params=dict(n_layers=4, growth_factor=1.2, thickness=spacing / 4,
                             spacing=spacing, r=spacing, unit_to_meters=UNIT),
            solver_params=solver_params, verbose=False)

    md, U = h.mesh_data, h.results.U
    cc = md['cell_centers'] / L          # normalise to the unit square
    tags = md['boundary_tags']
    # Add boundary-face centres with their known (no-slip / lid) velocity so the
    # interpolant covers the whole square, including the thin wall-adjacent strip.
    bf = np.where(tags >= 0)[0]
    bU = np.zeros((len(bf), 2))
    bU[tags[bf] == 4] = md['face_wall_velocity'][bf][tags[bf] == 4]
    pts = np.vstack([cc, md['Cf'][bf] / L])
    vals = np.vstack([U, bU])
    fu = LinearNDInterpolator(pts, vals[:, 0])
    fv = LinearNDInterpolator(pts, vals[:, 1])

    u_num = fu(np.full(len(GHIA_U), 0.5), GHIA_U[:, 0])
    v_num = fv(GHIA_V[:, 0], np.full(len(GHIA_V), 0.5))
    eu, ev = u_num - GHIA_U[:, 1], v_num - GHIA_V[:, 1]

    print(f"\nCells: {md['Nc']}   lid speed {U_LID}   Re = {RHO*U_LID*L/MU:.0f}")
    print(f"u(y) on x=0.5 : max |err| = {np.nanmax(np.abs(eu)):.4f}   "
          f"RMS = {np.sqrt(np.nanmean(eu**2)):.4f}   (Ghia range 1.0 -> -0.21)")
    print(f"v(x) on y=0.5 : max |err| = {np.nanmax(np.abs(ev)):.4f}   "
          f"RMS = {np.sqrt(np.nanmean(ev**2)):.4f}   (Ghia range +0.18 -> -0.25)")
    print(f"\n{'y':>7} {'u_ghia':>9} {'u_nfl':>9}   {'x':>7} {'v_ghia':>9} {'v_nfl':>9}")
    for i in range(max(len(GHIA_U), len(GHIA_V))):
        a = (f"{GHIA_U[i,0]:7.4f} {GHIA_U[i,1]:9.5f} {u_num[i]:9.5f}"
             if i < len(GHIA_U) else " " * 27)
        b = (f"{GHIA_V[i,0]:7.4f} {GHIA_V[i,1]:9.5f} {v_num[i]:9.5f}"
             if i < len(GHIA_V) else "")
        print(f"{a}   {b}")
    if kind == "quad":
        print_characteristics(characteristics(md, U, fu, fv))
    return h


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 2.0,
         sys.argv[2] if len(sys.argv) > 2 else "tri")
