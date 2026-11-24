"""Couette flow validation of the Moving Wall boundary (per-line velocity vector).

Exact solution between parallel plates, no pressure gradient:
    u(y) = U_bottom + (U_top - U_bottom) * y / H,   dp/dx = 0,   tau_w = mu*(U_top-U_bottom)/H
A finite channel with a uniform inlet at the exact MEAN velocity and a pressure
outlet reproduces it: the profile develops from uniform to linear within a few H
(Re = rho*U*H/mu is kept small so that entry length is short), and the mass flux
already matches the exact solution, so no pressure gradient is needed.

Cases (all use structured quad grids -- see cfdeditor/structured_mesh.py):
  A  top plate +U, bottom fixed         (one moving wall, inlet U/2)
  B  top +U/2, bottom -U/2              (two DIFFERENT per-line velocities, no net flow)
  C  vertical channel, right wall +U in y  (tests the y-component of wall_velocity)

Run:  .venv/Scripts/python.exe validation/couette.py [ny]
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cfdeditor.api import solve_mesh_data
from cfdeditor.structured_mesh import structured_rect_mesh

UNIT = 0.001
H_MM, L_MM = 10.0, 100.0
H, L = H_MM * UNIT, L_MM * UNIT
RHO, U = 1.0, 1.0
MU = RHO * U * H / 1.0            # Re = rho*U*H/mu = 1

# Stop rule is OFF by default (Solver.stop_rule): fixed iteration count past
# convergence. The old continuity-only rule fired at iteration ~51 on case B
# (no inlet flux) with the profile 7% off; it is converged (0.0005%) by ~800.
MAX_ITERS = 2000


def run(name, bc, wvel, inlet, vertical, ny):
    nx = int(round(ny * L_MM / H_MM))
    if vertical:   # tall channel: width H (x), length L (y)
        m = structured_rect_mesh(0, 0, H_MM, L_MM, ny, nx, bc, wvel, UNIT)
    else:
        m = structured_rect_mesh(0, 0, L_MM, H_MM, nx, ny, bc, wvel, UNIT)
    h = solve_mesh_data(m, dict(inlet_velocity=inlet, outlet_pressure=0.0, rho=RHO,
                                viscosity=MU, max_iterations=MAX_ITERS),
                        verbose=False)
    Uf, P = h.results.U, h.results.P
    cc = m['cell_centers']
    if vertical:
        s, a = cc[:, 0], cc[:, 1]       # across-channel coordinate, along-channel
        comp = Uf[:, 1]; lo, hi = 0.0, wvel[1][1]   # left wall fixed, right wall moves
    else:
        s, a = cc[:, 1], cc[:, 0]
        comp = Uf[:, 0]; lo = wvel[0][0] if 0 in wvel else 0.0; hi = wvel[2][0]
    exact = lo + (hi - lo) * s / H
    win = (a > 0.6 * L) & (a < 0.9 * L)                 # developed, away from the outlet
    err = comp[win] - exact[win]
    l2 = np.sqrt(np.mean(err**2)) / U
    cross = Uf[win, 0] if vertical else Uf[win, 1]       # spurious cross-stream velocity
    dpda = np.polyfit(a[win], P[win], 1)[0]
    print(f"  {name:<44} cells {m['Nc']:6d}   profile L2 = {100*l2:7.4f}% of U"
          f"   max|err| = {100*np.abs(err).max()/U:7.4f}%"
          f"   max|cross| = {np.abs(cross).max():.1e}   dp/dx = {dpda:+.2e} Pa/m (exact 0)")
    return l2


def main(ny=32):
    print(f"Couette flow, Re = {RHO*U*H/MU:.0f}, ny = {ny} cells across H, {int(round(ny*L_MM/H_MM))} along L")
    run("A: top +U, bottom fixed",
        {0: "Wall", 1: "Pressure Outlet", 2: "Moving Wall", 3: "Velocity Inlet"},
        {2: (U, 0.0)}, [0.5 * U, 0.0], False, ny)
    run("B: top +U/2, bottom -U/2 (per-line vectors)",
        {0: "Moving Wall", 1: "Pressure Outlet", 2: "Moving Wall", 3: "Velocity Inlet"},
        {0: (-0.5 * U, 0.0), 2: (0.5 * U, 0.0)}, [0.0, 0.0], False, ny)
    run("C: vertical channel, right wall +U in y",
        {0: "Velocity Inlet", 1: "Moving Wall", 2: "Pressure Outlet", 3: "Wall"},
        {1: (0.0, U)}, [0.0, 0.5 * U], True, ny)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
