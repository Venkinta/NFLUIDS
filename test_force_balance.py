"""Regression test: discrete force/mass balance on a freshly-solved channel,
read straight from the solver's own cell/face data -- no ParaView, no MATLAB.

For steady, fully-developed channel flow, momentum conservation over a slab
of the developed region requires exactly:

    -dp/dx * H  ==  2 * tau_w

This holds from the discretization's own conservation property, independent
of mesh coarseness -- unlike comparing dp/dx to analytical theory (which
does depend on how well-resolved the mesh is), a force-balance violation
here means the pressure-velocity coupling itself is broken. tau_w is
computed exactly the way the solver applies wall drag in its momentum
equation (see solver.py: diff[wall] = mu*magSf/magDf), not by any
after-the-fact fitting -- this is what closed the +10-20% gap an earlier
ParaView-based check showed, which turned out to be ParaView interpolating
u to ~3e-4 at the wall instead of the true 0.

Promoted from a scratchpad diagnostic (force_balance.py) that pointed at
solved .npz files in the out-of-repo Meshes/Poiseuille validation set. This
version is self-contained: it builds a small channel mesh in-code and solves
it directly (same Point/Line/Mesher/Solver pattern as test_holes.py), so it
needs no external data, no ParaView, and no MATLAB, and runs in seconds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from cfdeditor.mesher import Mesher
from cfdeditor.solver import Solver
from cfdeditor.geometry_helpers import rect_lines

# Same fluid as the Poiseuille validation suite (validation/poiseuille_config.m)
MU, RHO = 1.0, 1000.0
U_IN = 0.1

# Channel geometry, world units = mm (unit_to_meters converts to SI below).
# n_layers/growth_factor/thickness/spacing/r below reuse test_holes.py's
# known-stable Mesher settings, just on an elongated (channel) aspect ratio.
H_MM, L_MM = 100.0, 600.0
UNIT_TO_M = 0.001
H = H_MM * UNIT_TO_M
L = L_MM * UNIT_TO_M

DPDX_THEORY = -12.0 * MU * U_IN / H**2

# Developed window: skip entrance/exit effects (mirrors the MATLAB suite's
# FIT_LO/FIT_HI convention in validation/poiseuille_config.m)
FIT_LO = 0.20 * L
FIT_HI = L - 0.10 * L

# Regression bands -- set from this test's own solved output, not guessed.
# Force balance is a self-consistency check (independent of grid coarseness)
# so it gets the tightest band; dp/dx-vs-theory depends on how resolved this
# particular (deliberately small, fast) mesh is, so it gets a looser one.
MASS_ERROR_MAX_PCT = 1.0
FORCE_BALANCE_MAX_PCT = 3.0
DPDX_THEORY_MAX_PCT = 10.0


def main():
    lines = rect_lines(0, 0, L_MM, H_MM, {
        0: "Wall",              # bottom
        1: "Pressure Outlet",   # right
        2: "Wall",              # top
        3: "Velocity Inlet",    # left
    })

    mesher = Mesher(
        lines=lines,
        n_layers=3, growth_factor=1.2, thickness=1.0,
        spacing=5.0, r=4.0, unit_to_meters=UNIT_TO_M,
    )
    print("=== MESH ===")
    mesher.mesh()
    data = mesher.solver_data_pipeline()
    print(f"Cells: {data['Nc']}, Faces: {data['Nf']}")

    print("\n=== SOLVE ===")
    solver = Solver(data, [U_IN, 0.0], 0.0, RHO, MU,
                     alpha_u=0.7, alpha_p=0.3, max_iterations=1600, tolerance=1e-8)
    solver.Solve()

    owner, tags = solver.owner, solver.boundary_tags
    Sf, Cf, df, magDf, magSf = solver.Sf, solver.Cf, solver.df, solver.magDf, solver.magSf
    U, P, cc = solver.U, solver.P, solver.cell_centers

    inlet  = np.where(tags == 1)[0]
    outlet = np.where(tags == 2)[0]
    wall   = np.where(tags == 0)[0]

    # --- Mass conservation: measured outlet flux vs the exactly-known
    # prescribed inlet flux (uniform Dirichlet inlet velocity over height H) ---
    mdot_in  = RHO * U_IN * H
    mdot_out = RHO * float(np.sum(np.einsum('fj,fj->f', U[owner[outlet]], Sf[outlet])))
    mass_error_pct = abs(mdot_out - mdot_in) / mdot_in * 100.0

    # --- dp/dx from raw cell centers in the developed window ---
    m = (cc[:, 0] >= FIT_LO) & (cc[:, 0] <= FIT_HI)
    assert np.sum(m) >= 3, "developed window contains too few cells to fit dp/dx"
    dpdx = np.polyfit(cc[m, 0], P[m], 1)[0]
    dpdx_error_pct = abs(dpdx - DPDX_THEORY) / abs(DPDX_THEORY) * 100.0

    # --- tau_w exactly as the solver applies it, restricted to the
    # developed window, length-weighted mean over both walls ---
    wall_dev = wall[(Cf[wall, 0] >= FIT_LO) & (Cf[wall, 0] <= FIT_HI)]
    assert len(wall_dev) >= 2, "developed window contains too few wall faces"
    own_w = owner[wall_dev]
    tau = MU * np.abs(U[own_w, 0]) / magDf[wall_dev]
    tau_mean = float(np.sum(tau * magSf[wall_dev]) / np.sum(magSf[wall_dev]))

    F_press = -dpdx * H
    F_wall = 2.0 * tau_mean
    force_balance_pct = abs(F_press - F_wall) / F_wall * 100.0

    print("\n=== RESULT ===")
    print(f"  mass error       : {mass_error_pct:6.3f}%  (band < {MASS_ERROR_MAX_PCT}%)")
    print(f"  force balance    : {force_balance_pct:6.3f}%  (band < {FORCE_BALANCE_MAX_PCT}%)"
          f"   [-dp/dx*H={F_press:.3f}  2*tau_w={F_wall:.3f}]")
    print(f"  dp/dx vs theory  : {dpdx_error_pct:6.3f}%  (band < {DPDX_THEORY_MAX_PCT}%)"
          f"   [measured={dpdx:.1f}  theory={DPDX_THEORY:.1f}]")

    ok = (np.all(np.isfinite(solver.P)) and np.all(np.isfinite(solver.U))
          and mass_error_pct < MASS_ERROR_MAX_PCT
          and force_balance_pct < FORCE_BALANCE_MAX_PCT
          and dpdx_error_pct < DPDX_THEORY_MAX_PCT)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
