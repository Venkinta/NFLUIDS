"""
ParaView batch export for the Poiseuille validation suite.

SET `DATA_DIR` BELOW. That is the only configuration in this file.

How to run (either works):
  * ParaView GUI:  Tools > Python Shell, then:
        exec(open(r"C:\\...\\NFLUIDS\\validation\\paraview_export_velocity_poiseuille.py").read())
  * Command line:  pvpython paraview_export_velocity_poiseuille.py

Exports, per solved VTU:
  1. poiseuille_profile_<tag>.csv    velocity profile across the channel at 90% L
                                     (the primary profile used for L2 / u_max /
                                     symmetry / wall shear)
  2. poiseuille_pressure_<tag>.csv   centerline pressure along the full channel
  3. poiseuille_station_<tag>_NN.csv velocity profiles at several x stations
                                     (mass conservation vs x, profile development)
  4. poiseuille_centerline_<tag>.csv u along the centerline (entrance length)

Plus one shared file:
  5. poiseuille_meta.csv             real cell/point counts per mesh, so the
                                     MATLAB side never hardcodes "36000" when the
                                     mesh is actually 36498 cells. The grid-
                                     convergence (GCI) script needs true counts.

NOTE ON SAMPLING: PlotOverLine *interpolates* onto a uniform set of sample
points; the exported rows are not raw cell values. Points that land exactly on a
domain boundary fall outside every cell and come back NaN with
vtkValidPointMask=0 (the channel inlet does this) -- always filter on the mask
downstream. Station lines are therefore inset slightly from the walls/inlet.
"""
import os
import csv
from paraview.simple import *

# =============================================================================
#  >>> THE ONLY LINE YOU EVER NEED TO EDIT <<<
#  Where the solved .vtu files live. The CSVs are written back into the same
#  folder. No searching, no fallbacks: if this path is wrong the script stops
#  immediately and says so.
# =============================================================================
DATA_DIR = r"C:\Users\nesti\Documents\PYTHON_LEARNING\Meshes\Poiseuille"
# =============================================================================

print("=" * 70)
print("  Poiseuille ParaView export")
print("  DATA_DIR: %s" % DATA_DIR)
print("=" * 70)

# NEVER raise SystemExit here. This script is usually run inside ParaView's
# embedded interpreter (Tools > Python Shell), where SystemExit propagates and
# TERMINATES PARAVIEW -- no message, no CSVs, the window just closes. A normal
# exception is caught by ParaView and shown in the shell instead.
if not os.path.isdir(DATA_DIR):
    raise RuntimeError(
        "DATA_DIR does not exist:\n    %s\n"
        "Edit DATA_DIR at the top of this script." % DATA_DIR)

_vtus = [f for f in os.listdir(DATA_DIR) if f.endswith(".vtu")]
if not _vtus:
    raise RuntimeError(
        "DATA_DIR exists but contains no .vtu files:\n    %s\n"
        "Solve and export the meshes first, or fix DATA_DIR." % DATA_DIR)
print("Found %d .vtu file(s) in DATA_DIR." % len(_vtus))

data_dir = DATA_DIR

# Must match cfg.tags in poiseuille_config.m. Files are expected to follow the
# convention poiseuille_solved_<tag>.vtu -- keep new meshes to that pattern.
mesh_tags = ["3k", "10k", "36k", "62k", "135k", "quads", "quads_2"]

# Channel geometry (SI metres) -- must match poiseuille_config.m
X_START, X_END = 0.2365, 0.3365
Y_START, Y_END = 0.27, 0.28
L = X_END - X_START      # 0.1 m
H = Y_END - Y_START      # 0.01 m
Y_MID = 0.5 * (Y_START + Y_END)

PROFILE_X_FRAC = 0.90    # primary profile station (fully developed, see config)
PROFILE_RES = 1000       # samples across H
CENTERLINE_RES = 2000    # samples along L

# Mass-conservation / development stations, as fractions of L.
# Clustered near the inlet where the profile is still developing.
STATION_FRACS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.98]

# Keep sample lines a hair inside the domain so endpoints hit real cells.
EPS = 1e-6


def _plot_over_line(reader, p1, p2, resolution):
    """PlotOverLine with uniform sampling, tolerant of ParaView version drift."""
    pol = PlotOverLine(Input=reader)
    pol.Point1 = p1
    pol.Point2 = p2
    # 'Resolution'/'SamplingPattern' moved around between ParaView versions;
    # a missing property must not abort the whole export.
    try:
        pol.SamplingPattern = 'Sample Uniformly'
    except Exception:
        pass
    try:
        pol.Resolution = resolution
    except Exception:
        pass
    UpdatePipeline(proxy=pol)
    return pol


def _save(proxy, path, what):
    SaveData(path, proxy=proxy)
    print("  --> %-22s %s" % (what, os.path.basename(path)))


meta_rows = []

for tag in mesh_tags:
    filename = "poiseuille_solved_%s.vtu" % tag
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print("SKIP (missing): %s" % filename)
        continue

    print("Processing: %s ..." % filename)
    reader = OpenDataFile(filepath)
    UpdatePipeline(proxy=reader)

    # --- Real cell/point counts (kills the hardcoded-3000 problem) ---
    info = reader.GetDataInformation()
    n_cells = info.GetNumberOfCells()
    n_points = info.GetNumberOfPoints()
    meta_rows.append((tag, n_cells, n_points))
    print("  mesh: %d cells, %d points" % (n_cells, n_points))

    # --- A: primary velocity profile at 90% L ---
    x_prof = X_START + PROFILE_X_FRAC * L
    prof = _plot_over_line(reader,
                           [x_prof, Y_START, 0.0],
                           [x_prof, Y_END, 0.0],
                           PROFILE_RES)
    _save(prof, os.path.join(data_dir, "poiseuille_profile_%s.csv" % tag),
          "Velocity profile:")
    Delete(prof)

    # --- B: centerline pressure along the full channel ---
    press = _plot_over_line(reader,
                            [X_START, Y_MID, 0.0],
                            [X_END, Y_MID, 0.0],
                            CENTERLINE_RES)
    _save(press, os.path.join(data_dir, "poiseuille_pressure_%s.csv" % tag),
          "Centerline pressure:")
    Delete(press)

    # --- C: centerline u(x) for entrance length ---
    # Same line as B, but exported separately so the pressure file stays the
    # canonical dp/dx source and this one can be resampled independently.
    cline = _plot_over_line(reader,
                            [X_START + EPS, Y_MID, 0.0],
                            [X_END - EPS, Y_MID, 0.0],
                            CENTERLINE_RES)
    _save(cline, os.path.join(data_dir, "poiseuille_centerline_%s.csv" % tag),
          "Centerline u(x):")
    Delete(cline)

    # --- D: profiles at several stations for mass conservation vs x ---
    for i, frac in enumerate(STATION_FRACS):
        xs = X_START + frac * L
        xs = min(max(xs, X_START + EPS), X_END - EPS)
        st = _plot_over_line(reader,
                             [xs, Y_START, 0.0],
                             [xs, Y_END, 0.0],
                             PROFILE_RES)
        _save(st, os.path.join(data_dir,
                               "poiseuille_station_%s_%02d.csv" % (tag, i)),
              "Station %4.2fL:" % frac)
        Delete(st)

    Delete(reader)

# --- 2. METADATA ------------------------------------------------------------
meta_path = os.path.join(data_dir, "poiseuille_meta.csv")
with open(meta_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["tag", "n_cells", "n_points"])
    for row in meta_rows:
        w.writerow(row)
print("\nWrote %s" % meta_path)
print("Success! Export complete for %d mesh(es)." % len(meta_rows))
