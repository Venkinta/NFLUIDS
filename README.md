# NFLUIDS — Modular 2D Navier-Stokes Solver

A complete **2D Computational Fluid Dynamics (CFD) pipeline** with an interactive CAD frontend, built entirely in Python. Draw your geometry, assign boundary conditions, generate a hybrid mesh, solve the incompressible Navier-Stokes equations using the SIMPLE algorithm, and visualize results — all in one application.

## Verification

The solver is verified against the analytical plane-Poiseuille solution (ρ=1000 kg/m³, μ=1 Pa·s, H=0.01 m, ū=0.1 m/s prescribed ⇒ dp/dx = −12000 Pa/m):

| Quantity | Error vs analytical |
|---|---|
| Pressure gradient dp/dx | **0.11%** |
| Velocity profile (L2) | **0.26%** |
| Peak velocity u_max | **0.19%** |
| Mass conservation | **< 0.05%** |
| Discrete momentum balance (−dp/dx·H = 2τ_w) | **~1%** |

on a 5,706-cell quad-dominant mesh. Reproduce with the suite in [`validation/`](validation/) (`validate_all`).

**A note on mesh topology:** on hybrid meshes, thin interior triangles meeting the prismatic boundary layer inject a systematic **~+4.5% wall-drag error that does not reduce with refinement** — the error enters at the quad/triangle seam and offsets the whole boundary-layer stack. A 5,706-cell quad-dominant mesh reaches 0.11% dp/dx where a 36,026-cell hybrid mesh reaches only 6.4%. **For wall-bounded flows where wall shear or pressure drop matters, prefer quad-dominant meshes.** See `validation/` for the measurements.

## Quick Start

```bash
pip install -r requirements.txt
python NFluid.py
```

**Requires:** Python 3.10+, a GPU with OpenGL support.

## Workflow

```
CAD Drawing  →  Assign BCs  →  Generate Mesh  →  Solve   →  Visualize
   (EDITOR)      (PHYSICS)       (MESHER)      (SOLVING)   (VISUALIZER)
```

You can also **skip the CAD step** by loading a previously saved mesh:

```
Load Mesh  →  Inspect / Edit BCs  →  (Remesh)  →  Solve   →  Visualize
  (PHYSICS)      (PHYSICS)            (MESHER)   (SOLVING)   (VISUALIZER)
```

From Visualize you can also go straight back to **(PHYSICS)** — the mesh, boundary conditions, and view all stay intact, so you can tweak solver settings and re-solve immediately without remeshing.

### 1. Draw Your Geometry (EDITOR)
- **Left-click** to place points and draw lines (chain-drawing mode).
- **Escape** to cancel the current segment or undo the last line.
- **Mouse wheel** to zoom in/out; **middle-mouse drag** to pan.
- **Controls panel:**
  - **Units** — Choose mm, cm, or m for your drawing.
  - **Ortho (90° Snap)** — Constrain lines to horizontal/vertical.
  - **Exact Length** — Set a fixed line length (set to 0 for free drawing).
  - **Loop status** displayed in real-time — green "OK (Closed)" or yellow "OPEN!" with a warning if any loops are incomplete.
  - **"New Loop" button** — Breaks the drawing chain to start a separate shape (e.g. a hole). Hover for a tooltip explanation.
- **Vertex snapping** always takes priority over constraints, guaranteeing watertight loop closure.
- Click **"Finish CAD"** to proceed.

### 2. Assign Boundary Conditions (PHYSICS)
- **Click** any line to select it and open its properties panel.
- Assign each line a **boundary type**: `Wall`, `Velocity Inlet`, or `Pressure Outlet`.
- Configure **fluid properties** (density, viscosity).
- Configure **mesh parameters** (boundary layer settings, mesh size).
- Configure **Solver Settings** (relaxation factors `alpha_u`/`alpha_p`, max iterations, convergence tolerance, live-viz update interval).
- The **Reynolds number** estimator gives a DNS cell count prediction.
- Click **"Mesh"** to generate the computational grid.

When you **Load Mesh**, the saved meshing parameters (number of layers, growth
factor, first-layer thickness, boundary spacing, mesh size, and the world-unit
selection) are restored into the PHYSICS panel, so the UI reflects the values
that produced the loaded mesh. The CAD lines and their boundary types are also
reconstructed, so you can edit conditions or remesh before solving.
**Refinement zones** are also preserved across save/load.

**Load Visualization** (also in the PHYSICS panel) opens a previously
**saved visualization** file instead — same `.npz` format as Load Mesh, plus
the solved fields, so it jumps straight to the VISUALIZER screen showing the
same colors, vectors, and probe values as when it was saved, skipping the
mesh/solve steps entirely. "Back to Physics" from there works exactly like a
normal solve, with the mesh/BCs available for editing and re-solving.

### 3. Solve (SOLVING)
Clicking **"Solve"** launches the solver on a background thread, so the UI stays responsive throughout. It uses the **SIMPLE algorithm** with:
- Rhie-Chow interpolation for pressure-velocity coupling.
- BiCGSTAB with Jacobi (momentum) and ILU/PyAMG (pressure) preconditioners.
- Convergence is measured by RMS continuity residual (default tolerance: 1e-6).

A live **Solver Monitor** panel shows continuity and momentum RMS residual plots (log10) updated every frame, the current iteration count, and pause / step-one / stop controls. The mesh itself is colored live, refreshed every `viz_interval` iterations (configurable in Solver Settings) — a "Show" dropdown in the monitor lets you switch between Pressure, Velocity, Continuity Error, and Momentum Error while the solve is still running, same as the post-solve Visualizer.

### 4. Visualize Results (VISUALIZER)
- Switch between **Pressure**, **Velocity**, **Continuity Error**, and **Momentum Error** fields.
- Toggle **velocity vectors** with adjustable scale.
- Toggle **smoke particles** — tracer points spawned at the velocity inlet and advected through the solved velocity field, reseeding at the inlet only once they exit the mesh (they don't expire on a timer by default, so a particle drifting into a slow/recirculating region just stalls there). Adjustable speed, point size, and particle count, plus an optional lifetime limit if you want particles to expire after N seconds regardless of position.
- **Hover** over any cell to probe local values (P, Ux, Uy, residual errors).
- Click **"Save Visualization"** to export the mesh, solved fields, and current display settings (variable shown, vector toggle/scale) as a `.npz` — reopen it later with **Load Visualization** in the PHYSICS panel.
- Click **"Export VTU"** to write the mesh + solved fields (Pressure, Velocity, residuals) as a VTK XML UnstructuredGrid (`.vtu`) — open it in ParaView or another CFD tool to cross-validate the solver against a different code.
- Click **"Back to Physics"** to return to the Physics panel — mesh and BCs are preserved, so you can tweak solver settings and re-solve immediately.

## Features

### CAD Editor
- Vertex and axis snapping for precise geometry
- Orthogonal mode, exact lengths, configurable snap step
- Multiple unit systems (mm, cm, m)
- Real-time loop closure validation with warnings
- "New Loop" button for drawing holes and disconnected shapes
- OpenGL-backed smooth rendering

### Mesh Generation
- **Boundary layers**: Prismatic cells at walls with controlled growth rate
- **Interior mesh**: Poisson-disk Steiner point sampling + Delaunay triangulation (Bowyer-Watson)
- Hybrid mesh with quadrilaterals (boundary) and triangles (interior)
- **Multi-loop support**: Automatically detects outer boundary and internal holes via per-loop orientation analysis
- **Conformal hole meshing**: Triangle vertices from all loop layers feed into the triangulation, ensuring watertight connectivity between prisms and interior cells
- **Robust triangle filter**: Shapely polygon-intersection test removes any triangle that crosses a hole or outer boundary
- **Refinement zones**: Draw rectangular regions in the PHYSICS stage to locally refine the interior mesh. Each zone has a refinement factor `f`; the resulting Steiner point spacing inside the zone is `r / f` (where `r` is the global "Mesh size" parameter). The UI shows the resulting mesh size (`r / f`) next to each zone. Zones support graded transitions (smoothstep blending over a buffer zone) to avoid sudden cell-size jumps that cause numerical artefacts. Zones can be nested (e.g. a fine leading-edge box inside a coarser wingbox inside the background mesh) — wherever zones overlap, the smallest/most-specific zone always wins, so a nested zone's spacing applies across its entire footprint rather than losing out to a larger enclosing zone near its own center.
- Numba-accelerated circumcircle checks

### Solver
- Incompressible Navier-Stokes (SIMPLE algorithm)
- Finite-volume discretization on arbitrary polygonal meshes
- Rhie-Chow interpolation to prevent checkerboarding
- **Distance-weighted face interpolation** (`g_x = d_Pf / d_PN`) for correct behavior on non-uniform/refined meshes — makes refinement zones numerically sound (no artificial "resistance" at cell-size transitions)
- Multiple linear solver backends (BiCGSTAB + Jacobi/ILU/PyAMG)
- Numba JIT compilation for hot loops
- Built against a `SolverProtocol` ABC — alternate solver implementations (e.g. a future LES solver) are drop-in replacements with zero changes elsewhere in the app
- Runs on a background thread with a live residual-plot monitor, so the UI never blocks during a solve

### Post-Processor
- Color-mapped field visualization (jet-like native colormap)
- **Robust (percentile-clipped) color scaling** — the 2nd–98th percentile of each field sets the color range, so a single outlier cell (e.g. a leading-edge stagnation point) can't hijack the whole mesh's colors and make a converged field look like it's still swimming
- Interactive point probing with KD-tree spatial indexing
- Velocity vector glyphs with adjustable scale
- Smoke particle tracers, spawned at the velocity inlet and advected through the frozen post-solve velocity field, reseeding at the inlet on mesh exit — adjustable speed, point size, and particle count, with an optional user-set lifetime limit
- Log-scale residual visualization for error analysis
- OpenGL VBO-based rendering for large meshes

## Dependencies

| Package | Purpose |
|---------|---------|
| `PyOpenGL` | GPU-accelerated rendering |
| `pygame` | Windowing and event handling |
| `imgui` | UI framework |
| `numpy` | Array computations |
| `scipy` | Sparse linear algebra (BiCGSTAB, ILU) |
| `numba` | JIT compilation for hot loops |
| `shapely` | Polygon geometry (Poisson-disk sampling + triangle filtering) |
| `matplotlib` | Path containment tests (triangle filtering) |
| `pyamg` (optional) | Algebraic multigrid preconditioner |

Optional but recommended: `pip install pyamg` for faster pressure solves.

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Application entry point — pygame/OpenGL/imgui bootstrap and the frame loop |
| `app_state.py` | State machine: `AppState` enum, `AppContext`, per-state event/update/render dispatch tables |
| `renderer.py` | Shared rendering engine: frame lifecycle (one ImGui frame per loop), GL draw helpers, `VboHandle` buffer lifecycle, cross-cutting overlay hook (logo stamp) |
| `editor.py` | CAD drawing with snapping, constraints, loop validation |
| `physics_editor.py` | Boundary condition and mesh parameter UI |
| `mesher.py` | Mesh generation pipeline (multi-loop, holes) |
| `solver.py` | SIMPLE algorithm (Navier-Stokes solver) |
| `solver_protocol.py` | Abstract solver interface (`SolverProtocol`) |
| `solver_panel.py` | Threaded solve orchestration + live monitor UI |
| `visualizer.py` | Post-processing and field visualization |
| `camera.py` | World↔screen coordinate transforms and zoom (pure math — drawing lives in `renderer.py`) |
| `point.py` | 2D point geometry primitive |
| `line.py` | CAD edge with boundary metadata |
| `triangle.py` / `quad.py` | Cell geometry types |
| `triangulation.py` | O(1) add/remove triangle container |
| `bowyerwatson.py` | Delaunay triangulation algorithm |
| `constructor.py` | Numba JIT kernels and geometry helpers |
| `snapengine.py` | Vertex and axis snapping |
| `meshIO.py` | Save/Load mesh (and, via the same generic dict format, full visualizations) to/from compressed `.npz` files |
| `vtuIO.py` | Export mesh + solved fields as a VTK XML UnstructuredGrid (`.vtu`), for cross-validation in ParaView/other CFD codes |
| `test_holes.py` | End-to-end test for multi-loop meshing with holes |
| `test_force_balance.py` | Regression test: discrete force/mass balance on a solved channel |
| `test_state_transitions.py` | Characterization test for the `AppState` transition graph |
| `CODEBASE_REFERENCE.md` | Internal developer documentation |

## Known Limitations

- **Units migration in progress**: The solver expects SI units (metres). The mesher→solver handoff converts world units to metres via `unit_to_meters`, and boundary-face tagging tolerance is now scale-aware (`boundary_spacing`), so metre-scale geometries tag correctly. CAD defaults still assume mm.
- **Per-line BC values** (`u_val`, `v_val`, `p_val` on `Line`) are declared but not yet used by the solver.

## Technical Background

This project implements the **finite-volume method** for the incompressible Navier-Stokes equations on arbitrary 2D unstructured meshes. The core algorithm is the **SIMPLE** (Semi-Implicit Method for Pressure-Linked Equations) algorithm by Patankar and Spalding, with Rhie-Chow interpolation to prevent pressure checkerboarding on collocated grids.

For detailed internal documentation, see [`CODEBASE_REFERENCE.md`](./CODEBASE_REFERENCE.md).

## License

MIT