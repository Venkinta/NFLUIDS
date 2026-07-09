# CFD TextEditor — Interactive 2D Navier-Stokes Solver

A complete **2D Computational Fluid Dynamics (CFD) pipeline** with an interactive CAD frontend, built entirely in Python. Draw your geometry, assign boundary conditions, generate a hybrid mesh, solve the incompressible Navier-Stokes equations using the SIMPLE algorithm, and visualize results — all in one application.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

**Requires:** Python 3.10+, a GPU with OpenGL support.

## Workflow

```
CAD Drawing  →  Assign BCs  →  Generate Mesh  →  Solve  →  Visualize
   (EDITOR)      (PHYSICS)       (MESHER)       (SOLVER)   (VISUALIZER)
```

### 1. Draw Your Geometry (EDITOR)
- **Left-click** to place points and draw lines (chain-drawing mode).
- **Escape** to cancel the current segment or undo the last line.
- **Mouse wheel** to zoom in/out; **middle-mouse drag** to pan.
- **Controls panel:**
  - **Units** — Choose mm, cm, or m for your drawing.
  - **Ortho (90° Snap)** — Constrain lines to horizontal/vertical.
  - **Exact Length** — Set a fixed line length (set to 0 for free drawing).
- Click **"Finish CAD"** to proceed.

### 2. Assign Boundary Conditions (PHYSICS)
- **Click** any line to select it and open its properties panel.
- Assign each line a **boundary type**: `Wall`, `Velocity Inlet`, or `Pressure Outlet`.
- Configure **fluid properties** (density, viscosity).
- Configure **mesh parameters** (boundary layer settings, mesh size).
- The **Reynolds number** estimator gives a DNS cell count prediction.
- Click **"Mesh"** to generate the computational grid.

### 3. Solve (SOLVER)
The solver runs automatically after meshing. It uses the **SIMPLE algorithm** with:
- Rhie-Chow interpolation for pressure-velocity coupling.
- BiCGSTAB with Jacobi (momentum) and ILU/PyAMG (pressure) preconditioners.
- Convergence is measured by RMS continuity residual (default tolerance: 1e-6).

Progress is printed to the console every 10 iterations.

### 4. Visualize Results (VISUALIZER)
- Switch between **Pressure**, **Velocity**, **Continuity Error**, and **Momentum Error** fields.
- Toggle **velocity vectors** with adjustable scale.
- **Hover** over any cell to probe local values (P, Ux, Uy, residual errors).
- Click **"Return to Editor"** to start a new simulation.

## Features

### CAD Editor
- Vertex and axis snapping for precise geometry
- Orthogonal mode, exact lengths, configurable snap step
- Multiple unit systems (mm, cm, m)
- OpenGL-backed smooth rendering

### Mesh Generation
- **Boundary layers**: Prismatic cells at walls with controlled growth rate
- **Interior mesh**: Poisson-disk Steiner point sampling + Delaunay triangulation (Bowyer-Watson)
- Hybrid mesh with quadrilaterals (boundary) and triangles (interior)
- Numba-accelerated circumcircle checks

### Solver
- Incompressible Navier-Stokes (SIMPLE algorithm)
- Finite-volume discretization on arbitrary polygonal meshes
- Rhie-Chow interpolation to prevent checkerboarding
- Multiple linear solver backends (BiCGSTAB + Jacobi/ILU/PyAMG)
- Numba JIT compilation for hot loops

### Post-Processor
- Color-mapped field visualization (jet-like native colormap)
- Interactive point probing with KD-tree spatial indexing
- Velocity vector glyphs with adjustable scale
- Log-scale residual visualization for error analysis
- OpenGL VBO-based rendering for large meshes

## Dependencies

| Package | Purpose |
|---------|---------|
| `PyOpenGL` | GPU-accelerated rendering |
| `pygame` | Windowing and event handling |
| `imgui` / `pygame_widgets` | UI framework |
| `numpy` | Array computations |
| `scipy` | Sparse linear algebra (BiCGSTAB, ILU) |
| `numba` | JIT compilation for hot loops |
| `shapely` | Polygon geometry (Poisson-disk sampling) |
| `matplotlib` | Path containment tests (triangle filtering) |
| `pyamg` (optional) | Algebraic multigrid preconditioner |

Optional but recommended: `pip install pyamg` for faster pressure solves.

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Application entry point, state machine, main loop |
| `editor.py` | CAD drawing with snapping and constraints |
| `physics_editor.py` | Boundary condition and mesh parameter UI |
| `mesher.py` | Mesh generation pipeline |
| `solver.py` | SIMPLE algorithm (Navier-Stokes solver) |
| `visualizer.py` | Post-processing and field visualization |
| `camera.py` | World↔screen coordinate transforms, drawing primitives |
| `point.py` | 2D point geometry primitive |
| `line.py` | CAD edge with boundary metadata |
| `triangle.py` / `quad.py` | Cell geometry types |
| `triangulation.py` | O(1) add/remove triangle container |
| `bowyerwatson.py` | Delaunay triangulation algorithm |
| `constructor.py` | Numba JIT kernels and geometry helpers |
| `snapengine.py` | Vertex and axis snapping |
| `meshio_importer.py` | External mesh import (placeholder) |
| `CODEBASE_REFERENCE.md` | Internal developer documentation |

## Known Limitations

- **Units migration in progress**: The solver expects SI units (metres). The mesher→solver handoff converts world units to metres via `unit_to_meters`, and boundary-face tagging tolerance is now scale-aware (`boundary_spacing`), so metre-scale geometries tag correctly. CAD defaults still assume mm.
- **Per-line BC values** (`u_val`, `v_val`, `p_val` on `Line`) are declared but not yet used by the solver.
- **No unit tests** — currently relies on visual verification and console output.
- **Mesh import** (`meshio_importer.py`) is an empty placeholder.
- **Solver runs synchronously** in the SOLVER state and blocks the UI until convergence (no progress rendering mid-solve).

## Technical Background

This project implements the **finite-volume method** for the incompressible Navier-Stokes equations on arbitrary 2D unstructured meshes. The core algorithm is the **SIMPLE** (Semi-Implicit Method for Pressure-Linked Equations) algorithm by Patankar and Spalding, with Rhie-Chow interpolation to prevent pressure checkerboarding on collocated grids.

For detailed internal documentation, see [`CODEBASE_REFERENCE.md`](./CODEBASE_REFERENCE.md).

## License

MIT