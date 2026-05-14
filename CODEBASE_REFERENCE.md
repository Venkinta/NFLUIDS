# CFD Project — Codebase Reference

## Quick Navigation

| Problem area | Go to |
|---|---|
| Drawing/CAD not responding | `editor.py`, `snapengine.py` |
| Zoom / pan broken | `camera.py` |
| Lines not forming polygon | `mesher.py → build_polygon()` |
| Boundary layer collapsing | `mesher.py → boundary_layer()`, `connect_layers()` |
| Triangulation wrong | `bowyerwatson.py`, `constructor.py` |
| Solver diverging | `solver.py → Solve()`, `initialize_conditions()` |
| BC tags wrong (inlet/outlet/wall misidentified) | `mesher.py → solver_data_pipeline()`, `create_boundary_points()` |
| Units / scale wrong | `camera.py` (scale), `physics_editor.py` (defaults), `mesher.py → solver_data_pipeline()` |
| UI / ImGui broken | whichever module's `draw()` method + `main.py` event handling |
| State machine / screen transitions | `main.py` only |

---

## 1. Architecture Overview

The app is a **linear state machine**. Screens advance one-way:

```
EDITOR  →  PHYSICS  →  MESHER  →  SOLVER
```

Managed entirely in `main.py`. There is no back-navigation. Each state owns its own module instance. The renderer (`PygameRenderer`) is created once in `main.py` and shared across all modules — **never re-create it**.

---

## 2. Module-by-Module Reference

### `main.py` — Orchestrator
**Owns:** state machine, main loop, OpenGL init, ImGui context creation, camera instance, clock.

**Critical details:**
- ImGui context is created here ONCE (`imgui.create_context()`). Modules must NOT call this themselves.
- The renderer is passed into `Editor`, `PhysicsEditor`, `Mesher` constructors. Don't duplicate it.
- Event handling order matters: `renderer.process_event(event)` must come before any `imgui.get_io().want_capture_mouse` check.
- Camera is created here and passed into every `handle_event()` and `draw()` call — it is the single source of truth for zoom/pan state.
- The fixed-update loop (`accumulator`) is stubbed out and does nothing currently.
- Profiling (`cProfile`) wraps the entire `run_app()` — results print on exit.

**What NOT to touch here:** OpenGL init sequence (`glOrtho`, blend mode). Changing it will break ImGui rendering.

---

### `camera.py` — Coordinate System
**Owns:** world↔screen conversion, all OpenGL drawing primitives.

**The coordinate transform:**
```
screen_x = (world_x + offset[0]) * scale
world_x  = (screen_x / scale) - offset[0]
```
`scale` = pixels per world-unit. `offset` = world-space pan offset.

**Drawing methods (all OpenGL, no Pygame surface):**
- `draw_line(screen, line, ...)` — takes a `Line` object, converts internally.
- `draw_screen_line(screen, p0, p1, ...)` — raw pixel coords, used for previews.
- `draw_polygon(polygon_vertices, screen, ...)` — takes list of `Point` objects.
- `draw_circle(screen, color, center_screen, radius, ...)` — screen coords only.

**Critical:** `screen` parameter is passed everywhere but is essentially unused by the OpenGL calls — it's a legacy argument from when this used Pygame surfaces. Don't remove it; it's threaded through dozens of call sites.

---

### `point.py` — Base Geometry Primitive
**Used everywhere.** Supports: `__eq__`, `__hash__`, `__add__`, `__sub__`, `distance_to()`, `to_tuple()`.

**Critical:** `__hash__` is `hash((self.x, self.y))` — floats. Two points at the same coordinates ARE the same point for set/dict purposes. This is intentional and relied upon by Bowyer-Watson deduplication.

**Watch out:** `__eq__` uses exact float equality (`==`). This is intentional for vertex snapping, but can cause missed matches if coordinates drift by floating-point error. If you ever see "lines don't form a closed loop", check whether `build_polygon()` is comparing Points with `==` via numpy — it uses `np.array_equal(pivot, line.a)` which compares the *object* not coordinates.

---

### `line.py` — CAD Edge / Boundary Segment
Stores two `Point` endpoints (`a`, `b`), plus physics metadata:
- `boundary_type`: `"Wall"` | `"Velocity Inlet"` | `"Pressure Outlet"` (string, set in PhysicsEditor)
- `u_val`, `v_val`, `p_val`: not used by the solver yet (solver reads `inlet_velocity` from PhysicsEditor directly).

`is_mouse_over()` uses projected distance — threshold is in *world units* (5 units). With the current pixel-scale default this is 5 pixels. With mm scale this becomes 5mm — probably fine, but watch it.

`vector` property: returns `[dx, dy]` as a plain list, not a Point or numpy array.

---

### `editor.py` — CAD Module
**State:** `is_drawing` (bool), `start_pos` (Point or None), `lines` (list of Line).

**Click flow:**
1. Screen px → `camera.screen_to_world()` → world Point
2. World Point → `snap_engine.get_snapped_pos()` → snapped world Point
3. First click: store as `start_pos`. Second click: create `Line(start_pos, snapped_pos)`, chain.

**Escape key:** If drawing → cancel current segment. If not drawing → undo last line (pop).

**ImGui:** Renders a "Finish CAD" button. `finished` flag triggers state transition in `main.py`. The tooltip (length/dx/dy) is rendered as a floating ImGui window that follows the cursor.

---

### `snapengine.py` — Snap Logic
Two snap modes:
1. **Vertex snap** — checks all line endpoints within `pixel_radius / camera.scale` world units.
2. **Axis snap** — if cursor is within `world_radius` of `anchor_pos` on X or Y axis, locks that axis.

Returns a `Point`. The returned Point may be a reference to an *existing* endpoint — this is important for polygon closure.

---

### `physics_editor.py` — BC & Mesh Parameter Configurator
**Sits between EDITOR and MESHER.** Has no mesh logic itself; purely UI + data.

**Outputs (passed to Mesher constructor):**
| Parameter | Default | Meaning |
|---|---|---|
| `n_layers` | 4 | Boundary layer count |
| `growth_factor` | 1.4 | Layer thickness multiplier |
| `thickness` | 4 | First layer thickness (world units) |
| `boundary_spacing` | 35 | Arc length between boundary points (world units) |
| `r` | 20 | Steiner point minimum separation (world units) |
| `inlet_velocity` | 1 | m/s (SI, not world units) |
| `outlet_pressure` | 0 | Pa (SI) |
| `density` | 1.2 | kg/m³ (SI) |
| `viscosity` | 0.002 | Pa·s (SI) |

**BC assignment:** Clicking a line (via `handle_selection`) opens a per-line ImGui window. The `boundary_types` list order matters — index maps to the combo box index AND to the `bc_map` in Mesher.

**Important:** `inlet_velocity` is a single float here, but the Solver receives `[inlet_velocity, 0.0]` — a 2D vector. This is assembled in `main.py`.

---

### `mesher.py` — Mesh Generation Engine
The most complex module. Four main phases:

#### Phase 1: Boundary Points (`create_boundary_points`)
- Orders lines via `build_polygon()` into a consistent loop.
- Samples each edge uniformly at `boundary_spacing` intervals → `self.boundary_points` (Nx2 numpy array).
- Simultaneously builds `self.thickness_mask` (which points are on walls → get BL offsets) and `self.point_bc_mask` (integer BC tags: 0=Wall, 1=Inlet, 2=Outlet).
- `bc_map` must match the string values in `physics_editor.py → boundary_types` exactly.

#### Phase 2: Boundary Layers (`boundary_layer`, `connect_layers`)
- `boundary_layer()` offsets a ring of points inward using miter vectors. Handles CCW/CW correctly via `self.orientation`.
- `connect_layers()` stitches adjacent rings into Quads, degenerating to Triangles when an edge collapses (pinched corners).
- Output: `self.boundary_elements` (list of Quad and Triangle objects).

**Watch out:** `polygon_orientation()` returns `area_signed`. If positive → CW → normals flipped. If the boundary layer grows outward instead of inward, check this sign and the normal formula in `boundary_layer()`.

#### Phase 3: Steiner Points (`create_steiner_points`)
- Poisson-disk sampling using Bridson's algorithm with a spatial grid.
- Uses Shapely for the `safe_zone` (polygon buffered inward by `r * 0.8`).
- Grid cell size `w = r / sqrt(2)`.
- Hard cap at 500,000 grid cells — if triggered, `self.points` is empty.

#### Phase 4: Triangulation & Filter
- Calls `Bowyer_watson(all_interior_pts)`.
- `filter_triangles()` removes triangles outside the inner ring using `matplotlib.path.Path.contains_points`.

#### `solver_data_pipeline()`
The final output generator. Returns the mesh dict the Solver expects. Key steps:
1. Builds `bc_lookup` (edge → BC tag) from boundary points.
2. Merges boundary elements + triangles into `Cells`.
3. Builds `edge_map` (edge key → list of cell IDs). An edge with 1 cell is a boundary face.
4. Populates `owner`, `neighbor`, `Sf`, `Cf`, `df`, `magDf`, `boundary_tags`.
5. Boundary face tagging: nearest boundary segment search (within 1.0 world unit tolerance).

**Critical implementation note:** Edge keys are `(round(x,6), round(y,6))` sorted tuples. This rounding is essential — without it, floating-point jitter creates duplicate edges. Don't change the rounding precision without testing.

**The 1.0 tolerance in boundary tagging:** `if min_dist < 1.0` — this is in world units. If geometry is in mm (small numbers), this is 1mm, which is fine. If geometry were in metres, you'd need to scale this.

---

### `bowyerwatson.py` — Delaunay Triangulation
Standard Bowyer-Watson. Inputs: list of `Point` objects. Outputs: `Triangulation`.

**Deduplication:** `list(set(input_points))` — relies on `Point.__hash__`. Important: if two points have the same coordinates, only one survives. This is intentional (prevents degenerate triangles).

**Super-triangle cleanup:** Vertices of the super-triangle are kept in `super_verts` as a set and compared with triangle vertices. Since Point hash is coordinate-based, as long as no input point coincidentally matches a super-triangle vertex (very unlikely with the 20×dmax scale), this is safe.

---

### `constructor.py` — Geometry Helpers
- `create_super_triangle`: scale factor 20 on `dmax`. Safe for normal geometries; could theoretically cause precision issues if input points span many orders of magnitude.
- `checkCircumcentre`: standard determinant incircle test. Calls `orientCCW` first for consistency.
- `orientCCW`: mutates the triangle in-place by swapping b/c if CW. **Side effect** — be aware when debugging triangle winding.
- `intersect`: line-line intersection via Cramér's rule. Used by mesher (imported but check where).

---

### `triangle.py` / `quad.py` — Cell Geometry
Both implement the same interface: `vertices()`, `edges()`, `centroid` (property), `area` (property), `draw()`.

`edges()` returns `frozenset` pairs — this is what Bowyer-Watson and `edge_map` use as keys. **The frozenset hashing is what makes edge matching work.**

`Triangle.area`: cross-product formula, absolute value — always positive.
`Quad.area`: Shoelace formula, absolute value.
`Quad.centroid`: polygon centroid formula (not simple average) — important for irregular quads.

---

### `triangulation.py` — Triangle Container
Thin wrapper over a list. `remove_triangle` uses `list.remove()` which relies on object identity, not coordinates. Safe because the same object is added and removed within Bowyer-Watson.

---

## 3. Complete Data Flow

```
User draws lines in Editor
        │  List[Line] (world coords, unit = px currently)
        ▼
PhysicsEditor assigns BC strings + mesh params
        │  List[Line] (same), floats for n_layers/thickness/spacing/r
        ▼
Mesher.mesh()
  ├── build_polygon() → ordered List[Line]
  ├── create_boundary_points() → np.array (N,2), thickness_mask, point_bc_mask
  ├── boundary_layer() × n_layers → list of np.arrays (rings)
  ├── connect_layers() → List[Quad|Triangle]  ← boundary_elements
  ├── create_steiner_points() → np.array (M,2)
  ├── Bowyer_watson() → Triangulation
  └── filter_triangles() → cleaned Triangulation
        │
        ▼
Mesher.solver_data_pipeline()
  Returns dict: {Nc, Nf, owner, neighbor, Sf, magSf, Cf, df, magDf,
                 cell_centers, cell_areas, boundary_tags}
        │  All distances/coords in WORLD UNITS (currently px, should be m)
        ▼
Solver.__init__() → unpacks dict
Solver.Solve() → SIMPLE iterations
  ├── SIMPLE_UPDATE_FACE_FLUX_AND_DIFFUSSION()
  ├── assemble_momentum(axis=0), assemble_momentum(axis=1)
  ├── GET_VAR_STAR() → u*, v*
  ├── ASSEMBLE_PRESSURE_CORRECTION()
  ├── GET_VAR_CORRECTED() → p'
  └── CORRECT_PRESSURE_AND_VELOCITY()
```

---

## 4. The Units Problem (Current State)

### Root Cause
World coordinates were historically pixels. Camera started with `scale=1.0` meaning 1 world unit = 1 screen pixel. The editor never had a physical unit concept.

### Consequence
A 640-pixel-wide drawing = 640 "world units" fed to the solver as 640 metres. At those scales, Reynolds numbers are enormous, Kolmogorov scales are microscopic, and the solver is solving a physically absurd problem.

### Where Units Live
| Module | What's in world units | Notes |
|---|---|---|
| `camera.py` | `scale` (px/world-unit), `offset` | Transform only — no physics |
| `physics_editor.py` | `thickness`, `boundary_spacing`, `r` | Defaults assume old pixel scale |
| `mesher.py` | `boundary_points`, all coords, `Sf`, `Cf`, `df`, `magDf`, `cell_areas` | Pipeline output in world units |
| `solver.py` | Everything in `mesher_data` dict | Expects SI (metres) |

### What Does NOT Need Unit Conversion
- `inlet_velocity` (m/s, already SI)
- `outlet_pressure` (Pa, already SI)
- `density`, `viscosity` (SI)
- `n_layers`, `growth_factor` (dimensionless)

---

## 5. Things You Must NOT Break

1. **`renderer` singleton** — created once in `main.py`, passed everywhere. Never instantiate `PygameRenderer` in a module.
2. **Edge key rounding** — the `round(..., 6)` in `get_edge_key()` inside `solver_data_pipeline()`. Change precision and the edge map breaks silently.
3. **`frozenset` edges** — Triangle and Quad both use frozenset for edges. Changing to tuples breaks all dict lookups.
4. **`orientCCW` mutation** — `constructor.orientCCW()` mutates the triangle. The Bowyer-Watson loop calls `checkCircumcentre` which calls `orientCCW`. Don't assume triangle vertex order is stable after this.
5. **`polygon_orientation` sign convention** — Positive = CW in the shoelace convention used here (note: this is *opposite* to the standard mathematical convention where positive area = CCW). The `boundary_layer` normal-flip depends on this.
6. **`build_polygon` comparison** — Uses `np.array_equal(pivot, line.a)` where `pivot` starts as `line.b` (a `Point`). This compares object identity through numpy, which works because line endpoints are shared Point objects from snap. If you ever regenerate Point objects from coordinates, this will break — use `Point.__eq__` instead.
7. **`bc_map` string matching** — The strings in `bc_map` in `create_boundary_points()` must exactly match the `boundary_types` list in `physics_editor.py`.

---

## 6. Known Issues / Technical Debt

- `line.py`: `u_val`, `v_val`, `p_val` are unused (future per-line BC values).
- `solver.py → health_check()` prints every iteration — verbose, should be gated.
- `mesher.py → check_points()` method exists but is never called.
- `data_structures.txt` is partially outdated — `magSf` was added later and is in the actual pipeline but not the txt.
- `constructor.py → intersect()` is imported but not called in the current mesher flow.
- `main.py`: fixed-update accumulator does nothing (`while accumulator >= dt: accumulator -= dt` with no body).
- The 1.0 world-unit boundary tagging tolerance in `solver_data_pipeline()` is implicitly scale-dependent.
