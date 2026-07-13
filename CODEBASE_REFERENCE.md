# CFD Project — Codebase Reference

## Quick Navigation

| Problem area | Go to |
|---|---|
| Drawing/CAD not responding | `editor.py`, `snapengine.py` |
| Zoom / pan broken | `camera.py` |
| Lines not forming polygon | `mesher.py → build_polygon()` |
| Boundary layer collapsing | `mesher.py → boundary_layer()`, `connect_layers()` |
| Triangulation wrong | `bowyerwatson.py`, `constructor.py` |
| Solver diverging | `solver.py → step()`, `initialize_conditions()`, `solver_protocol.py` |
| BC tags wrong (inlet/outlet/wall misidentified) | `mesher.py → solver_data_pipeline()`, `create_boundary_points()` |
| Units / scale wrong | `camera.py` (scale), `physics_editor.py` (defaults), `mesher.py → solver_data_pipeline()` |
| UI / ImGui broken | whichever module's `draw()` method + `main.py` event handling |
| State machine / screen transitions | `main.py` only |
| Visualizer not showing results | `visualizer.py`, `main.py` SOLVING→VISUALIZER transition |
| Solve monitor frozen / plots not updating / crashing | `solver_panel.py` — thread loop, queues, `_drain_queues()` |

---

## 1. Architecture Overview

The app is a **state machine** with a forward pipeline and one loop-back:

```
EDITOR  →  PHYSICS  →  MESHER  →  SOLVING  →  VISUALIZER
              ▲                                    │
              └────────────────────────────────────┘
```

Managed entirely in `main.py` via a plain string `current_state` (`"EDITOR"` / `"PHYSICS"` / `"SOLVING"` / `"VISUALIZER"` — not a formal enum). `SOLVER`/`SOLVER_LOADED` no longer exist as separate states; both the live-mesh and loaded-`.npz` paths converge into the single `SOLVING` state. There is no VISUALIZER→EDITOR transition in the code — clicking "Back to Physics" in the Visualizer always loops back to PHYSICS (state preserved — mesh, BCs, VBOs, mesher, and physicseditor all survive), so the user can tweak solver settings and re-solve without rebuilding anything. The only way back to EDITOR is starting a fresh session. Each state owns its own module instance. The renderer (`PygameRenderer`) is created once in `main.py` and shared across all modules — **never re-create it**.

---

## 2. Module-by-Module Reference

### `main.py` — Orchestrator
**Owns:** state machine, main loop, OpenGL init, ImGui context creation, camera instance, clock.

**Critical details:**
- ImGui context is created here ONCE (`imgui.create_context()`). Modules must NOT call this themselves.
- The renderer is passed into `Editor`, `PhysicsEditor`, `Mesher` constructors. Don't duplicate it.
- Event handling order matters: for every event, the active state's `renderer.process_event(event)` is called **first**, then `imgui.get_io().want_capture_mouse` is read once and used to gate both the global mouse-wheel camera zoom and click-based module handlers. Getting this order backwards (checking `want_capture_mouse` before feeding the event to ImGui) is what caused the old bug where scrolling over an ImGui panel also zoomed the camera underneath — the wheel handler used to run unconditionally, ahead of and independent from the per-state `process_event()` calls.
- Camera is created here and passed into every `handle_event()` and `draw()` call — it is the single source of truth for zoom/pan state.
- The fixed-update loop (`accumulator`) is stubbed out and does nothing currently.
- Profiling (`cProfile`) wraps the entire `run_app()` — results print on exit.
- On `PHYSICS → SOLVING`, both the live-mesh path (`mesher.solver_data_pipeline()`) and the loaded-`.npz` path (dividing `cell_vertices`/`cell_centers` by `unit_to_meters`) build their `vis_mesher` before constructing `Solver(...)` and `SolverPanel(...)` — there is one construction site shared by both paths, not two.
- **Live solve preview:** a `Visualizer` instance (`live_field`) is constructed right alongside `SolverPanel`, seeded with zero `P`/`U`/residual arrays. Every frame in `SOLVING`, if `solver_panel.viz_snapshot` is non-`None` it's consumed (set back to `None`) and pushed into `live_field` via `update_fields()` + `update_vbo_colors()`, then `live_field.draw_geometry(camera)` paints the colored mesh before `solver_panel.draw(screen, camera, live_field=live_field)` draws the monitor panel (plus a "Show" combo reusing `live_field.vars`/`var_idx`) on top. `Solver.field_snapshot` now includes `res_cont`/`res_mom` (from `self._live_res_cont`/`_live_res_mom`, refreshed every `step()`) alongside `P`/`U`, so all four Visualizer variables — Pressure, Velocity, Continuity Error, Momentum Error — are switchable live, not just Pressure. On `SOLVING → VISUALIZER`, `live_field` is refreshed with the final fields and promoted to `visualizer` directly — no second `Visualizer`/VBO set is ever allocated for the same solve.
- On `VISUALIZER → PHYSICS`, `visualizer.destroy()` frees that cycle's `pos_vbo`/`color_vbo`/`vector_vbo` (a fresh `live_field` — and its own new VBOs — gets built on the next Solve). The wireframe `vbos` dict and `mesher`/`physicseditor` are untouched, so the "Solve" button is immediately available for a re-run with tweaked BCs/solver settings. Full teardown (`glDeleteBuffers` on `vbos`, resetting `mesher`/`physicseditor`) only happens on an explicit fresh mesh or load request.

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
| `growth_factor` | 1.1 | Layer thickness multiplier |
| `thickness` | 1.0 | First layer thickness (world units) |
| `boundary_spacing` | 6.0 | Arc length between boundary points (world units) |
 | `r` | 4.0 | Steiner point minimum separation (world units) |
| `inlet_velocity` | 1 | m/s (SI, not world units) |
| `outlet_pressure` | 0 | Pa (SI) |
| `density` | 1.2 | kg/m³ (SI) |
| `viscosity` | 0.002 | Pa·s (SI) |

**Refinement zones:** `refinement_zones` is a list of dicts `{ 'rect': (x1, y1, x2, y2), 'factor': float }`. Each zone is a rectangle drawn by the user (click-drag on canvas). The `factor` divides the global `r` to get the local Steiner spacing: `local_r = r / factor`. The UI shows the resulting mesh size (`r / factor`) next to each zone. Zones are converted to `(shapely_polygon, factor)` tuples via `_get_refinement_polygons()` and passed to the Mesher. **Zones persist across mesh save/load** — they are serialized as polygon exterior coords + factor in the `.npz` file and restored into `physicseditor.refinement_zones` on load.

**BC assignment:** Clicking a line (via `handle_selection`) opens a per-line ImGui window. The `boundary_types` list order matters — index maps to the combo box index AND to the `bc_map` in Mesher.

**Important:** `inlet_velocity` is a single float here, but the Solver receives `[inlet_velocity, 0.0]` — a 2D vector. This is assembled in `main.py`.

**Solver Settings:** A collapsing header between Refinement Zones and the action buttons. `alpha_u` and `alpha_p` are sliders (continuous feedback), `max_iterations` is a plain int input, `tolerance` is a log10 integer slider (`3`-`10`) that shows the resolved value inline (e.g. `= 1e-04`) so 1e-4 vs 1e-6 is never ambiguous, and `viz_interval` is editable so the user can trade off live-update frequency against GPU overhead on large meshes. These five fields are passed straight into `Solver.__init__()` / `SolverPanel.__init__()` when "Solve" is clicked.

**Window sizing:** The main "Mesher Settings" window uses `imgui.set_next_window_size_constraints((0, 0), (480, 0.9 * display_height))` plus `WINDOW_ALWAYS_AUTO_RESIZE`, so it grows to fit content (BC list, refinement zones, solver settings) but caps out at 90% of the screen height and gets its own internal scrollbar beyond that, instead of growing off-screen.

---

### `solver_protocol.py` — the Solver ABC

Defines `SolverProtocol(ABC)` with four abstract methods: `initialize_conditions()`, `step(**state)`, `finalize(**final_state)`, and `field_snapshot` (a property). The `**state`/`**final_state` convention means a concrete solver owns its own internal keys entirely — `SolverPanel` never inspects them. The only keys that are part of the **public contract** are `'residuals'` (dict of named floats) and `'converged'` (bool) in the dict returned by `step()`; everything else round-trips opaquely between `SolverPanel` and the concrete solver. `step()` returning `None` signals fatal divergence. `field_snapshot` must return copies (not views) of the current field arrays, at minimum `{'U': ndarray(Nc,2), 'P': ndarray(Nc,)}`.

**Why it exists:** a future LES solver (or anything else) just inherits this ABC, implements the four methods, and the rest of the app — `SolverPanel`, `main.py`, `Visualizer` — works with zero changes.

---

### `solver.py` — SIMPLE Algorithm Implementation

Implements `SolverProtocol`. Three surgical changes from the previous synchronous version:
- `__init__` now accepts `alpha_u`, `alpha_p`, `max_iterations`, `tolerance` as parameters (defaults preserved: `0.3`, `0.2`, `1600`, `1e-6`).
- `Solve()` is now a thin ~20-line wrapper around `step()`, kept purely for backward compatibility with any caller that wants a blocking, non-threaded solve.
- The old loop body became `step(**state)`. It returns the opaque solver state dict (`a_P_u`, `a_P_v`, `initial_cont_rms`, ...) plus the protocol-required `'residuals'`/`'converged'` keys, plus underscore-prefixed raw arrays (`'_b_p'`, `'_r_u'`, `'_r_v'`) that `finalize()` picks up. `finalize(**final_state)` extracts those into `self.final_res_cont` / `self.final_res_mom` (cell-level arrays consumed by `Visualizer`) — the leading underscore avoids colliding with the solver's own state keys. `step()` also refreshes `self._live_res_cont`/`self._live_res_mom` (same `abs(b_p)` / `sqrt(r_u²+r_v²)` formulas as `finalize()`, just computed every iteration instead of once at the end) so the live preview can show Continuity/Momentum Error mid-solve, not just Pressure/Velocity. `field_snapshot` returns `.copy()`'d `U`/`P`/`res_cont`/`res_mom` on every call so the solver thread can't corrupt a snapshot the main thread is still reading.

---

### `solver_panel.py` — Threaded Solve Orchestration + Monitor UI

Wraps a `SolverProtocol` instance and runs it on a background thread so the UI never blocks during a solve.

- The solver thread runs `step()` in a loop. It checks `stop_event` at the top of every iteration, plus a combined pause/run-to check *before* running the step — so "paused at iteration N" always means step N has **not** run yet, making a `step_one()` control unambiguous.
- Residuals are pushed onto an unbounded queue — no data point is ever lost, even if the main thread falls behind for a frame.
- The viz (field) snapshot queue has `maxsize=1` and drops stale snapshots, so the main thread always sees the *latest* state, never a growing backlog. `solver_panel.py` only stores the drained snapshot on `self.viz_snapshot`; it does not render it — `main.py`'s `SOLVING` draw block consumes it into the `live_field` `Visualizer` (see the `main.py` section above).
- `_drain_queues()` (called once per frame from `draw()`, main thread only) rebuilds the log10 float32 arrays consumed by `imgui.plot_lines()` for the residual plots. **Don't pass `scale_min`/`scale_max` as `float('nan')`** to force autoscale — ImGui's autoscale sentinel is `FLT_MAX` (`imgui.FLOAT_MAX`), and `NaN != FLT_MAX`, so a NaN "sentinel" is used as a literal (degenerate) axis range and the line never renders. Simplest fix: omit `scale_min`/`scale_max` entirely and let the defaults (`FLOAT_MAX`) trigger autoscale.
- `resume()` sets `self.state = "RUNNING"` in addition to clearing `_pause_event` — this is required for the Pause/Resume button and status badge to flip correctly. Without it, `self.state` only ever leaves `"PAUSED"` when the thread reports a terminal message (`converged`/`max_iters`/`diverged`), so Resume looked broken (badge stuck on "PAUSED", button never reverted to "Pause") even though the thread had genuinely resumed stepping.
- The `'done'` message handler moves state to `"DONE"` from either `"RUNNING"` **or** `"PAUSED"` (not just `"RUNNING"`) — otherwise clicking Stop while paused left `self.state` stuck at `"PAUSED"` forever after the thread actually exited.
- Thread finalization (`finalize()`) runs exactly once regardless of whether the loop exited via convergence, hitting `max_iterations`, or a user-initiated stop.
- **By design**, once `self.state` reaches `"DONE"`/`"DIVERGED"` the panel only offers "Open Visualizer" — Stop/Pause/+1 Step/Run-to-iter intentionally disappear rather than being shown as inert no-ops. The thread has already exited at that point, and making those controls *actually* resume iterating would require persisting `last_result`/`iteration` across thread restarts and suppressing the immediate re-trigger of the convergence check (`converged = iteration > 50 and res_cont_rms < tolerance` fires again on the very next step once you're already under tolerance). That's a real change to the solve-loop's control flow, not a UI tweak — deliberately left alone.

---

### `visualizer.py` — Post-Processing / Field Visualization

Renders the colored mesh (fill) plus, in the full post-solve UI, the "Post-Processor" control window and point-probe. Used two ways: as the final `visualizer` after a solve, and as the `live_field` preview during `SOLVING` (see the `main.py` section above) — same class, same VBOs, two call patterns:
- `draw(screen, camera)` — full experience: recolors on variable change, draws geometry + vectors, then the ImGui "Post-Processor" window and probe tooltip. Used only in the `VISUALIZER` state.
- `draw_geometry(camera)` — just the colored-mesh GL draw (position + color VBO), no ImGui at all. Used by `main.py` during `SOLVING` so the live preview doesn't pop up its own window on top of the Solver Monitor.
- `update_fields(P, U, res_cont=None, res_mom=None)` — swaps in new field data without touching geometry; caller must follow with `update_vbo_colors()` to push new colors to the GPU (`draw()`'s own auto-recolor only fires on a variable-combo change, not on new data — the live preview needs to recolor on *every* snapshot regardless of which variable is selected).
- `destroy()` — frees `pos_vbo`/`color_vbo`/`vector_vbo`. Call before dropping the last reference to a `Visualizer`/`live_field` (`main.py` calls this on `VISUALIZER → PHYSICS`) — GL buffers aren't reference-counted, so skipping this leaks GPU memory across repeated re-solves.
- `update_vbo_colors()` — maps `var_idx` (0=Pressure, 1=Velocity, 2=Continuity Error, 3=Momentum Error) to vertex colors. **Uses a robust, percentile-clipped range (2nd–98th percentile via `np.nanpercentile`), not true min/max.** A single outlier cell (classically a leading-edge stagnation point) sitting at the true extreme would otherwise single-handedly set the color scale for the entire field, so tiny genuine changes there make everything else look like it's oscillating wildly frame to frame — this is the fix for that. The "Range" readout in the Post-Processor window still shows the true (unclipped) min/max, which remains accurate; only the color mapping is clipped.

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
- Uses Shapely for the `safe_zone` (polygon buffered inward by `min_r * 0.8`, where `min_r` is the smallest spacing across all refinement zones — this lets Steiner points sit closer to prismatic layers when refinement is active).
- Grid cell size `w = min_r / sqrt(2)` (sized for the densest zone).
- **Refinement zones**: If `self.refinement_zones` is non-empty, a single unified Poisson-disk pass fills all zones + background simultaneously. Each zone gets a guaranteed seed at its centroid (so overlapping/disconnected zone arms are all populated). The spacing at each candidate is determined by `_get_local_r()` which uses signed-distance blending (smoothstep over a `5 * r` buffer) to transition smoothly from `r/factor` inside a zone to `r` outside — avoiding sudden cell-size jumps that cause solver artefacts.
- Hard cap at 2000×2000 grid cells — if triggered, `w` is rescaled to fit.

#### Phase 4: Triangulation & Filter
- Calls `Bowyer_watson(all_interior_pts)`.
- `filter_triangles()` removes triangles outside the inner ring using `matplotlib.path.Path.contains_points`.

#### Distance-Weighted Interpolation (`_gx_int`)
- Computed once in `_precompute_topology()` (lines 202-214) as `self._gx_int = d_Pf / d_PN`, where `d_Pf` is the distance from the owner cell center to the shared face midpoint and `d_PN` is the total owner→neighbor center distance (`magDf`).
- Replaces the implicit **0.5 arithmetic mean** (`0.5 * (U_own + U_nei)`) with proper distance-weighted interpolation: `U_interp = (1 - g_x) * U_own + g_x * U_nei`.
- Applied consistently to: velocity interpolation at faces (`U_interp` in `_compute_rhie_chow_flux` and `SIMPLE_UPDATE_FACE_FLUX_AND_DIFFUSSION`), pressure gradient interpolation (`gP_f`), coupling coefficient `a_P_f` (the `d = Sf²/a_P` term), cell volume/area interpolation (`vol_f`), and pressure face value `p_face` in the momentum RHS.
- **Why it matters:** On a uniform mesh `g_x = 0.5` exactly, so behavior is identical to before. On a refined/non-uniform mesh (small refined cell next to a large coarse cell), the face midpoint is much closer to the small cell's center — the 0.5 average biased interpolation toward the wrong cell, creating artificial pressure gradients that made refinement zones behave like solid bodies (numerical "resistance"). The distance-weighted form is the textbook-correct approach for non-uniform/skewed unstructured meshes and is what makes the v1.2.0 refinement zones numerically sound.
- Precomputed once (mesh topology is fixed during the solve) — zero per-iteration cost.

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
O(1) add/remove container backed by a pre-allocated NumPy array. `remove_triangle` uses a `_tri_to_idx` dict (keyed by `id(triangle)`) for O(1) lookup, then swaps the vacated slot with the last row (swap-with-last) so the array stays compact. `coords` property returns a zero-copy view of the live rows — no list conversion, no `np.asarray` copy.

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
Solver.__init__() → unpacks dict, stores alpha_u/alpha_p/max_iterations/tolerance
Solver.initialize_conditions() → sets P, U, phi, diff; primes face flux/diffusion
SolverPanel drives step() in a loop on a background thread:
Solver.step(**state) → one SIMPLE iteration
  ├── SIMPLE_UPDATE_FACE_FLUX_AND_DIFFUSSION()
  ├── assemble_momentum(axis=0), assemble_momentum(axis=1)
  ├── GET_VAR_STAR() → u*, v*
  ├── ASSEMBLE_PRESSURE_CORRECTION()
  ├── GET_VAR_CORRECTED() → p'
  ├── CORRECT_PRESSURE_AND_VELOCITY()
  └── returns {..solver state.., 'residuals': {...}, 'converged': bool}
Solver.finalize(**final_state) → final_res_cont / final_res_mom for Visualizer
```
`Solve()` still exists as a thin wrapper that loops over `step()` for callers that want a blocking, non-threaded solve.

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
6. **`build_polygon` comparison** — Uses `pivot == line.a` (i.e. `Point.__eq__`). `Point.__eq__` now uses a tight tolerance (`math.isclose`, abs_tol=1e-9) so it is robust to small coordinate drift while still treating distinct vertices as distinct. **`Point.__hash__` is intentionally left as the exact coordinate hash** — Bowyer-Watson dedup (`set()` on Points) relies on bit-identical coordinates, so do NOT make `__hash__` tolerance-based.
7. **`bc_map` string matching** — The strings in `bc_map` in `create_boundary_points()` must exactly match the `boundary_types` list in `physics_editor.py`.
8. **Distance-weighted interpolation (`_gx_int`)** — The solver uses `self._gx_int = d_Pf / d_PN` (distance from owner cell center to face midpoint, over total owner→neighbor distance) for all face interpolations (velocity, pressure gradient, coupling coefficient `a_P_f`, cell volume, `p_face`). Do NOT replace this with a fixed `0.5` arithmetic mean — on non-uniform/refined meshes the 0.5 average biases interpolation toward the wrong cell, creating artificial pressure gradients that make refinement zones behave like solid bodies. On uniform meshes `g_x = 0.5` exactly, so the distance-weighted form is strictly a superset.

---

## 6. Known Issues / Technical Debt

- `line.py`: `u_val`, `v_val`, `p_val` are unused (future per-line BC values).
- `solver.py → health_check()` prints every iteration — verbose, should be gated.
- `data_structures.txt` is partially outdated — `magSf` was added later and is in the actual pipeline but not the txt.

### Resolved technical debt (this pass)
- **Steiner grid OOB crash** — `get_grid_coords` in `create_steiner_points` now clamps indices to `[0, cols-1]`/`[0, rows-1]`, fixing an `IndexError` when a candidate landed exactly on the polygon bounds.
- **`create_steiner_points` default `r=550`** aligned to `4.0` to match `physics_editor` (was a latent mismatch).
- **`build_polygon` comparison** switched from `np.array_equal` (object identity through numpy) to `Point.__eq__` (tolerance-based) — see rule 6 above.
- **`Point.__eq__`** now uses `math.isclose` (abs_tol=1e-9) instead of exact float equality; `__hash__` left exact on purpose.
- **Boundary-face tagging tolerance** in `solver_data_pipeline()` is now `self.boundary_spacing` (scale-aware) instead of a hard-coded `1.0` world unit, so metre-scale geometries tag correctly.
- **Visualizer probe centroid** now reuses `cell.centroid` (shoelace for quads) instead of a naive vertex average, matching the solver's geometry.
- **Dead code removed**: `mesher.check_points()`, `mesher.create_boundary_layers()`, `constructor.intersect()`/`cross2d()`, and the empty `while accumulator >= dt` body in `main.py`.
- **Misleading comment** in `editor.py` ("Default to meters") corrected to reflect the actual mm default.
