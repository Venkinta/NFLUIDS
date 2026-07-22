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
| UI / ImGui broken | `renderer.py` (frame lifecycle — the ONLY `imgui.new_frame()`/`imgui.render()` pair) + whichever module's `draw()` builds the panels + `app_state.py` event handlers |
| GL drawing / VBOs / buffer leaks | `renderer.py` (`Renderer` draw helpers, `VboHandle` lifecycle) |
| Persistent overlays (logo stamp, future FPS counter) | `renderer.py → add_overlay()` / `logo_overlay()` |
| State machine / screen transitions | `app_state.py` (`AppState`, `AppContext`, the three dispatch tables) — `main.py` is just the bootstrap + loop |
| Visualizer not showing results | `visualizer.py`, `app_state.py`'s `update_solving()` (SOLVING→VISUALIZER transition) |
| Solve monitor frozen / plots not updating / crashing | `solver_panel.py` — thread loop, queues, `_drain_queues()` |

---

## 1. Architecture Overview

The app is a **state machine** with a forward pipeline and one loop-back:

```
EDITOR  →  PHYSICS  →  MESHER  →  SOLVING  →  VISUALIZER
              ▲                                    │
              └────────────────────────────────────┘
```

**Since v1.8.0**, this is a formal `AppState` enum (`cfdeditor/app_state.py`) — not a bare string — and the machine is driven by three per-state dispatch tables (`EVENT_HANDLERS`, `UPDATE_HANDLERS`, `RENDER_HANDLERS`, all in `app_state.py`) rather than four hand-synced `if/elif current_state == ...` chains in `main.py`. `main.py`'s `run_app()` is now just bootstrap (pygame/OpenGL/imgui init) plus a loop that does three dict lookups per frame: `EVENT_HANDLERS[ctx.state](ctx, event, want_mouse)` per event, `ctx.state = UPDATE_HANDLERS[ctx.state](ctx)` once per frame, then `RENDER_HANDLERS[ctx.state](ctx, dt)`. `SOLVER`/`SOLVER_LOADED` don't exist as separate states; both the live-mesh and loaded-`.npz` paths converge into the single `SOLVING` state. There is no VISUALIZER→EDITOR transition — clicking "Back to Physics" in the Visualizer always loops back to PHYSICS (state preserved — mesh, BCs, VBOs, mesher, and physicseditor all survive), so the user can tweak solver settings and re-solve without rebuilding anything. The only way back to EDITOR is starting a fresh session. Each state owns its own module instance, held on the shared `AppContext` (see the `app_state.py` section below) rather than as bare locals in `run_app()`.

**Since v1.9.0**, all rendering goes through one shared engine: `renderer.py`'s `Renderer` (held as `ctx.gfx`) owns the frame lifecycle (clear → state draw → overlays → one ImGui render → flip), every shared GL draw helper, and buffer lifecycle via `VboHandle`. Modules build ImGui panels and call `gfx.draw_*` helpers, but never touch the frame lifecycle or raw buffer objects. The ImGui backend (`PygameRenderer`) is still created once in `main.py` (held as `gfx.backend`) — **never re-create it**.

---

## 2. Module-by-Module Reference

### `main.py` — Bootstrap + Loop
**Owns:** pygame/OpenGL/imgui init, the `AppContext` construction, and the frame loop. As of v1.8.0 this is deliberately thin — the state machine itself lives in `app_state.py` (next section).

**Critical details:**
- ImGui context is created here ONCE (`imgui.create_context()`). Modules must NOT call this themselves.
- The one `PygameRenderer` (ImGui backend), one `Camera`, and one `Renderer` (`gfx`) are created here; `camera`/`gfx` are stored on `AppContext`. Since v1.9.0 modules no longer receive the backend or the pygame surface at all — everything they need to draw comes through the `gfx` argument to their `draw()` method.
- The frame loop body is `gfx.begin_frame()` → `RENDER_HANDLERS[ctx.state](ctx, dt)` → `gfx.end_frame()` — clear/new_frame and overlay/render/flip live inside the Renderer, not here.
- The `logo_overlay` is registered here once via `gfx.add_overlay(logo_overlay)`.
- `VboHandle.delete_all()` runs after the loop exits, before `pygame.quit()` — the shutdown cleanup for every GL buffer the app ever created.
- Event handling order matters: for every event, `renderer.process_event(event)` is called **first** (unconditionally, for all states — see below), then `imgui.get_io().want_capture_mouse` is read once and used to gate both the global mouse-wheel camera zoom and `EVENT_HANDLERS[ctx.state]`. Getting this order backwards (checking `want_capture_mouse` before feeding the event to ImGui) is what caused the old bug where scrolling over an ImGui panel also zoomed the camera underneath.
- The per-state ImGui-feed used to be a fourth `if/elif` chain keyed on state (one branch per state, each calling `renderer.process_event(event)` on what was always the same shared renderer object) — removed in v1.8.0 as pure duplication with zero behavioral difference.
- Profiling (`cProfile`) wraps the entire `run_app()` — results print on exit.
- `EVENT_HANDLERS` has no entries for `SOLVING`/`VISUALIZER` — those two states get no per-event handling beyond the unconditional ImGui feed and mouse-wheel zoom above, matching pre-v1.8.0 behavior (`.get(ctx.state)` returns `None` and is skipped).

**What NOT to touch here:** OpenGL init sequence (`glOrtho`, blend mode). Changing it will break ImGui rendering.

---

### `renderer.py` — Shared Rendering Engine (`Renderer`, `VboHandle`, since v1.9.0)

**Owns:** the frame lifecycle, all shared GL draw helpers, buffer lifecycle, and the cross-cutting overlay hook. This is the module the rendering-unification refactor produced; before it, five modules each carried their own `imgui.new_frame()`/`imgui.render()` pair and their own raw `glGenBuffers`/`glDrawArrays` boilerplate.

**The frame contract (the one rule that matters):** `run_app()` calls `gfx.begin_frame()` (clear + `imgui.new_frame()`), then the state's render handler, then `gfx.end_frame()` (overlays → `imgui.render()` → backend render → flip). Modules build their ImGui windows inside their `draw()` methods but must NEVER call `imgui.new_frame()`/`imgui.render()` themselves — a second pair anywhere crashes or blanks the UI.

**`Renderer(camera, imgui_backend, screen)`** — holds the app's single `Camera` (so draw calls don't thread it around; modules needing coordinate math use `gfx.camera`). Draw API, by path:
- *Immediate mode (CPU `camera.to_screen()`, screen-space pixels):* `draw_screen_line(p0, p1, color, width)`, `draw_world_line(a, b, color, width)`, `draw_circle(center, radius, color, width)`, `draw_rect(p1, p2, fill_rgba, outline_rgba, outline_width)`. Line/circle colors are 0–255 RGB; `draw_rect` takes 0–1 RGBA (its fills need alpha blending).
- *VBO path (GL modelview matrix via a private context-managed transform):* `draw_vbo(handle, color, mode, point_size)` for position-only buffers (wireframe `GL_LINES`, smoke `GL_POINTS`), `draw_vbo_colored(pos_handle, color_handle)` for the field fill (the app's one two-attribute draw; vertex count always comes from the *position* handle — 2 floats/vertex vs the color handle's 3).
- The two world→screen paths (CPU vs GL matrix) are **deliberately not merged** — doing so risks pixel drift at high zoom for zero user value.

**`VboHandle(components, usage)`** — one GL array buffer with a uniform lifecycle: `upload(float32_array)` (allocates lazily, sets `.count = size // components`), `delete()` (safe to double-call), and a class-level registry backing `VboHandle.delete_all()` at shutdown. All buffers in the app are `VboHandle`s: the wireframe dict (`ctx.vbos`), Visualizer's pos/color/vector, and the smoke particles.

**Overlays:** `add_overlay(fn)` registers a callable run every frame in every state, inside the ImGui frame, after the state's draw. `logo_overlay()` (module-level) is the first user: stamps `NFLUIDS v{__version__}` top-left via `imgui.get_foreground_draw_list().add_text(...)` — above all windows, captures no input. An FPS counter or screenshot watermark is one `add_overlay` away; that hook is the point of the refactor.

---

### `app_state.py` — State Machine (`AppState`, `AppContext`, dispatch tables)

**Owns:** the `AppState` enum, the `AppContext` dataclass, the per-state `EVENT_HANDLERS`/`UPDATE_HANDLERS`/`RENDER_HANDLERS` dicts, and `_apply_loaded_mesh_settings()`. Added in v1.8.0 to replace `main.py`'s four hand-synced `if/elif current_state == ...` chains — this is the module the Quick Navigation table above points to for state-machine questions.

**Design, deliberately not a class-per-state FSM:** each state has one function per concern (`update_editor`, `update_physics`, `update_solving`, `update_visualizer`, plus their event/render counterparts where applicable), not a `State` base class with `on_enter`/`handle_event`/`update`/`render` hooks. The reason: `VISUALIZER → PHYSICS` deliberately *reuses* the existing `PhysicsEditor` instance rather than re-entering fresh (mesh/BCs must survive so a re-solve is possible), so a "construct a new State object per visit" pattern would have to special-case the one thing its own on-enter/on-exit hooks are supposed to handle uniformly. Dispatch tables fix the real defect (four hand-synced chains, `PHYSICS`'s four request-flag `if`s having no structural mutual exclusivity) with the least new structure.

- **`AppContext`** — a plain dataclass holding what used to be a dozen bare local variables in `run_app()`'s closure: `camera`, `editor`, `gfx` (the shared `Renderer`, since v1.9.0), `state`, `vbos` (now `dict[str, VboHandle]`), `physicseditor`, `mesher`, `solver` (kept alive across SOLVING and VISUALIZER, same as before), `solver_panel`, `visualizer`, `vis_mesher`, `live_field`. The old `screen`/`renderer` fields were removed in v1.9.0 — nothing outside `main.py` reads the pygame surface or ImGui backend anymore.
- **`update_editor(ctx)`** — EDITOR → PHYSICS on `ctx.editor.finished`; constructs `PhysicsEditor(...)`.
- **`update_physics(ctx)`** — dispatches on `ctx.physicseditor.pending_action` (a `PhysicsAction` enum from `physics_editor.py`, replacing the old four independent `mesh_requested`/`load_requested`/`load_visualization_requested`/`solve_requested` booleans — see that module's section below). `MESH` and `LOAD_MESH` stay in `PHYSICS`; `LOAD_VISUALIZATION` jumps straight to `VISUALIZER` (skipping `SOLVING` entirely); `SOLVE` builds `Solver`+`SolverPanel`+the live-preview `Visualizer` (`live_field`) and moves to `SOLVING`. Same per-action logic as before v1.8.0, just organized as one function keyed on a single field instead of four independently-guarded `if`s.
- **`update_solving(ctx)`** — SOLVING → VISUALIZER when `ctx.solver_panel.finished`. Reads `ctx.solver.results` (a `SolverResults`, see `solver_protocol.py` below) instead of reaching into `solver.P`/`solver.U`/`solver.final_res_cont`/`solver.final_res_mom` as bare attributes — this is the v1.8.0 fix for the old solver-boundary leak (`main.py` used to depend on those four attribute names existing on whatever `SolverProtocol` implementation was in use, undocumented by the ABC). Reuses `live_field`'s VBOs by promoting it directly to `ctx.visualizer` — no second `Visualizer`/VBO set is ever allocated for the same solve.
- **`update_visualizer(ctx)`** — VISUALIZER → PHYSICS when `ctx.visualizer.finished`; calls `.destroy()` to free that cycle's VBOs. `ctx.vbos` (the wireframe mesh) and `ctx.mesher`/`ctx.physicseditor` are untouched, so "Solve" is immediately available again with tweaked BCs/solver settings.
- **`_apply_loaded_mesh_settings(physicseditor, loaded, vbos)`** (module-level) — rebuilds BC lines, meshing/solver scalar settings, `bc_spacing_map`/`refinement_zones`, and the wireframe preview VBOs from a loaded `.npz` dict. Shared by `LOAD_MESH` and `LOAD_VISUALIZATION` so returning to `PHYSICS` afterward always shows a consistent, populated state. Its VBO rebuild goes through `Mesher.upload_wireframe_bundles()` (see `mesher.py` below), which since v1.9.0 returns `VboHandle`s.
- **Render handlers** (`render_editor`/`render_physics`/`render_solving`/`render_visualizer`) are pure drawing since v1.9.0 — each passes `ctx.gfx` into the state object's `draw()`. The `viz_snapshot` drain that used to sit in `render_solving` (state mutation: consume the one-shot snapshot, re-upload the color VBO) moved into `update_solving`'s not-finished branch, so SOLVING is composed the same way as every other state: update mutates, render draws (`live_field.draw_geometry(gfx)` + walls `gfx.draw_vbo` + `solver_panel.draw(live_field=...)`).

---

### `camera.py` — Coordinate System (pure math since v1.9.0)
**Owns:** world↔screen conversion and cursor-anchored zoom. Nothing else — all drawing (immediate-mode helpers, the GL matrix transform, VBO draws) moved to `renderer.py` in v1.9.0, and the vestigial `screen` parameter that used to thread through every draw call is gone with them. `camera.py` has no OpenGL imports.

**The coordinate transform:**
```
screen_x = (world_x + offset[0]) * scale
world_x  = (screen_x / scale) - offset[0]
```
`scale` = pixels per world-unit. `offset` = world-space pan offset.

**Surface:** `to_screen(world_point)`, `screen_to_world(screen_pos)`, `handle_zoom(mouse_pos, scroll_y)`. The one instance lives on `ctx.camera` and is also held by the `Renderer` (`gfx.camera`) — same object, two access paths (event/hit-test code uses `ctx.camera`; draw code reaches it through `gfx`).

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

**Load Visualization:** `open_load_visualization_dialog()` sits alongside `open_load_dialog()` — same tkinter file-picker pattern, but validates `'P' in data and 'U' in data` before accepting the file (prints and returns otherwise, so picking a plain mesh-only save here fails clearly rather than crashing downstream), then sets `loaded_visualization` and `pending_action = PhysicsAction.LOAD_VISUALIZATION`. `app_state.py`'s `update_physics()` reads this to jump straight to `VISUALIZER` — see the `app_state.py` section above.

**`pending_action` (since v1.8.0):** a single `Optional[PhysicsAction]` field (`PhysicsAction` is a small enum: `MESH`, `LOAD_MESH`, `LOAD_VISUALIZATION`, `SOLVE`, defined in this module) replaces four independent booleans (`mesh_requested`/`load_requested`/`load_visualization_requested`/`solve_requested`) that used to be checked as four separate, unguarded `if`s in `main.py`'s `PHYSICS` transition. Each UI button/dialog sets `pending_action` to exactly one value; `app_state.py`'s `update_physics()` reads and clears it each frame. One nullable field composes correctly if a future action is added (e.g. a temperature-solve button); four independent booleans didn't structurally guarantee only one fires per frame.

**Important:** `inlet_velocity` is a single float here, but the Solver receives `[inlet_velocity, 0.0]` — a 2D vector. This is assembled in `main.py`.

**Solver Settings:** A collapsing header between Refinement Zones and the action buttons. `alpha_u` and `alpha_p` are sliders (continuous feedback), `max_iterations` is a plain int input, `tolerance` is a log10 integer slider (`3`-`10`) that shows the resolved value inline (e.g. `= 1e-04`) so 1e-4 vs 1e-6 is never ambiguous, and `viz_interval` is editable so the user can trade off live-update frequency against GPU overhead on large meshes. These five fields are passed straight into `Solver.__init__()` / `SolverPanel.__init__()` when "Solve" is clicked.

**Window sizing:** The main "Mesher Settings" window uses `imgui.set_next_window_size_constraints((0, 0), (480, 0.9 * display_height))` plus `WINDOW_ALWAYS_AUTO_RESIZE`, so it grows to fit content (BC list, refinement zones, solver settings) but caps out at 90% of the screen height and gets its own internal scrollbar beyond that, instead of growing off-screen.

---

### `solver_protocol.py` — the Solver ABC

Defines `SolverProtocol(ABC)` with five abstract members: `initialize_conditions()`, `step(**state)`, `finalize(**final_state)`, `field_snapshot` (a property), and `results` (a property, added v1.8.0). The `**state`/`**final_state` convention means a concrete solver owns its own internal keys entirely — `SolverPanel` never inspects them. The only keys that are part of the **public contract** are `'residuals'` (dict of named floats) and `'converged'` (bool) in the dict returned by `step()`; everything else round-trips opaquely between `SolverPanel` and the concrete solver. `step()` returning `None` signals fatal divergence. `field_snapshot` must return copies (not views) of the current field arrays, at minimum `{'U': ndarray(Nc,2), 'P': ndarray(Nc,)}`.

**`results` / `SolverResults` (since v1.8.0):** `results` must return a `SolverResults` dataclass (`U`, `P`, `res_cont`, `res_mom`, plus an `extra: dict` forward-compat escape hatch for fields a future solver adds — e.g. a temperature solver's `extra['T']` — without another breaking change to this ABC), valid after `finalize()`. This closes the one real solver-boundary leak that predated v1.8.0: `app_state.py`'s `update_solving()` used to read `solver.P`/`solver.U`/`solver.final_res_cont`/`solver.final_res_mom` as bare attributes, undocumented by the ABC — any alternative `SolverProtocol` implementation had to happen to expose those exact names for the app to work post-solve. `Solver.results` is a one-line property packaging the *same* underlying attributes (`self.U`/`self.P`/`self.final_res_cont`/`self.final_res_mom`) — those attributes were **not** renamed or removed, since `test_force_balance.py` and `test_holes.py` still read them (and a much wider attribute surface — `owner`, `boundary_tags`, `Sf`, `Cf`, `df`, `magDf`, `magSf`, `cell_centers`) directly, bypassing the ABC entirely, which is fine since those are test-only call sites, not `main.py`/`app_state.py`.

**Why it exists:** a future LES solver (or the planned temperature/scalar-transport solver) just inherits this ABC, implements the five methods, and the rest of the app — `SolverPanel`, `app_state.py`, `Visualizer` — works with zero changes.

---

### `solver.py` — SIMPLE Algorithm Implementation

Implements `SolverProtocol`. Three surgical changes from the previous synchronous version:
- `__init__` now accepts `alpha_u`, `alpha_p`, `max_iterations`, `tolerance` as parameters (defaults: `0.7`, `0.3`, `1600`, `1e-8`).

**2026-07 mesh-refinement-divergence fixes** (validated on the Poiseuille 3k–135k mesh series; before the fixes the 36k+ meshes produced 83–1183% dp/dx error, after: sub-1% velocity error and stable convergence):
- `_face_D()` — single source for the face pressure-velocity coupling coefficient. Interpolates the *cell ratio* `alpha_u·V/a_P` (OpenFOAM rAU-style, "average of ratios") instead of dividing interpolated V by interpolated a_P — the old form broke at the boundary-layer-quad/triangle interface where adjacent cell areas jump 40–70×. `alpha_u` makes it consistent with the relaxed momentum diagonal actually solved; without it the effective pressure relaxation was `alpha_p·alpha_u` (~0.06). Same `alpha_u` factor in `CORRECT_PRESSURE_AND_VELOCITY`.
- `ASSEMBLE_PRESSURE_CORRECTION` no longer adds diagonal terms for inlet/wall faces (prescribed-flux boundaries contribute nothing to the p' equation; they acted as a spurious p'→0 anchor along every wall). Only the outlet (pressure-Dirichlet) contributes. Safety pins for a no-outlet (cavity) mesh and empty rows reuse `_impose_dirichlet_on_system`.
- `_lambda_int` clamped to `[1, 5]×` the orthogonal value `|Sf|/|df|` (was unbounded and could go negative on skewed pairs, destroying the p'-matrix M-property); `_T_int` computed after the clamp absorbs the remainder explicitly.
- `_solve_pressure` falls back to a direct `spsolve` when BiCGSTAB stalls twice, instead of silently accepting the stalled iterate (`_n_pressure_direct` counts fallbacks). pyamg is now a hard dependency so the AMG preconditioner path actually runs.
- Convergence: `cont_rms` is compared against `tolerance × total inlet mass flux` (relative, mesh-independent). `GET_VAR_STAR` warns loudly when the 5×|U_in| clip fires — a clipped run is untrustworthy.
- `Solve()` is now a thin ~20-line wrapper around `step()`, kept purely for backward compatibility with any caller that wants a blocking, non-threaded solve.
- The old loop body became `step(**state)`. It returns the opaque solver state dict (`a_P_u`, `a_P_v`, `initial_cont_rms`, ...) plus the protocol-required `'residuals'`/`'converged'` keys, plus underscore-prefixed raw arrays (`'_b_p'`, `'_r_u'`, `'_r_v'`) that `finalize()` picks up. `finalize(**final_state)` extracts those into `self.final_res_cont` / `self.final_res_mom` (cell-level arrays consumed by `Visualizer`) — the leading underscore avoids colliding with the solver's own state keys. `step()` also refreshes `self._live_res_cont`/`self._live_res_mom` (same `abs(b_p)` / `sqrt(r_u²+r_v²)` formulas as `finalize()`, just computed every iteration instead of once at the end) so the live preview can show Continuity/Momentum Error mid-solve, not just Pressure/Velocity. `field_snapshot` returns `.copy()`'d `U`/`P`/`res_cont`/`res_mom` on every call so the solver thread can't corrupt a snapshot the main thread is still reading.

---

### `solver_panel.py` — Threaded Solve Orchestration + Monitor UI

Wraps a `SolverProtocol` instance and runs it on a background thread so the UI never blocks during a solve.

- The solver thread runs `step()` in a loop. It checks `stop_event` at the top of every iteration, plus a combined pause/run-to check *before* running the step — so "paused at iteration N" always means step N has **not** run yet, making a `step_one()` control unambiguous.
- Residuals are pushed onto an unbounded queue — no data point is ever lost, even if the main thread falls behind for a frame.
- The viz (field) snapshot queue has `maxsize=1` and drops stale snapshots, so the main thread always sees the *latest* state, never a growing backlog. `solver_panel.py` only stores the drained snapshot on `self.viz_snapshot`; it does not render it — `app_state.py`'s `update_solving()` consumes it into the `live_field` `Visualizer` (moved out of the render handler in v1.9.0; see the `app_state.py` section above).
- `_drain_queues()` (called once per frame from `draw()`, main thread only) rebuilds the log10 float32 arrays consumed by `imgui.plot_lines()` for the residual plots. **Don't pass `scale_min`/`scale_max` as `float('nan')`** to force autoscale — ImGui's autoscale sentinel is `FLT_MAX` (`imgui.FLOAT_MAX`), and `NaN != FLT_MAX`, so a NaN "sentinel" is used as a literal (degenerate) axis range and the line never renders. Simplest fix: omit `scale_min`/`scale_max` entirely and let the defaults (`FLOAT_MAX`) trigger autoscale.
- `resume()` sets `self.state = "RUNNING"` in addition to clearing `_pause_event` — this is required for the Pause/Resume button and status badge to flip correctly. Without it, `self.state` only ever leaves `"PAUSED"` when the thread reports a terminal message (`converged`/`max_iters`/`diverged`), so Resume looked broken (badge stuck on "PAUSED", button never reverted to "Pause") even though the thread had genuinely resumed stepping.
- The `'done'` message handler moves state to `"DONE"` from either `"RUNNING"` **or** `"PAUSED"` (not just `"RUNNING"`) — otherwise clicking Stop while paused left `self.state` stuck at `"PAUSED"` forever after the thread actually exited.
- Thread finalization (`finalize()`) runs exactly once regardless of whether the loop exited via convergence, hitting `max_iterations`, or a user-initiated stop.
- **By design**, once `self.state` reaches `"DONE"`/`"DIVERGED"` the panel only offers "Open Visualizer" — Stop/Pause/+1 Step/Run-to-iter intentionally disappear rather than being shown as inert no-ops. The thread has already exited at that point, and making those controls *actually* resume iterating would require persisting `last_result`/`iteration` across thread restarts and suppressing the immediate re-trigger of the convergence check (`converged = iteration > 50 and res_cont_rms < tolerance` fires again on the very next step once you're already under tolerance). That's a real change to the solve-loop's control flow, not a UI tweak — deliberately left alone.

---

### `visualizer.py` — Post-Processing / Field Visualization

Renders the colored mesh (fill) plus, in the full post-solve UI, the "Post-Processor" control window and point-probe. Used two ways: as the final `visualizer` after a solve, and as the `live_field` preview during `SOLVING` (see the `main.py` section above) — same class, same VBOs, two call patterns:
- `draw(gfx, dt)` — full experience: recolors on variable change, draws geometry + vectors + smoke particles (all through `gfx` helpers), then the ImGui "Post-Processor" window and probe tooltip (probe uses `gfx.camera.screen_to_world`). Used only in the `VISUALIZER` state. `dt` (seconds, from `main.py`'s `clock.tick(60)`) drives smoke-particle advection only — everything else in `draw()` is per-frame-independent.
- `draw_geometry(gfx)` — just the colored-mesh fill (`gfx.draw_vbo_colored(pos_vbo, color_vbo)`), no ImGui, no smoke particles. Used by `render_solving` so the live preview doesn't pop up its own window (or animate particles through a still-changing field) on top of the Solver Monitor. This is *why* smoke particles only ever animate post-solve, with no extra state-machine guard needed.
- Buffers (`pos_vbo`/`color_vbo`/`vector_vbo`) are `VboHandle`s since v1.9.0 — positions static, colors/vectors `GL_DYNAMIC_DRAW`, re-uploaded via `handle.upload(...)`.
- `update_fields(P, U, res_cont=None, res_mom=None)` — swaps in new field data without touching geometry; caller must follow with `update_vbo_colors()` to push new colors to the GPU (`draw()`'s own auto-recolor only fires on a variable-combo change, not on new data — the live preview needs to recolor on *every* snapshot regardless of which variable is selected). `SmokeParticles` doesn't need to be told about this — it holds a back-reference to the owning `Visualizer` and reads `owner.U` live every step, so a stale-velocity bug is structurally impossible.
- `destroy()` — frees `pos_vbo`/`color_vbo`/`vector_vbo`, plus `self.smoke.destroy()` for the particle VBO. Call before dropping the last reference to a `Visualizer`/`live_field` (`main.py` calls this on `VISUALIZER → PHYSICS`) — GL buffers aren't reference-counted, so skipping this leaks GPU memory across repeated re-solves.
- `update_vbo_colors()` — maps `var_idx` (0=Pressure, 1=Velocity, 2=Continuity Error, 3=Momentum Error) to vertex colors. **Uses a robust, percentile-clipped range (2nd–98th percentile via `np.nanpercentile`), not true min/max.** A single outlier cell (classically a leading-edge stagnation point) sitting at the true extreme would otherwise single-handedly set the color scale for the entire field, so tiny genuine changes there make everything else look like it's oscillating wildly frame to frame — this is the fix for that. The "Range" readout in the Post-Processor window still shows the true (unclipped) min/max, which remains accurate; only the color mapping is clipped.
- `open_save_dialog()` — "Save Visualization" button in the Post-Processor window. Mirrors `PhysicsEditor.open_save_dialog()`'s tkinter pattern exactly (title "Export Visualization", distinct from the mesh dialog's "Export Solver Mesh"). Merges `self.mesh_data` with `P`/`U`/`res_cont`/`res_mom`/`var_idx`/`show_vectors`/`vector_scale` into one dict and passes it through the unmodified `meshIO.save_mesh_for_solver()` — same generic dict→`.npz` round-trip as a plain mesh save, just a bigger dict. Deliberately does **not** persist smoke-particle display settings (count, speed, lifetime, visibility toggle) — a scoped-out nice-to-have, not an oversight.
- `restore_display_settings(data)` — the load-side counterpart, called by `main.py` right after constructing a `Visualizer` from a loaded visualization dict. Casts `var_idx`/`show_vectors`/`vector_scale` back from the 0-d arrays `np.savez` produces (same reasoning as `meshIO.load_mesh_for_solver`'s `Nc`/`Nf` casting), each individually guarded by `if 'x' in data` so older/partial files degrade gracefully, then rebuilds the vector VBO and recolors.
- `open_export_vtu_dialog()` — "Export VTU" button in the Post-Processor window, same tkinter dialog pattern as `open_save_dialog()`, title "Export VTU", `.vtu` filter. Calls `vtuIO.export_vtu(self.mesh_data, filepath, P=self.P, U=self.U, res_cont=self.res_cont, res_mom=self.res_mom)` — see the `vtuIO.py` section below.

---

### `smoke_particles.py` — Smoke Particle Tracer

`SmokeParticles`, owned by `Visualizer` (`self.smoke`, constructed in `Visualizer.__init__` right after `self.tree`). Tracer particles advected through the solver's frozen post-solve velocity snapshot and continuously reseeded, so they read visually as flowing smoke — **not** real unsteady transport (the SIMPLE solver has no time dimension).

- Holds a **back-reference to the owning `Visualizer`** (`self.owner`), not copies of `centroids`/`U`/`tree` — every `step()` reads `owner.U`/`owner.centroids`/`owner.tree` live, so `Visualizer.update_fields()` needs no changes to keep particles in sync.
- **Advection**: inverse-distance-weighted sample over the 3 nearest cell centroids (`owner.tree.query(positions, k=3)`, vectorized over all particles in one call), then explicit Euler (`pos += U_sampled * dt * speed_scale`). Deliberately not using `solver.py`'s `_gx_int` (that's a face-midpoint interpolation factor tied to owner/neighbor topology, not a general point sampler).
- **Seeding**: particles spawn at random points among the mesh's velocity-inlet face midpoints (`owner.mesh_data['boundary_tags'] == 1`, paired with `owner.mesh_data['Cf']`), jittered along the boundary by roughly one local cell width. `mesh_data` is the `solver_data_pipeline()` dict — its `boundary_tags`/`Cf` are in SI metres regardless of the CAD drawing's units, so `_compute_inlet_points()` divides back out by `mesh_data['unit_to_meters']` to match `owner.centroids`' scale. Falls back to rejection-sampling a random point in the domain bounding box (verified with `Visualizer._is_point_in_cell`, capped retries, centroid fallback) if `mesh_data` is missing or has no tagged inlet faces.
- **Despawn/respawn**: by default a particle is *only* reseeded when it actually leaves the mesh — wanders past `despawn_multiplier × sqrt(bbox_area / Nc)` from its nearest centroid, a cheap distance heuristic reusing the same `tree.query` call rather than a per-frame exact polygon test. There is deliberately no default time-based despawn: a particle drifting into a slow/recirculating region just visually stalls (physically correct-*looking*, since sampled velocity there is near zero) rather than popping out on an arbitrary timer. `limit_lifetime` (off by default) opts back into a user-controlled lifetime (`lifetime`, ±20% randomized per particle so a population doesn't expire in sync) for anyone who wants particles to also expire after N seconds regardless of position.
- `set_count(n)` grows (reseeding new particles at the inlet/fallback) or shrinks (truncates) the position/age arrays live — backs the "Particle Count" slider.
- **Rendering**: one dynamic `VboHandle` (`GL_DYNAMIC_DRAW`, re-uploaded every `step()`), drawn via `gfx.draw_vbo(handle, mode=GL_POINTS, point_size=...)` — same shared path as the vector glyphs and wireframe.
- UI: "Show Smoke Particles" checkbox + "Particle Speed"/"Particle Size"/"Particle Count" sliders and a "Limit Particle Lifetime" checkbox (revealing a "Lifetime (s)" slider) in the Post-Processor window, mirroring the existing "Show Velocity Vectors" pattern.
- `Visualizer.__init__` takes an optional `mesh_data` param (the `solver_data_pipeline()` dict) purely to hand to `SmokeParticles` for inlet lookup — separate from `mesher` (the display-geometry source, which may be a live `Mesher` or a unit-rescaled dict). `main.py` passes its already-in-scope `mesh_data` through at the single `Visualizer(...)` construction site (used for both the fresh-mesh and loaded-`.npz` paths).

---

### `vtuIO.py` — VTK XML UnstructuredGrid Export

`export_vtu(mesh_data, filepath, P=None, U=None, res_cont=None, res_mom=None)` — hand-rolled ASCII `.vtu` writer (plain string formatting, no `pyevtk`/`meshio` dependency), for cross-validating the solver against other CFD codes (e.g. opening the result in ParaView).

- **Key simplification**: `P`/`U`/residuals are cell-centered (finite volume), which maps directly onto VTU's `CellData` rather than `PointData` — so points never need global deduplication/sharing across cells. Each cell writes its own private vertex copy straight from `mesh_data['cell_vertices']` (already SI metres — `solver_data_pipeline()` applies `unit_to_meters` to this array, unlike `Visualizer`'s rendering path which divides back out; VTU export wants real physical coordinates, so using it unconverted is correct here), with trivially sequential connectivity (a running vertex-count offset per cell) — no vertex-merge step.
- **Cell types**: `mesh_data['cell_types']` (0=triangle, 1=quad) maps directly to VTK's type codes (`VTK_TRIANGLE=5`, `VTK_QUAD=9`) — no re-deriving cell type from vertex count.
- **Velocity is padded to 3 components** (`U3[:, :2] = U`, z stays 0) since VTK vectors are always 3D even for a 2D solve.
- `P`/`U`/`res_cont`/`res_mom` are all optional — each is only written as a `CellData` `DataArray` if not `None`, so the same function works for a fields-only geometry export too.
- Verified via `xml.etree.ElementTree` round-trip against a real meshed case (rectangle + hole, `test_holes.py`'s setup) — not just eyeballed: `NumberOfPoints`/`NumberOfCells` match `sum(cell_nverts)`/`Nc`, connectivity is a clean `0..N-1` permutation, per-cell VTK type codes match `cell_types`, and CellData values round-trip numerically.

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
- **Refinement zones**: If `self.refinement_zones` is non-empty, a single unified Poisson-disk pass fills all zones + background simultaneously. Each zone gets a guaranteed seed at its centroid (so overlapping/disconnected zone arms are all populated). Zone selection at a point is two-tier: if the point is inside one or more zones, the **smallest-area** containing zone wins outright (so a nested, more-refined zone dominates its own footprint instead of losing to a larger enclosing zone that happens to have a more negative raw distance-to-boundary there); only points outside every zone fall back to nearest-zone-by-distance. `_get_local_r()` (scalar, used for the one-time zone-centroid seeds) and the vectorised `_precompute_domain_grids()` (the hot Poisson-disk pass) both implement this same two-tier rule. Once the winning zone is picked, spacing uses signed-distance blending (smoothstep over that zone's own `buffer_mult * r` buffer) to transition smoothly from `r/factor` inside the zone to the global background `r` outside — avoiding sudden cell-size jumps that cause solver artefacts. Note the blend target is always the global `r`, not an enclosing zone's spacing, so a small nested zone's own `buffer_mult` should stay small relative to its size to avoid blending past the enclosing zone before it takes over.
- Hard cap at 2000×2000 grid cells — if triggered, `w` is rescaled to fit.

#### Phase 4: Triangulation & Filter
- Calls `Bowyer_watson(all_interior_pts)`.
- `filter_triangles()` removes triangles outside the inner ring using `matplotlib.path.Path.contains_points`.

#### Phase 5 (opt-in, post-mesh): Smoothing (`smooth_mesh()`, since v1.10.0)
- Addresses a known mesh-quality gap: Steiner points only have a floor (`safe_zone`, Phase 3) on how close they land to the ring, not a target, so seam-triangle quality (ring ↔ first Steiner row) varies widely — measured 3–5x spread in height/spacing ratio across mesh densities. Confirmed *not* correlated with solver accuracy; it's a mesh-quality-only concern. Not run automatically — the destructive, automatic corner-sliver fix attempted earlier broke prismatic layers and was reverted (see git history around that era), so this is deliberately additive/reversible: a button the user presses, undoable only by re-meshing.
- Ring points (`stack[-1]` of each `self.loop_layer_stacks` entry) and all boundary-layer points are frozen; only Steiner points move, toward the centroid of their Delaunay neighbours (`relaxation` factor, default 0.5), for up to `passes` iterations (default 3).
- **Guard against ring encroachment:** per-point — if a move would more than halve a point's distance to its nearest ring point, that point is skipped for the pass (not moved), independent of the pass-level accept/reject below.
- **Pass accept/reject:** uses `mesh_quality.seam_quality()` — the worst (smallest) min-angle among *true* seam triangles (at least one ring vertex AND at least one non-ring vertex; a triangle with all three vertices on the ring is a corner artifact of frozen geometry, excluded so it can't dominate the metric or block on regressions it structurally can't fix). A pass is committed unless it drops quality by more than `tolerance_deg` (default 1.0) below the best value seen so far — **not** a strict never-get-worse rule, because moving thousands of points reshuffles Delaunay topology everywhere; some unrelated triangle drifting by a fraction of a degree on any pass is normal noise, and a strict rule rejects nearly every pass on a realistically sized mesh (this was found empirically: the first version required literal non-regression and committed 0/N passes on a ~5k-cell test mesh, and was fixed by adding the tolerance band; see git log for the exact before/after).
- `cfdeditor/mesh_quality.py` (new file, v1.10.0): `triangle_min_angle()` (law of cosines) and `seam_quality()` — kept separate from `mesher.py` since it's a small, independently-testable geometric utility.
- UI: `PhysicsAction.SMOOTH_MESH` (`physics_editor.py`), button only shown when `self.has_mesh and self.mesher is not None` (a *loaded* `.npz` mesh sets `has_mesh` without ever constructing a `Mesher`, so the guard matters — this crashed once before the `mesher is not None` check was added). Passes/relaxation are in a "Smoothing settings" collapsing section next to Mesher settings/Refinement Zones, not inline on the button row. Handled in `app_state.py → update_physics()`, which also re-runs `rebuild_wireframe_vbos()` and invalidates `loaded_mesh` (same pattern as the `MESH` branch) since a stale loaded mesh shouldn't silently override the smoothed one on Solve.

#### Wireframe VBOs (`rebuild_wireframe_vbos()`, `upload_wireframe_bundles()`, since v1.8.0)
- `get_render_data()` (pre-existing) turns the live mesh's triangulation/boundary/CAD-line objects into `{key: (float32 coords array, point count)}` bundles.
- `rebuild_wireframe_vbos(self, old_vbos)` — instance method; deletes `old_vbos`' buffers and uploads fresh ones from `self.get_render_data()`. Called from `app_state.py`'s `update_physics()` on `MESH`/remesh.
- `upload_wireframe_bundles(old_vbos, bundles)` — `@staticmethod`; since v1.9.0 returns `{key: VboHandle}` (delete old handles → `VboHandle().upload(data)` per key with count>0). Also called directly (not via an instance) from `_apply_loaded_mesh_settings()` in `app_state.py`, which builds its own bundles dict from a loaded `.npz`'s raw `cell_vertices`/`cell_nverts` arrays — there's no live `Mesher` object in that path, so it can't call `get_render_data()`.
- These handles are drawn by `PhysicsEditor.draw()` and `render_solving` via `gfx.draw_vbo(vbos.get(key), color=...)` — the dead `Mesher.draw()` fallback renderer (plus its `finished`/`finish()` flag and imgui/pygame imports) was deleted in v1.9.0.

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
- `check_circum_bulk`: vectorized (numba `prange`, parallel) determinant incircle test against every candidate triangle at once — the only incircle check left; the old scalar `checkCircumcentre` was removed in v1.7.1 as dead code (superseded, zero call sites).
- `orientCCW`: mutates the triangle in-place by swapping b/c if CW. **Side effect** — be aware when debugging triangle winding. Called directly by `bowyerwatson.py`.

---

### `triangle.py` / `quad.py` — Cell Geometry
Both implement the same interface: `vertices()`, `edges()`, `centroid` (property), `area` (property). (Their `draw()` methods — reachable only from the dead `Mesher.draw()` chain — were deleted in v1.9.0.)

`edges()` returns `frozenset` pairs — this is what Bowyer-Watson and `edge_map` use as keys. **The frozenset hashing is what makes edge matching work.**

`Triangle.area`: cross-product formula, absolute value — always positive.
`Quad.area`: Shoelace formula, absolute value.
`Quad.centroid`: polygon centroid formula (not simple average) — important for irregular quads.

---

### `triangulation.py` — Triangle Container
O(1) add/remove container backed by a pre-allocated NumPy array. `remove_triangle` uses a `_tri_to_idx` dict (keyed by `id(triangle)`) for O(1) lookup, then swaps the vacated slot with the last row (swap-with-last) so the array stays compact. `coords` property returns a zero-copy view of the live rows — no list conversion, no `np.asarray` copy.

### `mesh_audit.py` (repo root) — Offline Mesh-Integrity Audit
Standalone CLI (`python mesh_audit.py mesh.npz ...`) that validates a saved solver mesh without running the solver: phantom boundary faces (boundary-tagged faces whose midpoint lies on no CAD line — i.e. holes in the cell fabric mistagged by `solver_data_pipeline`'s nearest-midpoint fallback), per-tag boundary length vs CAD length, per-cell face closure Σ±Sf≈0, non-orthogonality / `lambda_int` sign and magnitude, and `gx` / area-ratio extremes. Exit code 2 if phantom faces are found. Built during the mesh-refinement-divergence investigation (2026-07).

---

## `validation/` — Poiseuille Verification Suite

Analytical verification of the solver against plane Poiseuille flow (ρ=1000, μ=1, H=0.01 m, ū=0.1 m/s prescribed ⇒ dp/dx = −12000 Pa/m, u_max = 0.15 m/s, τ_w = 60 Pa).

**Entry point: `validate_all`.** Prerequisite: run `paraview_export_velocity_poiseuille.py` inside ParaView first — it writes the CSVs *and* `poiseuille_meta.csv` (true cell counts, which the GCI depends on).

**Data location.** `PYTHON_LEARNING/Meshes/Poiseuille/` — deliberately **outside the repo** (~107 MB of regenerable output; keeping it out means it can never be committed). Exactly two places name it, both a single clearly-marked constant: `cfg.data_dir` in `poiseuille_config.m` and `DATA_DIR` in the ParaView script. **There is no path searching and no fallback** — a wrong path stops the run with a named error. (An earlier version searched four candidate directories and silently picked one; besides being able to bind the suite to a stale mesh set, its `raise SystemExit` on no-match *terminated ParaView* when run from the GUI Python Shell, where `__file__` is undefined.)

| File | Role |
|---|---|
| `validate_all.m` | **Entry point** — runs the four validators in order. Uses a `run_isolated` helper because `run()` executes in the caller's workspace and the validators reuse `i` as a loop index. |
| `poiseuille_config.m` | **Single source of truth** — fluid/geometry constants, the *fixed* analytical anchors, ordinal colour ramp, `cfg.data_dir`, and real cell counts read from `poiseuille_meta.csv` (hard error if absent — guessing counts would silently corrupt `h = √(A/N)` and thus the GCI). Edit constants here, never in the scripts. |
| `read_poiseuille_csv.m` | Robust ParaView-CSV reader; defensive column matching (`Velocity:0`→`Velocity_0`) and mandatory `vtkValidPointMask` filtering. |
| `validate_poiseuille.m` | Velocity: **absolute** L2 (vs true parabola), **shape** L2 (vs ū-rescaled parabola), mass error, u_max, symmetry, wall shear. |
| `validate_pressure.m` | dp/dx fitted over the **developed region only**, vs the fixed −12000 Pa/m; linearity R²/RMS. |
| `validate_gci.m` / `gci_triplet.m` | Grid-convergence study — observed order *p*, Richardson extrapolation, GCI (Roache / ASME V&V 20, Celik et al. 2008). Flags oscillatory/degenerate triplets instead of inventing numbers. |
| `test_gci.m` | Unit test for `gci_triplet` (14 assertions): recovers p=1.0/1.5/2.0/3.0 from synthetic power-law data at the real non-constant refinement ratios. Self-contained — needs no exported data, so it runs on a bare checkout. |
| `validate_diagnostics.m` | Mass conservation vs x, entrance length (Chen correlation), and the `ContinuityResidual`/`MomentumResidual` fields. |
| `paraview_export_velocity_poiseuille.py` | `pvpython` batch export → profile, centreline pressure, centreline u(x), per-station profiles, plus `poiseuille_meta.csv` with true cell counts. |

**The rule this suite exists to enforce:** the analytical solution is anchored to the *prescribed* inlet velocity and is therefore mesh-independent. An earlier version derived ū from the CFD result and rebuilt "theory" from it — circular, and it silently absorbed a +13.6% mass-conservation error into a moving "Theory dp/dx" column. Never re-derive the theory from solver output.

Run from inside `validation/` (or `addpath` it): `validate_all` for everything; `test_gci` to check the GCI maths alone; each validator by name to run just one.

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
Solver.results → SolverResults(U, P, res_cont, res_mom, extra={})  # what app_state.py reads post-solve
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

1. **The frame contract** — `renderer.py`'s `Renderer` owns the ONLY `imgui.new_frame()`/`imgui.render()` pair and the ONLY `PygameRenderer` instance (`gfx.backend`, created once in `main.py`). Never call the frame-lifecycle functions or instantiate `PygameRenderer` in a module — a second `new_frame`/`render` anywhere crashes or blanks the UI.
2. **Edge key rounding** — the `round(..., 6)` in `get_edge_key()` inside `solver_data_pipeline()`. Change precision and the edge map breaks silently.
3. **`frozenset` edges** — Triangle and Quad both use frozenset for edges. Changing to tuples breaks all dict lookups.
4. **`orientCCW` mutation** — `constructor.orientCCW()` mutates the triangle. The Bowyer-Watson loop (`bowyerwatson.py`) calls it directly on the super-triangle and every new triangle. Don't assume triangle vertex order is stable after this.
5. **`polygon_orientation` sign convention** — Positive = CW in the shoelace convention used here (note: this is *opposite* to the standard mathematical convention where positive area = CCW). The `boundary_layer` normal-flip depends on this.
6. **`build_polygon` comparison** — Uses `pivot == line.a` (i.e. `Point.__eq__`). `Point.__eq__` now uses a tight tolerance (`math.isclose`, abs_tol=1e-9) so it is robust to small coordinate drift while still treating distinct vertices as distinct. **`Point.__hash__` is intentionally left as the exact coordinate hash** — Bowyer-Watson dedup (`set()` on Points) relies on bit-identical coordinates, so do NOT make `__hash__` tolerance-based.
7. **`bc_map` string matching** — The strings in `bc_map` in `create_boundary_points()` must exactly match the `boundary_types` list in `physics_editor.py`.
8. **Distance-weighted interpolation (`_gx_int`)** — The solver uses `self._gx_int = d_Pf / d_PN` (distance from owner cell center to face midpoint, over total owner→neighbor distance) for all face interpolations (velocity, pressure gradient, coupling coefficient `a_P_f`, cell volume, `p_face`). Do NOT replace this with a fixed `0.5` arithmetic mean — on non-uniform/refined meshes the 0.5 average biases interpolation toward the wrong cell, creating artificial pressure gradients that make refinement zones behave like solid bodies. On uniform meshes `g_x = 0.5` exactly, so the distance-weighted form is strictly a superset.

---

## 6. Known Issues / Technical Debt

- `line.py`: `u_val`, `v_val`, `p_val` are unused (future per-line BC values).
- `solver.py → health_check()` prints every iteration — verbose, should be gated.
- `data_structures.txt` is partially outdated — `magSf` was added later and is in the actual pipeline but not the txt.

### Resolved technical debt (v1.9.0 — rendering-engine unification)
- **Frame lifecycle unified** — the `imgui.new_frame()` … `imgui.render()` + `renderer.render(get_draw_data())` bookends duplicated across five modules (editor, physics_editor, solver_panel, visualizer, and the dead mesher fallback) hoisted into `Renderer.begin_frame()`/`end_frame()`. Modules now only build panels.
- **Overlay hook added** — `Renderer.add_overlay(fn)` runs registered callables every frame in every state; first user is the persistent `NFLUIDS v{x.y.z}` stamp (`logo_overlay`, foreground draw list, captures no input). Cross-cutting UI no longer requires touching every state's draw path.
- **Buffer lifecycle uniform** — every GL buffer is a `VboHandle` (create/upload/delete + class registry); `VboHandle.delete_all()` at shutdown fixes the old "buffers only ever reclaimed by the next rebuild, never freed on exit" leak.
- **Camera demoted to pure math** — its five draw methods, the matrix push/pop pair, and `draw_vbo` moved to `Renderer`; the vestigial `screen` parameter (dead since the Pygame-surface days) is gone from every draw signature, and constructors no longer take `screen`/`renderer` at all.
- **`render_solving` asymmetry fixed** — the `viz_snapshot` drain (state mutation) moved to `update_solving`; render handlers are pure drawing.
- **Dead code deleted** — `Mesher.draw()` + `finished`/`finish()`, `Triangulation.draw`/`Triangle.draw`/`Quad.draw`/`Camera.draw_polygon` (orphan chain), the never-produced `'loaded'` wireframe key branch, and the `pygame.draw.line` origin crosshair in `editor.py` (a silent no-op inside an `OPENGL`-flagged window since it was written).
- The two copy-pasted refinement-zone `GL_QUADS`+`GL_LINE_LOOP` blocks collapsed into `Renderer.draw_rect` (identical colors/alphas/widths).

### Resolved technical debt (v1.8.0 — main.py / solver-boundary refactor)
- **State machine restructured** — `main.py`'s four hand-synced `if/elif current_state == ...` chains (bare-string state, ImGui event feed, gated event handling, transitions, rendering) replaced with a formal `AppState` enum and three per-state dispatch tables in the new `cfdeditor/app_state.py`. `main.py` is now just bootstrap + a three-dict-lookup loop. See the `app_state.py` section above.
- **Redundant ImGui event-feed chain removed** — all four states called `renderer.process_event(event)` on the identical shared renderer instance; collapsed to one unconditional call.
- **Solver-boundary leak closed** — `SolverProtocol` gained a `results` property (`SolverResults` dataclass) so `app_state.py` no longer depends on undocumented bare attributes (`solver.P`/`.U`/`.final_res_cont`/`.final_res_mom`) that happened to work by convention. See the `solver_protocol.py` section above.
- **Duplicated wireframe-VBO upload code merged** — the mesh-rebuild and loaded-`.npz` paths each inlined an identical delete/gen/bind/buffer OpenGL sequence; now both call `Mesher.upload_wireframe_bundles()` (or the instance-level `rebuild_wireframe_vbos()` wrapper). See the `mesher.py` section above.
- **`PhysicsEditor`'s four request booleans collapsed** — `mesh_requested`/`load_requested`/`load_visualization_requested`/`solve_requested` replaced with one `pending_action: Optional[PhysicsAction]`, removing a class of "two flags true in the same frame" risk that four independent `if`s (not `elif`s) didn't structurally prevent.
- **New regression coverage**: `test_state_transitions.py` (root-level, matching `test_holes.py`'s plain-script convention) characterizes the `AppState` transition graph using bare stand-in objects — no pygame/OpenGL context needed. `update_editor`/`update_solving`/`update_visualizer` are exercised directly (none of their code paths touch GL); `update_physics`'s `MESH`/`LOAD_MESH`/`LOAD_VISUALIZATION`/`SOLVE` branches are mirrored rather than executed directly, since they construct real `Mesher`/`Solver`/`SolverPanel`/`Visualizer` objects — `Visualizer.__init__` and `Mesher.rebuild_wireframe_vbos()` call `glGenBuffers()` directly (needs a bound GL context) and `SolverPanel.__init__` spawns a background thread, neither of which is safe or meaningful to exercise headlessly.
- **Deliberately deferred, not fixed**: a `MesherProtocol` analogous to `SolverProtocol` — there is one meshing algorithm and no second one planned (unlike the solver side, where a temperature solver and a transient solver are both on the roadmap), so a mesher ABC now would be speculative symmetry, not a fix for a real leak. `Mesher.draw()` (dead code — never called by `main.py`/`app_state.py`) and `Mesher.finished`/`finish()` (set but never read) were noticed while touching this file but left alone at the time — cleaned up in v1.9.0's rendering-engine unification as planned.

### Resolved technical debt (this pass)
- **Steiner grid OOB crash** — `get_grid_coords` in `create_steiner_points` now clamps indices to `[0, cols-1]`/`[0, rows-1]`, fixing an `IndexError` when a candidate landed exactly on the polygon bounds.
- **`create_steiner_points` default `r=550`** aligned to `4.0` to match `physics_editor` (was a latent mismatch).
- **`build_polygon` comparison** switched from `np.array_equal` (object identity through numpy) to `Point.__eq__` (tolerance-based) — see rule 6 above.
- **`Point.__eq__`** now uses `math.isclose` (abs_tol=1e-9) instead of exact float equality; `__hash__` left exact on purpose.
- **Boundary-face tagging tolerance** in `solver_data_pipeline()` is now `self.boundary_spacing` (scale-aware) instead of a hard-coded `1.0` world unit, so metre-scale geometries tag correctly.
- **Visualizer probe centroid** now reuses `cell.centroid` (shoelace for quads) instead of a naive vertex average, matching the solver's geometry.
- **Dead code removed**: `mesher.check_points()`, `mesher.create_boundary_layers()`, `constructor.intersect()`/`cross2d()`, and the empty `while accumulator >= dt` body in `main.py`.
- **Misleading comment** in `editor.py` ("Default to meters") corrected to reflect the actual mm default.
- **MultiPolygon safe-zone crash** — `_precompute_domain_grids` (`mesher.py:1043`) assumed `full_poly.buffer(-min_r*0.8)` always returned a single `Polygon`. When the eroded interior safe zone splits into disconnected pieces (e.g. a hole's boundary layers nearly touching the outer wall's), Shapely returns a `MultiPolygon` and `.exterior` raised `AttributeError`. Now iterates every piece and meshes all of them; a validity check on the pre-erosion domain polygon also warns when boundary layers actually overlap instead of silently mis-meshing.
- **Nested refinement-zone priority** — zone overlap used to resolve by raw, unnormalized distance-to-boundary, which let a larger zone win near its own center even where a smaller, more-refined zone was nested inside it. `_get_local_r()` (`mesher.py:261`) and `_precompute_domain_grids()` (`mesher.py:1043`) now use the two-tier containment-by-area rule described in Phase 3 above.

### Resolved technical debt (v1.7.1)
- **PyAMG silent fallback removed** — `solver.py` used to `try: import pyamg / except ImportError: _HAS_PYAMG = False` and silently use ILU instead of AMG with zero warning if pyamg wasn't installed, despite pyproject.toml already declaring it a hard dependency. `import pyamg` is now unconditional; a missing install now fails loudly at startup instead of quietly degrading solve quality forever.
- **Silent refinement-zone drop now warns** — `create_steiner_points()` (`mesher.py`) prints a warning when a zone's seed point can't be placed after 100 attempts, instead of silently giving that zone zero targeted refinement.
- **`filter_triangles()` overlap-check exception now warns** — the `except Exception: pass` around the Shapely intersection call (mesh-filtering edge case) now prints before falling back to the centroid-only check, instead of failing silently.
- **Dead code removed**: `constructor.checkCircumcentre()` and `constructor.updatebadedges()` (both superseded, zero call sites) plus their now-orphaned JIT kernels `_circum_scalar`/`_cross2d`; `Editor._cancel_or_undo()` (exact duplicate of `_handle_escape()`, never wired); a stray trailing `pass` in `Editor.finish()`; several unused imports across `camera.py`/`line.py`/`triangle.py`/`quad.py`/`bowyerwatson.py`/`editor.py`/`main.py`/`mesher.py`/`physics_editor.py`; unused locals in `solver.py` (`own_out` in `assemble_momentum_both`, `A_diag` in `_impose_dirichlet_on_system`, `f_int` in `health_check`).
- **`pygame_widgets` dependency dropped** — imported in three files but `Button(...)` was never instantiated anywhere in the repo; removed from `pyproject.toml` along with the dead imports.
