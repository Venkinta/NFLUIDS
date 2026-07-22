from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .line import Line
from .point import Point
from .mesher import Mesher
from .solver import Solver
from .solver_panel import SolverPanel
from .physics_editor import PhysicsEditor, PhysicsAction
from .visualizer import Visualizer


class AppState(Enum):
    """The app's top-level states, in their usual transition order:
    EDITOR -> PHYSICS <-> SOLVING -> VISUALIZER -> PHYSICS.
    """
    EDITOR = auto()
    PHYSICS = auto()
    SOLVING = auto()
    VISUALIZER = auto()


@dataclass
class AppContext:
    """Holds the mutable objects run_app()'s loop operates on.

    This isn't new architecture — it's naming what used to be a dozen bare
    local variables in run_app()'s closure, so the handler functions below
    can take one `ctx` argument instead of a long parameter list.
    """
    camera: object
    editor: object
    gfx: object = None             # Renderer — owns frame lifecycle + shared GL draws
    state: AppState = AppState.EDITOR
    vbos: dict = field(default_factory=dict)
    physicseditor: object = None
    mesher: object = None
    solver: object = None          # kept alive across SOLVING and VISUALIZER
    solver_panel: object = None
    visualizer: object = None
    vis_mesher: object = None      # Mesher instance or unit-adjusted dict, for Visualizer
    live_field: object = None      # Visualizer instance used as the live solve preview


def _apply_loaded_mesh_settings(physicseditor, loaded, vbos):
    """Rebuild BC lines, solver/mesh settings, and the wireframe preview VBOs
    from a loaded .npz dict (`loaded` — a plain mesh save or a visualization
    save, both carry the same mesh-side keys). Shared by both the "Load
    Mesh" and "Load Visualization" flows so returning to PHYSICS afterward
    always shows a consistent, populated state. Returns the new `vbos` dict."""
    bc_names = ["Wall", "Velocity Inlet", "Pressure Outlet", "Symmetry"]
    cad = loaded['cad_lines']
    lines = []
    for row in cad:
        ax, ay, bx, by, bc_idx = row
        line = Line(Point(ax, ay), Point(bx, by))
        line.boundary_type = bc_names[int(bc_idx)]
        lines.append(line)
    physicseditor.lines = lines

    physicseditor.n_layers         = int(loaded['n_layers'])
    physicseditor.growth_factor    = float(loaded['growth_factor'])
    physicseditor.thickness        = float(loaded['thickness'])
    physicseditor.boundary_spacing = float(loaded['boundary_spacing'])
    physicseditor.r                = float(loaded['r'])
    physicseditor.unit_to_meters   = float(loaded['unit_to_meters'])

    _unit_factors = {"mm": 0.001, "cm": 0.01, "m": 1.0}
    for i, name in enumerate(["mm", "cm", "m"]):
        if abs(_unit_factors[name] - physicseditor.unit_to_meters) < 1e-12:
            physicseditor._unit_idx = i
            break

    if 'bc_spacing_map' in loaded:
        physicseditor._bc_spacing = loaded['bc_spacing_map'].item()
        physicseditor._spacing_linked = False
        vals = list(physicseditor._bc_spacing.values())
        if len(set(round(v, 9) for v in vals)) == 1:
            physicseditor._spacing_linked = True
    else:
        for k in physicseditor._bc_spacing:
            physicseditor._bc_spacing[k] = physicseditor.boundary_spacing
        physicseditor._spacing_linked = True

    if 'refinement_zones' in loaded:
        zones = []
        for entry in loaded['refinement_zones']:
            coords      = entry[0]
            factor      = float(entry[1])
            buffer_mult = float(entry[2]) if len(entry) >= 3 else 5.0
            xs = coords[:, 0]; ys = coords[:, 1]
            zones.append({
                'rect': (float(xs.min()), float(ys.min()),
                         float(xs.max()), float(ys.max())),
                'factor': factor,
                'buffer_mult': buffer_mult,
            })
        physicseditor.refinement_zones = zones

    cv = loaded['cell_vertices'] / physicseditor.unit_to_meters
    nv = loaded['cell_nverts']
    tri_coords, quad_coords = [], []
    for i in range(len(nv)):
        n   = int(nv[i])
        pts = [(float(cv[i, v, 0]), float(cv[i, v, 1])) for v in range(n)]
        if n == 4:
            quad_coords.extend([pts[0], pts[1], pts[1], pts[2],
                                pts[2], pts[3], pts[3], pts[0]])
        else:
            tri_coords.extend([pts[0], pts[1], pts[1], pts[2],
                               pts[2], pts[0]])

    bundles = {
        'triangles': (np.array(tri_coords, dtype=np.float32), len(tri_coords)),
        'quads':     (np.array(quad_coords, dtype=np.float32), len(quad_coords)),
    }
    vbos = Mesher.upload_wireframe_bundles(vbos, bundles)

    physicseditor.has_mesh = True
    physicseditor.mesher   = None
    return vbos


# ----------------------------------------------------------------------
# Event handlers — called once per pygame event, only for the current state.
# States absent from this dict (SOLVING, VISUALIZER) get no per-event
# handling beyond the unconditional ImGui feed / mouse-wheel zoom that
# run_app() applies before this dispatch, matching current behavior.
# ----------------------------------------------------------------------

def handle_event_editor(ctx, event, want_mouse):
    if not want_mouse:
        ctx.editor.handle_event(event, ctx.camera)


def handle_event_physics(ctx, event, want_mouse):
    ctx.physicseditor.handle_event(event, ctx.camera)


EVENT_HANDLERS = {
    AppState.EDITOR: handle_event_editor,
    AppState.PHYSICS: handle_event_physics,
}


# ----------------------------------------------------------------------
# Update handlers — one per state, called once per frame. Each returns the
# next AppState (unchanged if no transition fires this frame).
# ----------------------------------------------------------------------

def update_editor(ctx):
    if not ctx.editor.finished:
        return AppState.EDITOR
    ctx.physicseditor = PhysicsEditor(ctx.editor.lines, ctx.editor.unit_idx)
    return AppState.PHYSICS


def update_physics(ctx):
    physicseditor = ctx.physicseditor
    action = physicseditor.pending_action
    physicseditor.pending_action = None

    # --- Mesh / Remesh ---
    if action == PhysicsAction.MESH:
        refinement_zones = physicseditor._get_refinement_polygons()
        mesher = Mesher(
            physicseditor.lines, physicseditor.n_layers,
            physicseditor.growth_factor, physicseditor.thickness,
            physicseditor.boundary_spacing, physicseditor.r,
            unit_to_meters=physicseditor.unit_to_meters,
            refinement_zones=refinement_zones,
            bc_spacing_map=physicseditor._bc_spacing
        )
        mesher.mesh()

        ctx.vbos = mesher.rebuild_wireframe_vbos(ctx.vbos)
        physicseditor.has_mesh = True
        physicseditor.mesher = mesher
        physicseditor.loaded_mesh = None  # invalidate stale loaded mesh so Solve uses this fresh remesh
        ctx.mesher = mesher

    # --- Smooth the current mesh in place (opt-in, no full remesh) ---
    elif action == PhysicsAction.SMOOTH_MESH:
        physicseditor.mesher.smooth_mesh(
            passes=physicseditor.smooth_passes,
            relaxation=physicseditor.smooth_relaxation)
        ctx.vbos = physicseditor.mesher.rebuild_wireframe_vbos(ctx.vbos)
        physicseditor.loaded_mesh = None  # stale after in-place smoothing

    # --- Load saved mesh (.npz) ---
    elif action == PhysicsAction.LOAD_MESH:
        ctx.vbos = _apply_loaded_mesh_settings(physicseditor, physicseditor.loaded_mesh, ctx.vbos)

    # --- Load saved visualization (.npz with solved fields) → straight to VISUALIZER ---
    elif action == PhysicsAction.LOAD_VISUALIZATION:
        data = physicseditor.loaded_visualization
        ctx.vbos = _apply_loaded_mesh_settings(physicseditor, data, ctx.vbos)
        physicseditor.loaded_mesh = data  # so Save Mesh / re-solve still works later

        vis_dict = dict(data)
        vis_dict['cell_vertices'] = data['cell_vertices'] / physicseditor.unit_to_meters
        vis_dict['cell_centers']  = data['cell_centers']  / physicseditor.unit_to_meters

        ctx.visualizer = Visualizer(
            vis_dict, data['P'], data['U'],
            res_cont=data.get('res_cont'), res_mom=data.get('res_mom'),
            mesh_data=data,
        )
        ctx.visualizer.restore_display_settings(data)
        return AppState.VISUALIZER

    # --- Solve → launch threaded SolverPanel ---
    elif action == PhysicsAction.SOLVE:
        # Build mesh data and decide what the Visualizer will use
        if physicseditor.loaded_mesh is not None:
            mesh_data = physicseditor.loaded_mesh
            scale     = physicseditor.unit_to_meters
            vis_dict  = dict(mesh_data)
            vis_dict['cell_vertices'] = mesh_data['cell_vertices'] / scale
            vis_dict['cell_centers']  = mesh_data['cell_centers']  / scale
            ctx.vis_mesher = vis_dict
        else:
            mesh_data      = ctx.mesher.solver_data_pipeline()
            ctx.vis_mesher = ctx.mesher

        ctx.solver = Solver(
            mesh_data,
            [physicseditor.inlet_velocity, 0.0],
            physicseditor.outlet_pressure,
            physicseditor.density,
            physicseditor.viscosity,
            alpha_u       = physicseditor.alpha_u,
            alpha_p       = physicseditor.alpha_p,
            max_iterations= physicseditor.max_iterations,
            tolerance     = physicseditor.tolerance,
        )
        ctx.solver_panel = SolverPanel(
            ctx.solver,
            max_iterations = physicseditor.max_iterations,
            viz_interval   = physicseditor.viz_interval,
        )
        # Live field preview: same colored-mesh renderer the post-solve
        # Visualizer uses, seeded with zero fields until the first
        # snapshot arrives from the solver thread.
        ctx.live_field = Visualizer(
            ctx.vis_mesher,
            np.zeros(mesh_data['Nc']),
            np.zeros((mesh_data['Nc'], 2)),
            mesh_data=mesh_data,
        )
        return AppState.SOLVING

    return AppState.PHYSICS


def update_solving(ctx):
    """SOLVING → VISUALIZER when user clicks "Open Visualizer"."""
    if not ctx.solver_panel.finished:
        # Pick up the latest live field snapshot (if any arrived this
        # frame) and push it to the live-preview color VBO. State
        # mutation, so it lives here rather than in render_solving.
        if ctx.solver_panel.viz_snapshot is not None:
            snap = ctx.solver_panel.viz_snapshot
            ctx.solver_panel.viz_snapshot = None
            ctx.live_field.update_fields(snap['P'], snap['U'],
                                         res_cont=snap.get('res_cont'),
                                         res_mom=snap.get('res_mom'))
            ctx.live_field.update_vbo_colors()
        return AppState.SOLVING
    # Reuse the live preview's geometry/VBOs — just refresh with the
    # final fields instead of allocating a second Visualizer.
    results = ctx.solver.results
    ctx.live_field.update_fields(results.P, results.U,
                                 res_cont=results.res_cont,
                                 res_mom=results.res_mom)
    ctx.live_field.update_vbo_colors()
    ctx.visualizer   = ctx.live_field
    ctx.live_field   = None
    ctx.solver_panel = None
    return AppState.VISUALIZER


def update_visualizer(ctx):
    """VISUALIZER → PHYSICS (state preserved — mesh/BCs intact, re-solve available)."""
    if not ctx.visualizer.finished:
        return AppState.VISUALIZER
    ctx.visualizer.destroy()   # free this cycle's pos/color/vector VBOs
    ctx.visualizer = None
    # Mesh wireframe VBOs (ctx.vbos) are kept so the mesh stays visible in
    # Physics Editor — only the Visualizer's own buffers are freed.
    return AppState.PHYSICS


UPDATE_HANDLERS = {
    AppState.EDITOR: update_editor,
    AppState.PHYSICS: update_physics,
    AppState.SOLVING: update_solving,
    AppState.VISUALIZER: update_visualizer,
}


# ----------------------------------------------------------------------
# Render handlers — one per state, called once per frame after update.
# ----------------------------------------------------------------------

def render_editor(ctx, dt):
    ctx.editor.draw(ctx.gfx)


def render_physics(ctx, dt):
    ctx.physicseditor.draw(ctx.gfx, ctx.vbos)


def render_solving(ctx, dt):
    ctx.live_field.draw_geometry(ctx.gfx)
    ctx.gfx.draw_vbo(ctx.vbos.get('walls'), color=(255, 255, 255))
    ctx.solver_panel.draw(live_field=ctx.live_field)


def render_visualizer(ctx, dt):
    ctx.visualizer.draw(ctx.gfx, dt)


RENDER_HANDLERS = {
    AppState.EDITOR: render_editor,
    AppState.PHYSICS: render_physics,
    AppState.SOLVING: render_solving,
    AppState.VISUALIZER: render_visualizer,
}
