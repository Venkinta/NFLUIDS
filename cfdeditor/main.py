from .line import Line
from .editor import Editor
from .mesher import Mesher
from .solver import Solver
from .solver_panel import SolverPanel
from .camera import Camera
from .physics_editor import PhysicsEditor
import pygame
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_ON_COPY = False
from OpenGL.GL import *
import imgui
from imgui.integrations.pygame import PygameRenderer
from .visualizer import Visualizer
from .point import Point

import numpy as np
import cProfile
import pstats


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

    for _vbo_id, _ in vbos.values():
        glDeleteBuffers(1, [_vbo_id])
    vbos = {}
    for key, coords in (('triangles', tri_coords), ('quads', quad_coords)):
        if coords:
            vbo_data = np.array(coords, dtype=np.float32)
            vbo_id   = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
            glBufferData(GL_ARRAY_BUFFER, vbo_data.nbytes, vbo_data, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            vbos[key] = (vbo_id, len(coords))

    physicseditor.has_mesh = True
    physicseditor.mesher   = None
    return vbos


def run_app():
    pygame.init()
    WIDTH, HEIGHT = 1920, 1080
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)

    def init_gpu(width, height):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    init_gpu(WIDTH, HEIGHT)
    clock = pygame.time.Clock()
    imgui.create_context()
    renderer = PygameRenderer()
    imgui.get_io().display_size = (WIDTH, HEIGHT)

    # --- Module instances ---
    editor        = Editor(screen, renderer)
    physicseditor = None
    mesher        = None
    solver        = None       # kept alive across SOLVING and VISUALIZER
    solver_panel  = None
    visualizer    = None
    vis_mesher    = None       # what Visualizer uses (Mesher obj or unit-adj dict)
    live_field    = None       # Visualizer instance used as the live solve preview

    current_state = "EDITOR"
    running = True
    camera  = Camera()
    vbos    = {}

    while running:
        dt = clock.tick(60) / 1000.0
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

            # Feed ImGui first so want_capture_mouse reflects this event
            # before any camera/module handling decides whether to react to it.
            if current_state == "EDITOR":
                renderer.process_event(event)
            elif current_state == "PHYSICS":
                physicseditor.renderer.process_event(event)
            elif current_state == "SOLVING":
                renderer.process_event(event)
            elif current_state == "VISUALIZER":
                visualizer.renderer.process_event(event)

            want_mouse = imgui.get_io().want_capture_mouse

            if event.type == pygame.MOUSEWHEEL and not want_mouse:
                camera.handle_zoom(pygame.mouse.get_pos(), event.y)

            if current_state == "EDITOR":
                if not want_mouse:
                    editor.handle_event(event, camera)

            elif current_state == "PHYSICS":
                physicseditor.handle_event(event, camera)

        # ------------------------------------------------------------------
        # State transitions
        # ------------------------------------------------------------------

        # EDITOR → PHYSICS
        if current_state == "EDITOR" and editor.finished:
            physicseditor = PhysicsEditor(screen, editor.lines, renderer, editor.unit_idx)
            current_state = "PHYSICS"

        elif current_state == "PHYSICS":

            # --- Mesh / Remesh ---
            if physicseditor.mesh_requested:
                physicseditor.mesh_requested = False
                refinement_zones = physicseditor._get_refinement_polygons()
                mesher = Mesher(
                    screen, physicseditor.lines, physicseditor.n_layers,
                    physicseditor.growth_factor, physicseditor.thickness,
                    physicseditor.boundary_spacing, physicseditor.r, renderer,
                    unit_to_meters=physicseditor.unit_to_meters,
                    refinement_zones=refinement_zones,
                    bc_spacing_map=physicseditor._bc_spacing
                )
                mesher.mesh()

                for _vbo_id, _ in vbos.values():
                    glDeleteBuffers(1, [_vbo_id])
                vbos = {}

                mesh_bundles = mesher.get_render_data()
                for key, (data, count) in mesh_bundles.items():
                    if count > 0:
                        vbo_id = glGenBuffers(1)
                        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
                        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
                        vbos[key] = (vbo_id, count)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                physicseditor.has_mesh = True
                physicseditor.mesher = mesher
                physicseditor.loaded_mesh = None  # invalidate stale loaded mesh so Solve uses this fresh remesh

            # --- Load saved mesh (.npz) ---
            if physicseditor.load_requested:
                physicseditor.load_requested = False
                vbos = _apply_loaded_mesh_settings(physicseditor, physicseditor.loaded_mesh, vbos)

            # --- Load saved visualization (.npz with solved fields) → straight to VISUALIZER ---
            if physicseditor.load_visualization_requested:
                physicseditor.load_visualization_requested = False
                data = physicseditor.loaded_visualization
                vbos = _apply_loaded_mesh_settings(physicseditor, data, vbos)
                physicseditor.loaded_mesh = data  # so Save Mesh / re-solve still works later

                vis_dict = dict(data)
                vis_dict['cell_vertices'] = data['cell_vertices'] / physicseditor.unit_to_meters
                vis_dict['cell_centers']  = data['cell_centers']  / physicseditor.unit_to_meters

                visualizer = Visualizer(
                    renderer, vis_dict, data['P'], data['U'],
                    res_cont=data.get('res_cont'), res_mom=data.get('res_mom'),
                    mesh_data=data,
                )
                visualizer.restore_display_settings(data)
                current_state = "VISUALIZER"

            # --- Solve → launch threaded SolverPanel ---
            if physicseditor.solve_requested:
                physicseditor.solve_requested = False

                # Build mesh data and decide what the Visualizer will use
                if physicseditor.loaded_mesh is not None:
                    mesh_data = physicseditor.loaded_mesh
                    scale     = physicseditor.unit_to_meters
                    vis_dict  = dict(mesh_data)
                    vis_dict['cell_vertices'] = mesh_data['cell_vertices'] / scale
                    vis_dict['cell_centers']  = mesh_data['cell_centers']  / scale
                    vis_mesher = vis_dict
                else:
                    mesh_data  = mesher.solver_data_pipeline()
                    vis_mesher = mesher

                solver = Solver(
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
                solver_panel = SolverPanel(
                    solver, renderer,
                    max_iterations = physicseditor.max_iterations,
                    viz_interval   = physicseditor.viz_interval,
                )
                # Live field preview: same colored-mesh renderer the post-solve
                # Visualizer uses, seeded with zero fields until the first
                # snapshot arrives from the solver thread.
                live_field = Visualizer(
                    renderer, vis_mesher,
                    np.zeros(mesh_data['Nc']),
                    np.zeros((mesh_data['Nc'], 2)),
                    mesh_data=mesh_data,
                )
                current_state = "SOLVING"

        # SOLVING → VISUALIZER (when user clicks "Open Visualizer")
        elif current_state == "SOLVING" and solver_panel.finished:
            # Reuse the live preview's geometry/VBOs — just refresh with the
            # final fields instead of allocating a second Visualizer.
            live_field.update_fields(solver.P, solver.U,
                                     res_cont=solver.final_res_cont,
                                     res_mom=solver.final_res_mom)
            live_field.update_vbo_colors()
            visualizer    = live_field
            live_field    = None
            solver_panel  = None
            current_state = "VISUALIZER"

        # VISUALIZER → PHYSICS  (state preserved — mesh/BCs intact, re-solve available)
        elif current_state == "VISUALIZER" and visualizer.finished:
            visualizer.destroy()   # free this cycle's pos/color/vector VBOs
            visualizer = None
            # Mesh wireframe VBOs (`vbos`) are kept so the mesh stays visible
            # in Physics Editor — only the Visualizer's own buffers are freed.
            current_state = "PHYSICS"

        # ------------------------------------------------------------------
        # Rendering
        # ------------------------------------------------------------------
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        if current_state == "EDITOR":
            editor.draw(screen, camera)

        elif current_state == "PHYSICS":
            physicseditor.draw(screen, camera, vbos)

        elif current_state == "SOLVING":
            # Pick up the latest live field snapshot (if any arrived this
            # frame) and push it to the live-preview color VBO.
            if solver_panel.viz_snapshot is not None:
                snap = solver_panel.viz_snapshot
                solver_panel.viz_snapshot = None
                live_field.update_fields(snap['P'], snap['U'],
                                         res_cont=snap.get('res_cont'),
                                         res_mom=snap.get('res_mom'))
                live_field.update_vbo_colors()

            live_field.draw_geometry(camera)
            if vbos and 'walls' in vbos:
                camera.draw_vbo(vbos['walls'][0], vbos['walls'][1], color=(255, 255, 255))
            solver_panel.draw(screen, camera, live_field=live_field)

        elif current_state == "VISUALIZER":
            visualizer.draw(screen, camera, dt)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_app()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)
