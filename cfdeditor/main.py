import math
from .line import Line
from .editor import Editor
from .snapengine import SnapEngine
from pygame_widgets.button import Button
import pygame_widgets
from .mesher import Mesher
from .solver import Solver
from .quad import Quad
from .camera import Camera
from .physics_editor import PhysicsEditor
import pygame
import OpenGL
OpenGL.ERROR_CHECKING = False   # eliminates glCheckError calls
OpenGL.ERROR_ON_COPY = False
from OpenGL.GL import *
import imgui
from imgui.integrations.pygame import PygameRenderer
from .visualizer import Visualizer
from .point import Point

import numpy as np
import cProfile
import pstats



def run_app():
    # Open pygame with default resolution and OPENGL
    pygame.init()
    WIDTH, HEIGHT = 1920, 1080
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)

    #OPENGL 
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

    # State objects. These serve as initializers for our different modules in the future
    editor = Editor(screen, renderer)
    physicseditor = None
    mesher = None
    visualizer = None

    #Initialize the EDITOR state, start the loop and set frames to 60fps. 
    current_state = "EDITOR"
    running = True
    dt = 1 / 60
    accumulator = 0.0

    #Initialize our camera class, handler of all rendering. 
    camera = Camera()
    
    # --- Dictionary to hold multiple VBOs ---
    vbos = {} 


    #Initialize main loop
    while running:
        #Frame logic
        frame_time = clock.tick(60) / 1000.0
        accumulator += frame_time

        """Pygame logic: Each frame you get events with pygame.event.get check documentation for full list. 
            Simple loop logic: If X button, end the program"""
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

            # Camera handles rendering and zoom
            if event.type == pygame.MOUSEWHEEL:
                camera.handle_zoom(pygame.mouse.get_pos(), event.y)

            # MODULE logic handler each module switches when its own .finish() is called.
            if current_state == "EDITOR":
                renderer.process_event(event)
                if not imgui.get_io().want_capture_mouse:
                    editor.handle_event(event, camera)
            
            # Physics editor launches after the editor. It is responsible for controlling boundary conditions,
            # meshing and solving parameters (and now also refinement zone drawing).
            elif current_state == "PHYSICS":
                physicseditor.renderer.process_event(event)
                physicseditor.handle_event(event, camera)
            #visualizer is a simple visual way to check results and residuals
            elif current_state == "VISUALIZER":
                visualizer.renderer.process_event(event)


        # State transitions. Hands off data from one module to another 
        if current_state == "EDITOR" and editor.finished:
            #Needs to pass off all lines, the renderer object and the unit id for the global units
            physicseditor = PhysicsEditor(screen, editor.lines, renderer, editor.unit_idx)
            current_state = "PHYSICS"

        elif current_state == "PHYSICS":
            # --- Mesh / Remesh ---
            if physicseditor.mesh_requested:
                physicseditor.mesh_requested = False
                # Convert refinement zone dicts to (shapely_polygon, factor) tuples
                refinement_zones = physicseditor._get_refinement_polygons()
                mesher = Mesher(
                    screen, physicseditor.lines, physicseditor.n_layers,
                    physicseditor.growth_factor, physicseditor.thickness,
                    physicseditor.boundary_spacing, physicseditor.r, renderer,
                    unit_to_meters=physicseditor.unit_to_meters,
                    refinement_zones=refinement_zones
                )
                mesher.mesh()

                # Free any previous VBOs before uploading new ones
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

            # --- Load a previously saved mesh (.npz) ---
            if physicseditor.load_requested:
                physicseditor.load_requested = False
                loaded = physicseditor.loaded_mesh

                # Reconstruct CAD lines from the saved dict so the user can
                # edit boundary conditions and remesh.
                bc_names = ["Wall", "Velocity Inlet", "Pressure Outlet"]
                cad = loaded['cad_lines']
                lines = []
                for row in cad:
                    ax, ay, bx, by, bc_idx = row
                    line = Line(Point(ax, ay), Point(bx, by))
                    line.boundary_type = bc_names[int(bc_idx)]
                    lines.append(line)
                physicseditor.lines = lines

                # Restore the meshing parameters so the UI reflects the values
                # that produced the saved mesh (instead of the defaults).
                physicseditor.n_layers         = int(loaded['n_layers'])
                physicseditor.growth_factor    = float(loaded['growth_factor'])
                physicseditor.thickness        = float(loaded['thickness'])
                physicseditor.boundary_spacing = float(loaded['boundary_spacing'])
                physicseditor.r                = float(loaded['r'])
                physicseditor.unit_to_meters   = float(loaded['unit_to_meters'])
                # Restore the World-units combo index to match the saved unit.
                _unit_factors = {"mm": 0.001, "cm": 0.01, "m": 1.0}
                for i, name in enumerate(["mm", "cm", "m"]):
                    if abs(_unit_factors[name] - physicseditor.unit_to_meters) < 1e-12:
                        physicseditor._unit_idx = i
                        break

                # Build coloured wireframe VBOs from cell_vertices in the
                # loaded dict. The dict stores SI metres; convert back to
                # world units for display (the camera renders in world units).
                cv = loaded['cell_vertices'] / physicseditor.unit_to_meters
                nv = loaded['cell_nverts']
                tri_coords, quad_coords = [], []
                for i in range(len(nv)):
                    n = int(nv[i])
                    pts = [(float(cv[i, v, 0]), float(cv[i, v, 1])) for v in range(n)]
                    if n == 4:  # Quad edges (8 vertices)
                        quad_coords.extend([pts[0], pts[1], pts[1], pts[2],
                                            pts[2], pts[3], pts[3], pts[0]])
                    else:        # Triangle edges (6 vertices)
                        tri_coords.extend([pts[0], pts[1], pts[1], pts[2],
                                           pts[2], pts[0]])

                # Free any previous VBOs before uploading new ones
                for _vbo_id, _ in vbos.values():
                    glDeleteBuffers(1, [_vbo_id])
                vbos = {}
                if tri_coords:
                    vbo_data = np.array(tri_coords, dtype=np.float32)
                    vbo_id = glGenBuffers(1)
                    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
                    glBufferData(GL_ARRAY_BUFFER, vbo_data.nbytes, vbo_data, GL_STATIC_DRAW)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)
                    vbos['triangles'] = (vbo_id, len(tri_coords))
                if quad_coords:
                    vbo_data = np.array(quad_coords, dtype=np.float32)
                    vbo_id = glGenBuffers(1)
                    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
                    glBufferData(GL_ARRAY_BUFFER, vbo_data.nbytes, vbo_data, GL_STATIC_DRAW)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)
                    vbos['quads'] = (vbo_id, len(quad_coords))

                physicseditor.has_mesh = True
                physicseditor.mesher = None   # no original Mesher object
                # REMAIN in PHYSICS — user sees the mesh and can adjust params / Solve

            # --- Solve ---
            if physicseditor.solve_requested:
                physicseditor.solve_requested = False
                if physicseditor.loaded_mesh is not None:
                    # Loaded mesh → no mesher object; use SOLVER_LOADED path
                    current_state = "SOLVER_LOADED"
                else:
                    current_state = "SOLVER"

        elif current_state == "SOLVER":
            solver = Solver(
                mesher.solver_data_pipeline(),
                [physicseditor.inlet_velocity, 0.0],
                physicseditor.outlet_pressure,
                physicseditor.density,
                physicseditor.viscosity,
            )
            solver.Solve()
            visualizer = Visualizer(renderer, mesher, solver.P, solver.U,
                                    res_cont=solver.final_res_cont, 
                                    res_mom=solver.final_res_mom)
            current_state = "VISUALIZER"

        elif current_state == "SOLVER_LOADED":
            # No mesher available — geometry comes from the loaded dict itself.
            loaded = physicseditor.loaded_mesh
            solver = Solver(
                loaded,
                [physicseditor.inlet_velocity, 0.0],
                physicseditor.outlet_pressure,
                physicseditor.density,
                physicseditor.viscosity,
            )
            solver.Solve()
            # Convert cell_vertices back to world units for the Visualizer
            # (the camera renders in world units, but the dict stores SI metres).
            vis_dict = dict(loaded)
            vis_dict['cell_vertices'] = loaded['cell_vertices'] / physicseditor.unit_to_meters
            visualizer = Visualizer(renderer, vis_dict, solver.P, solver.U,
                                    res_cont=solver.final_res_cont,
                                    res_mom=solver.final_res_mom)
            current_state = "VISUALIZER"

        elif current_state == "VISUALIZER" and visualizer.finished:
            editor = Editor(screen, renderer)
            physicseditor = None
            mesher = None
            for _vbo_id, _ in vbos.values():
                glDeleteBuffers(1, [_vbo_id])
            vbos = {}
            current_state = "EDITOR"

        # NOTE: the fixed-update accumulator is intentionally left as a no-op
        # for now (no fixed-timestep physics step is implemented yet).

        # Rendering
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        if current_state == "EDITOR":
            editor.draw(screen, camera)
        elif current_state == "PHYSICS":
            physicseditor.draw(screen, camera, vbos)
        elif current_state == "SOLVER":
            if vbos:
                # Interior Triangles (Blue)
                if 'triangles' in vbos:
                    camera.draw_vbo(vbos['triangles'][0], vbos['triangles'][1], color=(0, 100, 255))
                # Boundary Quads (Green)
                if 'quads' in vbos:
                    camera.draw_vbo(vbos['quads'][0], vbos['quads'][1], color=(0, 255, 100))
                # CAD Walls (White)
                if 'walls' in vbos:
                    camera.draw_vbo(vbos['walls'][0], vbos['walls'][1], color=(255, 255, 255))
        elif current_state == "SOLVER_LOADED":
            if vbos:
                # Interior Triangles (Blue)
                if 'triangles' in vbos:
                    camera.draw_vbo(vbos['triangles'][0], vbos['triangles'][1], color=(0, 100, 255))
                # Boundary Quads (Green)
                if 'quads' in vbos:
                    camera.draw_vbo(vbos['quads'][0], vbos['quads'][1], color=(0, 255, 100))
                # CAD Walls (White)
                if 'walls' in vbos:
                    camera.draw_vbo(vbos['walls'][0], vbos['walls'][1], color=(255, 255, 255))
        elif current_state == "VISUALIZER":
            visualizer.draw(screen, camera)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_app()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)