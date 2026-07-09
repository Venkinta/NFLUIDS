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

import cProfile
import pstats



def run_app():
    # Open pygame with default resolution and OPENGL
    pygame.init()
    WIDTH, HEIGHT = 1280, 720
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
            # meshing and solving parameters. 
            elif current_state == "PHYSICS":
                physicseditor.renderer.process_event(event)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not imgui.get_io().want_capture_mouse:
                        physicseditor.handle_selection(camera.screen_to_world(event.pos),camera)
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
                mesher = Mesher(
                    screen, physicseditor.lines, physicseditor.n_layers,
                    physicseditor.growth_factor, physicseditor.thickness,
                    physicseditor.boundary_spacing, physicseditor.r, renderer,
                    unit_to_meters=physicseditor.unit_to_meters
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

            # --- Solve ---
            if physicseditor.solve_requested:
                physicseditor.solve_requested = False
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