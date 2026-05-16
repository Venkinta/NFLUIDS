import pygame
import math
from line import Line
from editor import Editor
from snapengine import SnapEngine
from pygame_widgets.button import Button
import pygame_widgets
from mesher import Mesher
from solver import Solver
from quad import Quad
from camera import Camera
from physics_editor import PhysicsEditor
import OpenGL
OpenGL.ERROR_CHECKING = False   # eliminates ~10M glCheckError calls per session
OpenGL.ERROR_ON_COPY = False
from OpenGL.GL import *
import imgui
from imgui.integrations.pygame import PygameRenderer
from visualizer import Visualizer

import cProfile
import pstats

def run_app():
    pygame.init()
    WIDTH, HEIGHT = 1280, 720
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

    # State objects
    editor = Editor(screen, renderer)
    physicseditor = None
    mesher = None
    visualizer = None

    current_state = "EDITOR"
    running = True
    dt = 1 / 60
    accumulator = 0.0

    camera = Camera()
    
    # --- CHANGED: Dictionary to hold multiple VBOs ---
    vbos = {} 

    while running:
        frame_time = clock.tick(60) / 1000.0
        accumulator += frame_time

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEWHEEL:
                camera.handle_zoom(pygame.mouse.get_pos(), event.y)

            if current_state == "EDITOR":
                renderer.process_event(event)
                if not imgui.get_io().want_capture_mouse:
                    editor.handle_event(event, camera)
            elif current_state == "PHYSICS":
                physicseditor.renderer.process_event(event)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not imgui.get_io().want_capture_mouse:
                        physicseditor.handle_selection(camera.screen_to_world(event.pos),camera)
            elif current_state == "MESHER":
                mesher.renderer.process_event(event)
            elif current_state == "VISUALIZER":
                visualizer.renderer.process_event(event)

        # State transitions
        if current_state == "EDITOR" and editor.finished:
            physicseditor = PhysicsEditor(screen, editor.lines, renderer, editor.unit_idx)
            current_state = "PHYSICS"

        elif current_state == "PHYSICS" and physicseditor.finished:
            mesher = Mesher(
                screen, physicseditor.lines, physicseditor.n_layers,
                physicseditor.growth_factor, physicseditor.thickness,
                physicseditor.boundary_spacing, physicseditor.r, renderer,
                unit_to_meters=physicseditor.unit_to_meters
            )
            mesher.mesh()
            
            # --- GENERATE MULTIPLE VBOs ---
            mesh_bundles = mesher.get_render_data() 
            vbos = {} 
            for key, (data, count) in mesh_bundles.items():
                if count > 0:
                    vbo_id = glGenBuffers(1)
                    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
                    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
                    vbos[key] = (vbo_id, count)
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            current_state = "MESHER"

        elif current_state == "MESHER" and mesher.finished:
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
            visualizer = Visualizer(renderer, mesher, solver.P, solver.U)
            current_state = "VISUALIZER"

        elif current_state == "VISUALIZER" and visualizer.finished:
            editor = Editor(screen, renderer)
            current_state = "EDITOR"

        while accumulator >= dt:
            accumulator -= dt

        # Rendering
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        if current_state == "EDITOR":
            editor.draw(screen, camera)
        elif current_state == "PHYSICS":
            physicseditor.draw(screen, camera)
        elif current_state == "MESHER":
            # Pass the entire vbo dictionary to mesher.draw
            mesher.draw(screen, camera, vbos) 
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