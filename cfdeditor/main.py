from .editor import Editor
from .camera import Camera
from .renderer import Renderer, logo_overlay
from .app_state import AppContext, EVENT_HANDLERS, UPDATE_HANDLERS, RENDER_HANDLERS
import pygame
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_ON_COPY = False
from OpenGL.GL import *
import imgui
from imgui.integrations.pygame import PygameRenderer

import cProfile
import pstats


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

    camera = Camera()
    gfx = Renderer(camera, renderer, screen)
    gfx.add_overlay(logo_overlay)

    ctx = AppContext(
        screen=screen,
        renderer=renderer,
        camera=camera,
        gfx=gfx,
        editor=Editor(screen, renderer),
    )
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

            # Feed ImGui first so want_capture_mouse reflects this event
            # before any camera/module handling decides whether to react to it.
            # All states share the one PygameRenderer instance created above.
            renderer.process_event(event)

            want_mouse = imgui.get_io().want_capture_mouse

            if event.type == pygame.MOUSEWHEEL and not want_mouse:
                ctx.camera.handle_zoom(pygame.mouse.get_pos(), event.y)

            event_handler = EVENT_HANDLERS.get(ctx.state)
            if event_handler is not None:
                event_handler(ctx, event, want_mouse)

        ctx.state = UPDATE_HANDLERS[ctx.state](ctx)

        gfx.begin_frame()
        RENDER_HANDLERS[ctx.state](ctx, dt)
        gfx.end_frame()

    pygame.quit()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_app()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)
