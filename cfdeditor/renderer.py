"""Shared rendering engine.

Owns the frame lifecycle: clear -> state draw (world + screen GL) ->
overlays -> one ImGui render -> buffer flip.

Frame contract
--------------
run_app() calls ``gfx.begin_frame()`` once per frame, then the current
state's render handler, then ``gfx.end_frame()``. Modules build their
ImGui panels inside their draw() methods but must NEVER call
``imgui.new_frame()`` / ``imgui.render()`` themselves — the Renderer owns
that pair, along with the GL clear and the display flip.

Overlays registered with ``add_overlay(fn)`` run every frame in every
state, after the state's draw and just before the ImGui frame is
finalized — the hook for cross-cutting UI (logo stamp, an FPS counter,
etc.) that no single state owns.

Coordinate paths (deliberate, do not merge)
-------------------------------------------
Two world->screen paths coexist by design: immediate-mode helpers
pre-transform points on the CPU via ``camera.to_screen()``, while VBO
draws push the camera as a GL modelview matrix. Merging them risks
subtle pixel drift at high zoom for zero user value.
"""
import pygame
import imgui
from OpenGL.GL import *


class Renderer:
    """Owns the frame; holds the app's single Camera for world-space draws."""

    def __init__(self, camera, imgui_backend, screen):
        self.camera = camera            # the one Camera (view math lives there)
        self.backend = imgui_backend    # the one imgui PygameRenderer
        self.screen = screen
        self._overlays = []

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        imgui.new_frame()

    def end_frame(self):
        for fn in self._overlays:
            fn()
        imgui.render()
        self.backend.render(imgui.get_draw_data())
        pygame.display.flip()

    def add_overlay(self, fn):
        """Register fn() to be called every frame, in every state."""
        self._overlays.append(fn)
