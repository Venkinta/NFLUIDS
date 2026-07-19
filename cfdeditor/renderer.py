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
import math
from contextlib import contextmanager

import pygame
import imgui
from OpenGL.GL import *

from . import __version__


class VboHandle:
    """One GL array buffer with a uniform create/upload/delete lifecycle.

    components: floats per vertex (2 = xy positions, 3 = rgb colors).
    A class-level registry lets main.py free everything at shutdown via
    delete_all() — previously buffers were only ever reclaimed as a side
    effect of the next rebuild.
    """
    _live = []

    def __init__(self, components=2, usage=GL_STATIC_DRAW):
        self.components = components
        self.usage = usage
        self.id = None
        self.count = 0          # vertices in the last upload

    def upload(self, data):
        """Upload a flat float32 array (allocates the GL buffer on first use)."""
        if self.id is None:
            self.id = glGenBuffers(1)
            VboHandle._live.append(self)
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, self.usage)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.count = data.size // self.components

    def delete(self):
        """Free the GL buffer. Safe to call twice."""
        if self.id is not None:
            glDeleteBuffers(1, [self.id])
            if self in VboHandle._live:
                VboHandle._live.remove(self)
            self.id = None
            self.count = 0

    @classmethod
    def delete_all(cls):
        for handle in list(cls._live):
            handle.delete()


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

    # ------------------------------------------------------------------
    # Immediate-mode primitives (CPU to_screen path, screen-space pixels).
    # Line/circle colors are 0-255 RGB; draw_rect takes 0-1 RGBA because
    # its fills rely on alpha blending (GL_BLEND is enabled at init).
    # ------------------------------------------------------------------

    def draw_screen_line(self, p0, p1, color=(255, 255, 255), width=1):
        """Line between two screen-space (pixel) points."""
        r, g, b = [c / 255.0 for c in color]
        glLineWidth(width)
        glBegin(GL_LINES)
        glColor3f(r, g, b)
        glVertex2f(p0[0], p0[1])
        glVertex2f(p1[0], p1[1])
        glEnd()

    def draw_world_line(self, a, b, color=(255, 255, 255), width=1):
        """Line between two world-space points (Point or (x, y))."""
        self.draw_screen_line(self.camera.to_screen(a),
                              self.camera.to_screen(b), color, width)

    def draw_circle(self, center, radius, color, width=1):
        """Outline circle, screen-space center and pixel radius."""
        r, g, b = [c / 255.0 for c in color]
        glColor3f(r, g, b)
        glLineWidth(width)
        glBegin(GL_LINE_LOOP)
        for i in range(32):
            angle = 2 * math.pi * i / 32
            glVertex2f(center[0] + math.cos(angle) * radius,
                       center[1] + math.sin(angle) * radius)
        glEnd()

    def draw_rect(self, p1, p2, fill_rgba=None, outline_rgba=None,
                  outline_width=2):
        """Axis-aligned rectangle between screen-space corners p1 and p2,
        with optional translucent fill and/or outline (0-1 RGBA)."""
        if fill_rgba is not None:
            glColor4f(*fill_rgba)
            glBegin(GL_QUADS)
            glVertex2f(p1[0], p1[1])
            glVertex2f(p2[0], p1[1])
            glVertex2f(p2[0], p2[1])
            glVertex2f(p1[0], p2[1])
            glEnd()
        if outline_rgba is not None:
            glColor4f(*outline_rgba)
            glLineWidth(outline_width)
            glBegin(GL_LINE_LOOP)
            glVertex2f(p1[0], p1[1])
            glVertex2f(p2[0], p1[1])
            glVertex2f(p2[0], p2[1])
            glVertex2f(p1[0], p2[1])
            glEnd()

    # ------------------------------------------------------------------
    # VBO draws (world space via the GL modelview matrix)
    # ------------------------------------------------------------------

    @contextmanager
    def _world_transform(self):
        glPushMatrix()
        glScalef(self.camera.scale, self.camera.scale, 1.0)
        glTranslatef(self.camera.offset[0], self.camera.offset[1], 0.0)
        try:
            yield
        finally:
            glPopMatrix()

    def draw_vbo(self, handle, color=(100, 255, 100), mode=GL_LINES,
                 point_size=None):
        """Draw a position-only VboHandle (2 floats/vertex) in one flat
        color (0-255 RGB). mode: GL_LINES, GL_POINTS, ..."""
        if handle is None or handle.id is None or handle.count == 0:
            return
        r, g, b = [c / 255.0 for c in color]
        glColor3f(r, g, b)
        if point_size is not None:
            glPointSize(point_size)
        with self._world_transform():
            glEnableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, handle.id)
            glVertexPointer(2, GL_FLOAT, 0, None)
            glDrawArrays(mode, 0, handle.count)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDisableClientState(GL_VERTEX_ARRAY)

    def draw_vbo_colored(self, pos_handle, color_handle, mode=GL_TRIANGLES):
        """Draw a position VboHandle (2 floats/vertex) with per-vertex RGB
        from a second handle (3 floats/vertex). The vertex count comes from
        the position handle — the color handle must cover as many vertices."""
        if pos_handle is None or pos_handle.id is None or pos_handle.count == 0:
            return
        with self._world_transform():
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, pos_handle.id)
            glVertexPointer(2, GL_FLOAT, 0, None)
            glBindBuffer(GL_ARRAY_BUFFER, color_handle.id)
            glColorPointer(3, GL_FLOAT, 0, None)
            glDrawArrays(mode, 0, pos_handle.count)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)


def logo_overlay():
    """Persistent NFLUIDS stamp, top-left, in every state.

    Drawn on ImGui's foreground draw list so it sits above all windows
    (Solver Monitor and Post-Processor are both pinned near the top-left)
    without capturing any mouse input.
    """
    draw_list = imgui.get_foreground_draw_list()
    color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.45)
    draw_list.add_text(8, 6, color, f"NFLUIDS v{__version__}")
