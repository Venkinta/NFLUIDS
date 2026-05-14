import pygame
from OpenGL.GL import *
import imgui
import numpy as np

class Visualizer:
    def __init__(self, renderer, mesher, P, U):
        self.renderer = renderer
        self.mesher = mesher
        self.P = P
        # U is likely [Nc, 2], get magnitude for color mapping
        self.U_mag = np.linalg.norm(U, axis=1)
        
        self.current_var = "Pressure"
        self.vars = ["Pressure", "Velocity"]
        self.var_idx = 0
        self.finished = False

    def get_color(self, val, min_val, max_val):
        """Maps a value to a Jet color scale (Blue -> Red)"""
        if np.isnan(val) or np.isinf(val):
            return (1.0, 1.0, 1.0) # White for bad data
        
        if max_val == min_val:
            return (0.5, 0.5, 0.5)
        
        f = (val - min_val) / (max_val - min_val + 1e-9)
        f = np.clip(f, 0, 1)
        
        # Jet Approximation
        r = np.clip(min(4 * f - 1.5, -4 * f + 4.5), 0, 1)
        g = np.clip(min(4 * f - 0.5, -4 * f + 3.5), 0, 1)
        b = np.clip(min(4 * f + 0.5, -4 * f + 2.5), 0, 1)
        return (r, g, b)

    def draw(self, screen, camera):
        data = self.P if self.current_var == "Pressure" else self.U_mag
        d_min, d_max = np.nanmin(data), np.nanmax(data)

        # 1. Valid Boundary Elements
        valid_boundary_elements = [c for c in self.mesher.boundary_elements if float(c.area) > 1e-8]
        
        idx = 0
        
        # Draw Quads and Triangles from boundary layers
        for cell in valid_boundary_elements:
            color = self.get_color(data[idx], d_min, d_max)
            glColor3f(*color)
            
            glBegin(GL_POLYGON)
            for p in cell.vertices():
                # Use to_screen as defined in camera.py
                pos = camera.to_screen(p) 
                glVertex2f(pos[0], pos[1])
            glEnd()
            idx += 1

        # 2. Draw Internal Triangles
        glBegin(GL_TRIANGLES)
        for tri in self.mesher.triangulation.triangles:
            color = self.get_color(data[idx], d_min, d_max)
            glColor3f(*color)
            for p in tri.vertices():
                # Use to_screen as defined in camera.py
                pos = camera.to_screen(p)
                glVertex2f(pos[0], pos[1])
            idx += 1
        glEnd()

        # UI Overlay
        imgui.new_frame()
        imgui.set_next_window_position(10, 10, imgui.ALWAYS)
        imgui.set_next_window_size(300, 150)
        imgui.begin("Post-Processor", True)
        
        changed, self.var_idx = imgui.combo("Visualize", self.var_idx, self.vars)
        if changed:
            self.current_var = self.vars[self.var_idx]

        imgui.text(f"Range: {d_min:.2e} to {d_max:.2e}")
        
        if imgui.button("Return to Editor"):
            self.finished = True
            
        imgui.end()
        imgui.render()
        self.renderer.render(imgui.get_draw_data())