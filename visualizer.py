import pygame
from OpenGL.GL import *
import imgui
import numpy as np
from scipy.spatial import cKDTree # Efficient spatial searching

class Visualizer:
    def __init__(self, renderer, mesher, P, U, res_cont=None, res_mom=None):
        self.renderer = renderer
        self.mesher = mesher
        self.P = P
        self.U = U
        self.U_mag = np.linalg.norm(U, axis=1)
        
        # === NEW: Capture local cell-by-cell residuals ===
        self.res_cont = res_cont if res_cont is not None else np.zeros_like(P)
        self.res_mom = res_mom if res_mom is not None else np.zeros_like(P)
        
        # === CHANGED: Added diagnostic options into your existing variables ===
        self.vars = ["Pressure", "Velocity", "Continuity Error", "Momentum Error"]
        self.var_idx = 0
        self.last_var_idx = -1
        self.finished = False

        # Vector Glyph Settings
        self.show_vectors = False
        self.vector_scale = 1
        self.vector_color = (1.0, 1.0, 1.0)

        # GPU Buffer IDs
        self.pos_vbo = glGenBuffers(1)
        self.color_vbo = glGenBuffers(1)
        self.vector_vbo = glGenBuffers(1)
        self.vertex_count = 0
        self.vector_vertex_count = 0

        # Spatial Indexing Data
        self.centroids = []
        self.cell_data_map = [] 
        self.tree = None

        self._setup_geometry_vbo()
        self._update_vector_vbo()
        
        # Build the KDTree for the probe
        self.tree = cKDTree(self.centroids)

    def _setup_geometry_vbo(self):
        """Flattens cells and stores centroids for the spatial index."""
        vertices = []
        self.cell_vertex_counts = []
        centroids = []
        self.cell_data_map = []

        # Process all cells (boundary and internal)
        all_cells = [c for c in self.mesher.boundary_elements if float(c.area) > 1e-8]
        all_cells.extend(self.mesher.triangulation.triangles)

        for i, cell in enumerate(all_cells):
            pts = cell.vertices()
            # Calculate Centroid
            cx = sum(p.x for p in pts) / len(pts)
            cy = sum(p.y for p in pts) / len(pts)
            centroids.append([cx, cy])
            self.cell_data_map.append(cell)

            if len(pts) == 4: # Quad
                vertices.extend([(pts[0].x, pts[0].y), (pts[1].x, pts[1].y), (pts[2].x, pts[2].y)])
                vertices.extend([(pts[0].x, pts[0].y), (pts[2].x, pts[2].y), (pts[3].x, pts[3].y)])
                self.cell_vertex_counts.append(6)
            else: # Triangle
                vertices.extend([(pts[0].x, pts[0].y), (pts[1].x, pts[1].y), (pts[2].x, pts[2].y)])
                self.cell_vertex_counts.append(3)

        v_array = np.array(vertices, dtype=np.float32)
        self.centroids = np.array(centroids, dtype=np.float32)
        self.vertex_count = len(v_array)

        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        glBufferData(GL_ARRAY_BUFFER, v_array.nbytes, v_array, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _is_point_in_cell(self, px, py, cell):
        """Checks if point (px, py) is inside a convex polygon, regardless of winding."""
        pts = cell.vertices()
        n = len(pts)
        signs = []
        
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            
            # 2D Cross Product: (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
            val = (p2.x - p1.x) * (py - p1.y) - (p2.y - p1.y) * (px - p1.x)
            
            # Exact edge hits count as inside
            if abs(val) < 1e-9:
                continue
                
            signs.append(val > 0)

        # If all edges produced the same sign, the point is inside
        return all(signs) or not any(signs)

    def _update_vector_vbo(self):
        endpoints = self.centroids + (self.U * self.vector_scale)
        vector_lines = np.empty((len(self.centroids) * 2, 2), dtype=np.float32)
        vector_lines[0::2] = self.centroids
        vector_lines[1::2] = endpoints
        self.vector_vertex_count = len(vector_lines)
        glBindBuffer(GL_ARRAY_BUFFER, self.vector_vbo)
        glBufferData(GL_ARRAY_BUFFER, vector_lines.nbytes, vector_lines, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_vbo_colors(self):
        """Maps physical fields or mathematical log-scaled residuals to vertex colors."""
        # === CHANGED: Map variables based on combo selection ===
        if self.var_idx == 0:
            data = self.P
            is_residual = False
        elif self.var_idx == 1:
            data = self.U_mag
            is_residual = False
        elif self.var_idx == 2:
            data = self.res_cont
            is_residual = True
        elif self.var_idx == 3:
            data = self.res_mom
            is_residual = True

        if is_residual:
            # Residual values span orders of magnitude; use Log10 so variations aren't washed out
            # We clip at 1e-10 to prevent log10(0) explosion errors
            log_data = np.log10(np.maximum(data, 1e-10))
            d_min, d_max = log_data.min(), log_data.max()
            denom = d_max - d_min if d_max != d_min else 1.0
            f = np.clip((log_data - d_min) / denom, 0, 1)
        else:
            # Standard linear interpolation for regular physical fields
            d_min, d_max = np.nanmin(data), np.nanmax(data)
            denom = d_max - d_min if d_max != d_min else 1.0
            f = np.clip((data - d_min) / denom, 0, 1)

        # Your beautiful native color map math
        r = np.clip(np.minimum(4 * f - 1.5, -4 * f + 4.5), 0, 1)
        g = np.clip(np.minimum(4 * f - 0.5, -4 * f + 3.5), 0, 1)
        b = np.clip(np.minimum(4 * f + 0.5, -4 * f + 2.5), 0, 1)
        
        rgb = np.stack([r, g, b], axis=1).astype(np.float32)
        expanded_colors = np.repeat(rgb, self.cell_vertex_counts, axis=0)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
        glBufferData(GL_ARRAY_BUFFER, expanded_colors.nbytes, expanded_colors, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, screen, camera):
        if self.var_idx != self.last_var_idx:
            self.update_vbo_colors()
            self.last_var_idx = self.var_idx

        camera.apply_gl_transform()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        glVertexPointer(2, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
        glColorPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glDisableClientState(GL_COLOR_ARRAY)

        if self.show_vectors:
            glColor3f(*self.vector_color)
            glBindBuffer(GL_ARRAY_BUFFER, self.vector_vbo)
            glVertexPointer(2, GL_FLOAT, 0, None)
            glDrawArrays(GL_LINES, 0, self.vector_vertex_count)

        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        camera.remove_gl_transform()

        # --- UI and Probing ---
        imgui.new_frame()
        
        # Point Probe Logic
        if not imgui.get_io().want_capture_mouse:
            m_pos = pygame.mouse.get_pos()
            world_m = camera.screen_to_world(m_pos)
            
            dist, indices = self.tree.query([world_m.x, world_m.y], k=5)
            
            for idx in indices:
                cell = self.cell_data_map[idx]
                if self._is_point_in_cell(world_m.x, world_m.y, cell):
                    imgui.set_next_window_position(m_pos[0] + 15, m_pos[1] + 15)
                    imgui.begin("Probe", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
                    imgui.text(f"P: {self.P[idx]:.4e} Pa")
                    imgui.text(f"U: {self.U_mag[idx]:.4f} m/s")
                    imgui.text(f"Ux: {self.U[idx,0]:.4f}")
                    imgui.text(f"Uy: {self.U[idx,1]:.4f}")
                    imgui.separator()
                    # === NEW: Show live localized mathematical errors on hover ===
                    imgui.text(f"Cont Err: {self.res_cont[idx]:.4e}")
                    imgui.text(f"Mom Err:  {self.res_mom[idx]:.4e}")
                    imgui.end()
                    break

        # Main Control Window
        imgui.set_next_window_position(10, 10, imgui.ALWAYS)
        imgui.set_next_window_size(300, 260) # Expanded slightly for visibility
        imgui.begin("Post-Processor", True)
        _, self.var_idx = imgui.combo("Visualize", self.var_idx, self.vars)
        imgui.separator()
        changed, self.show_vectors = imgui.checkbox("Show Velocity Vectors", self.show_vectors)
        if self.show_vectors:
            changed_scale, self.vector_scale = imgui.slider_float("Vector Scale", self.vector_scale, 0.01, 50.0)
            if changed_scale: self._update_vector_vbo()
        imgui.separator()
        
        # === CHANGED: Dynamic range tracking for residuals vs fields ===
        if self.var_idx == 0: data = self.P
        elif self.var_idx == 1: data = self.U_mag
        elif self.var_idx == 2: data = self.res_cont
        elif self.var_idx == 3: data = self.res_mom
        
        imgui.text(f"Range: {np.nanmin(data):.2e} to {np.nanmax(data):.2e}")
        if imgui.button("Return to Editor", width=-1, height=30): self.finished = True
        imgui.end()
        
        imgui.render()
        self.renderer.render(imgui.get_draw_data())