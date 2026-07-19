import pygame
from OpenGL.GL import *
import imgui
import numpy as np
from scipy.spatial import cKDTree # Efficient spatial searching
from .smoke_particles import SmokeParticles
from .renderer import VboHandle

class Visualizer:
    def __init__(self, mesher, P, U, res_cont=None, res_mom=None, mesh_data=None):
        self.mesher = mesher
        # solver_data_pipeline() dict (boundary_tags/Cf/unit_to_meters) — used
        # by SmokeParticles to find velocity-inlet faces to seed from. Not
        # used for rendering/geometry (that's still `mesher` above).
        self.mesh_data = mesh_data
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

        # GPU buffers (positions are static; colors/vectors re-upload on change)
        self.pos_vbo = VboHandle(components=2)
        self.color_vbo = VboHandle(components=3, usage=GL_DYNAMIC_DRAW)
        self.vector_vbo = VboHandle(components=2, usage=GL_DYNAMIC_DRAW)

        # Spatial Indexing Data
        self.centroids = []
        self.cell_data_map = []
        self.tree = None

        self._setup_geometry_vbo()
        self._update_vector_vbo()

        # Build the KDTree for the probe
        self.tree = cKDTree(self.centroids)

        self.show_particles = False
        self.smoke = SmokeParticles(owner=self)

    def _setup_geometry_vbo(self):
        """Flattens cells and stores centroids for the spatial index.

        Two geometry sources are supported:
          * a `Mesher` object (live solve) — uses its cell polygons directly;
          * a solver-data dict (loaded .npz) — uses the `cell_vertices` /
            `cell_nverts` / `cell_centers` arrays that ride along in the file.
        """
        vertices = []
        self.cell_vertex_counts = []
        centroids = []
        self.cell_data_map = []

        if isinstance(self.mesher, dict):
            # Loaded mesh: geometry comes from the saved dict (SI metres).
            cv = self.mesher['cell_vertices']   # (Nc, max_verts, 2)
            nv = self.mesher['cell_nverts']      # (Nc,)
            cc = self.mesher['cell_centers']     # (Nc, 2)
            for i in range(len(nv)):
                n = int(nv[i])
                pts = [(float(cv[i, v, 0]), float(cv[i, v, 1])) for v in range(n)]
                centroids.append([cc[i, 0], cc[i, 1]])
                self.cell_data_map.append(pts)  # store polygon for probe test
                if n == 4:  # Quad
                    vertices.extend([pts[0], pts[1], pts[2]])
                    vertices.extend([pts[0], pts[2], pts[3]])
                    self.cell_vertex_counts.append(6)
                else:       # Triangle
                    vertices.extend([pts[0], pts[1], pts[2]])
                    self.cell_vertex_counts.append(3)
        else:
            # Live mesh: use the Mesher's cell objects directly.
            all_cells = [c for c in self.mesher.boundary_elements if float(c.area) > 1e-8]
            all_cells.extend(self.mesher.triangulation.triangles)
            for cell in all_cells:
                pts = cell.vertices()
                # Use the cell's own centroid (shoelace for quads, average for
                # triangles) so the probe location matches the solver's geometry.
                c = cell.centroid
                centroids.append([c.x, c.y])
                self.cell_data_map.append(cell)
                if len(pts) == 4:  # Quad
                    vertices.extend([(pts[0].x, pts[0].y), (pts[1].x, pts[1].y), (pts[2].x, pts[2].y)])
                    vertices.extend([(pts[0].x, pts[0].y), (pts[2].x, pts[2].y), (pts[3].x, pts[3].y)])
                    self.cell_vertex_counts.append(6)
                else:  # Triangle
                    vertices.extend([(pts[0].x, pts[0].y), (pts[1].x, pts[1].y), (pts[2].x, pts[2].y)])
                    self.cell_vertex_counts.append(3)

        v_array = np.array(vertices, dtype=np.float32)
        self.centroids = np.array(centroids, dtype=np.float32)
        self.pos_vbo.upload(v_array)

    def _is_point_in_cell(self, px, py, cell):
        """Checks if point (px, py) is inside a convex polygon, regardless of winding.

        `cell` is either a Mesher cell object (has .vertices()) or, for a loaded
        mesh, a plain list of (x, y) tuples stored in cell_data_map.
        """
        if hasattr(cell, "vertices"):
            pts = [(p.x, p.y) for p in cell.vertices()]
        else:
            pts = cell
        n = len(pts)
        signs = []

        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]

            # 2D Cross Product: (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
            val = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

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
        self.vector_vbo.upload(vector_lines)

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

        # Robust (percentile-clipped) range instead of true min/max: a single
        # outlier cell (e.g. a stagnation point) would otherwise hijack the
        # whole field's color scale and make everything else look like it's
        # swimming from frame to frame even when it barely changed.
        _LO_PCT, _HI_PCT = 2, 98

        if is_residual:
            # Residual values span orders of magnitude; use Log10 so variations aren't washed out
            # We clip at 1e-10 to prevent log10(0) explosion errors
            log_data = np.log10(np.maximum(data, 1e-10))
            d_min, d_max = np.nanpercentile(log_data, [_LO_PCT, _HI_PCT])
            denom = d_max - d_min if d_max != d_min else 1.0
            f = np.clip((log_data - d_min) / denom, 0, 1)
        else:
            # Standard linear interpolation for regular physical fields
            d_min, d_max = np.nanpercentile(data, [_LO_PCT, _HI_PCT])
            denom = d_max - d_min if d_max != d_min else 1.0
            f = np.clip((data - d_min) / denom, 0, 1)

        # Your beautiful native color map math
        r = np.clip(np.minimum(4 * f - 1.5, -4 * f + 4.5), 0, 1)
        g = np.clip(np.minimum(4 * f - 0.5, -4 * f + 3.5), 0, 1)
        b = np.clip(np.minimum(4 * f + 0.5, -4 * f + 2.5), 0, 1)
        
        rgb = np.stack([r, g, b], axis=1).astype(np.float32)
        expanded_colors = np.repeat(rgb, self.cell_vertex_counts, axis=0)
        self.color_vbo.upload(expanded_colors)

    def update_fields(self, P, U, res_cont=None, res_mom=None):
        """Swap in new field data (e.g. a live solve snapshot or the final
        result) without rebuilding geometry. Caller must call
        update_vbo_colors() afterward to push the new colors to the GPU."""
        self.P = P
        self.U = U
        self.U_mag = np.linalg.norm(U, axis=1)
        if res_cont is not None:
            self.res_cont = res_cont
        if res_mom is not None:
            self.res_mom = res_mom

    def destroy(self):
        """Free GPU buffers. Call before dropping the last reference,
        since a re-solve allocates a brand-new Visualizer/live preview."""
        self.pos_vbo.delete()
        self.color_vbo.delete()
        self.vector_vbo.delete()
        self.smoke.destroy()

    def restore_display_settings(self, data):
        """Apply saved var_idx/show_vectors/vector_scale from a loaded
        visualization dict. np.savez turns Python scalars into 0-d arrays,
        same reasoning as meshIO.load_mesh_for_solver's Nc/Nf casting."""
        if 'var_idx' in data:
            self.var_idx = int(data['var_idx'])
        if 'show_vectors' in data:
            self.show_vectors = bool(data['show_vectors'])
        if 'vector_scale' in data:
            self.vector_scale = float(data['vector_scale'])
        self._update_vector_vbo()
        self.update_vbo_colors()
        self.last_var_idx = self.var_idx

    def open_save_dialog(self):
        """Opens a native OS file dialog to save this visualization (mesh +
        solved fields + a few display settings) as a .npz, mirroring
        PhysicsEditor.open_save_dialog()."""
        import tkinter as tk
        from tkinter import filedialog
        from . import meshIO

        if not self.mesh_data:
            print("[UI] No mesh data available to save this visualization.")
            return

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy Compressed Archive", "*.npz"), ("All Files", "*.*")],
            title="Export Visualization"
        )

        root.destroy()

        if filepath:
            print(f"[UI] User selected visualization save path: {filepath}")
            combined = dict(self.mesh_data)
            combined.update({
                'P': self.P, 'U': self.U,
                'res_cont': self.res_cont, 'res_mom': self.res_mom,
                'var_idx': self.var_idx, 'show_vectors': self.show_vectors,
                'vector_scale': self.vector_scale,
            })
            meshIO.save_mesh_for_solver(combined, filepath)

    def open_export_vtu_dialog(self):
        """Opens a native OS file dialog to export this visualization as a
        VTK XML UnstructuredGrid (.vtu) — for cross-validating the solver
        against other CFD codes (e.g. opening the result in ParaView)."""
        import tkinter as tk
        from tkinter import filedialog
        from . import vtuIO

        if not self.mesh_data:
            print("[UI] No mesh data available to export.")
            return

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        filepath = filedialog.asksaveasfilename(
            defaultextension=".vtu",
            filetypes=[("VTK UnstructuredGrid", "*.vtu"), ("All Files", "*.*")],
            title="Export VTU"
        )

        root.destroy()

        if filepath:
            print(f"[UI] User selected VTU export path: {filepath}")
            vtuIO.export_vtu(
                self.mesh_data, filepath,
                P=self.P, U=self.U,
                res_cont=self.res_cont, res_mom=self.res_mom,
            )

    def draw_geometry(self, gfx):
        """Just the colored mesh fill — no ImGui overlay. Used both by the
        full post-processor draw() below and by the live solve preview."""
        gfx.draw_vbo_colored(self.pos_vbo, self.color_vbo)

    def draw(self, gfx, dt):
        if self.var_idx != self.last_var_idx:
            self.update_vbo_colors()
            self.last_var_idx = self.var_idx

        self.draw_geometry(gfx)

        if self.show_vectors:
            gfx.draw_vbo(self.vector_vbo,
                         color=tuple(c * 255 for c in self.vector_color),
                         mode=GL_LINES)

        if self.show_particles:
            self.smoke.step(dt)
            self.smoke.draw(gfx)

        # --- UI and Probing ---
        # Point Probe Logic
        if not imgui.get_io().want_capture_mouse:
            m_pos = pygame.mouse.get_pos()
            world_m = gfx.camera.screen_to_world(m_pos)
            
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
        imgui.set_next_window_size(300, 495) # Expanded for vectors + particle + save + export controls
        imgui.begin("Post-Processor", True)
        _, self.var_idx = imgui.combo("Visualize", self.var_idx, self.vars)
        imgui.separator()
        changed, self.show_vectors = imgui.checkbox("Show Velocity Vectors", self.show_vectors)
        if self.show_vectors:
            changed_scale, self.vector_scale = imgui.slider_float("Vector Scale", self.vector_scale, 0.01, 50.0)
            if changed_scale: self._update_vector_vbo()
        imgui.separator()
        _, self.show_particles = imgui.checkbox("Show Smoke Particles", self.show_particles)
        if self.show_particles:
            _, self.smoke.speed_scale = imgui.slider_float("Particle Speed", self.smoke.speed_scale, 0.05, 50.0)
            _, self.smoke.point_size = imgui.slider_float("Particle Size", self.smoke.point_size, 1.0, 10.0)
            changed_count, new_count = imgui.slider_int("Particle Count", self.smoke.count, 50, 5000)
            if changed_count: self.smoke.set_count(new_count)
            _, self.smoke.limit_lifetime = imgui.checkbox("Limit Particle Lifetime", self.smoke.limit_lifetime)
            if self.smoke.limit_lifetime:
                _, self.smoke.lifetime = imgui.slider_float("Lifetime (s)", self.smoke.lifetime, 1.0, 30.0)
        imgui.separator()
        
        # === CHANGED: Dynamic range tracking for residuals vs fields ===
        if self.var_idx == 0: data = self.P
        elif self.var_idx == 1: data = self.U_mag
        elif self.var_idx == 2: data = self.res_cont
        elif self.var_idx == 3: data = self.res_mom
        
        imgui.text(f"Range: {np.nanmin(data):.2e} to {np.nanmax(data):.2e}")
        if imgui.button("Save Visualization", width=-1, height=30): self.open_save_dialog()
        if imgui.button("Export VTU", width=-1, height=30): self.open_export_vtu_dialog()
        if imgui.button("Back to Physics", width=-1, height=30): self.finished = True
        imgui.end()