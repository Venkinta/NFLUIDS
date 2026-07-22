import pygame
import imgui
from enum import Enum, auto
from .line import Line
from .point import Point
import math
import numpy as np
from shapely.geometry import Polygon as ShapelyPoly

# Conversion factors to SI (metres)
_UNIT_FACTORS = {"mm": 0.001, "cm": 0.01, "m": 1.0}


class PhysicsAction(Enum):
    """A one-shot intent raised by a PHYSICS-state UI action, read and
    cleared by main.py's PHYSICS transition handling. Replaces four
    independent request booleans so at most one action is ever pending."""
    MESH = auto()
    SMOOTH_MESH = auto()
    LOAD_MESH = auto()
    LOAD_VISUALIZATION = auto()
    SOLVE = auto()


class PhysicsEditor:

    def __init__(self, lines, initial_unit_idx=0):
        self.lines = lines

        # Meshing / solving intent (read and cleared by main.py)
        self.pending_action = None  # Optional[PhysicsAction]
        self.has_mesh = False

        # Reference to the Mesher instance (set by main.py after meshing) so
        # the Save dialog can reach solver_data_pipeline(). None until meshed.
        self.mesher = None

        # Loaded-mesh handoff: when a .npz is loaded, main.py consumes this
        # dict and skips the meshing step entirely.
        self.loaded_mesh = None

        # Loaded-visualization handoff: like loaded_mesh, but the .npz also
        # carries solved fields (P/U/...), so main.py jumps straight to the
        # VISUALIZER state instead of landing in PHYSICS for a re-solve.
        self.loaded_visualization = None

        # --- Mesh smoothing parameters (Smooth Mesh button) ---
        self.smooth_passes = 3
        self.smooth_relaxation = 0.5

        # --- Fluid Properties (always SI) ---
        self.density = 1.2       # kg/m3
        self.viscosity = 0.002   # Pa*s

        # --- Unit System ---
        self._unit_names = ["mm", "cm", "m"]
        self._unit_idx = initial_unit_idx 
        self.unit_to_meters = _UNIT_FACTORS[self._unit_names[self._unit_idx]]

        # --- Boundary Layer Settings (in world units) ---
        self.n_layers = 4
        self.growth_factor = 1.1
        self.thickness = 1.0
        self.boundary_spacing = 6.0

        # --- Per-BC boundary spacing (world units) ---
        # Initialised to the global default.  When _spacing_linked is True,
        # changing the global boundary_spacing updates all of these; when
        # unlinked, each BC type can have its own value.
        self._bc_spacing = {
            "Wall":            self.boundary_spacing,
            "Symmetry":        self.boundary_spacing,
            "Velocity Inlet":  self.boundary_spacing,
            "Pressure Outlet": self.boundary_spacing,
        }
        self._spacing_linked = True

        # --- Mesh Generation (in world units) ---
        self.r = 4.0

        # --- Refinement Zones ---
        # Each zone: { 'rect': (x1, y1, x2, y2), 'factor': float, 'buffer_mult': float }
        self.refinement_zones = []
        self._drawing_refinement = False  # True while user is dragging a new rect
        self._refine_start = None         # world-coord corner of current drag
        self._refine_current = None       # world-coord opposite corner
        self._refine_factor = 2.0         # default refinement factor for new zones
        self._refine_buffer_mult = 5.0    # default buffer multiplier for new zones

        # --- Boundary Conditions (SI) ---
        self.inlet_velocity = 1.0
        self.outlet_pressure = 0.0

        # --- Line Selection State ---
        self.selected_line = None
        self.boundary_types = ["Wall", "Symmetry", "Velocity Inlet", "Pressure Outlet"]
        self.current_line_idx = 0
        
        # --- Re calculator part ---
        self.char_length = 1.0 # Default to 1m, 1mm, etc.

        # --- Solver Settings (passed to Solver.__init__ and SolverPanel) ---
        self.alpha_u        = 0.7      # velocity under-relaxation (SIMPLE standard)
        self.alpha_p        = 0.3      # pressure under-relaxation (SIMPLE standard)
        self.max_iterations = 1600
        # Continuity convergence criterion, RELATIVE to inlet mass flux since
        # the 2026-07 solver fixes. 1e-6 was shown to fire while the velocity
        # profile is still developing (Poiseuille 36k validation); 1e-8 tracks
        # true convergence.
        self.tolerance      = 1e-8
        self.viz_interval   = 10       # live field snapshot every N iterations

    # ------------------------------------------------------------------
    def _line_length(self, line):
        """Returns length of a line in world units."""
        dx = line.b.x - line.a.x
        dy = line.b.y - line.a.y
        return math.hypot(dx, dy)

    # ------------------------------------------------------------------
    def _get_refinement_polygons(self):
        """Return a list of (shapely_polygon, factor, buffer_mult) for all refinement zones."""
        result = []
        for zone in self.refinement_zones:
            x1, y1, x2, y2 = zone['rect']
            # Normalise so x1 <= x2, y1 <= y2
            rx1, rx2 = min(x1, x2), max(x1, x2)
            ry1, ry2 = min(y1, y2), max(y1, y2)
            poly = ShapelyPoly([(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)])
            result.append((poly, zone['factor'], zone.get('buffer_mult', 5.0)))
        return result

    # ------------------------------------------------------------------
    def draw(self, gfx, vbos=None):
        camera = gfx.camera

        # --- Mesh overlay (drawn before imgui so UI sits on top) ---
        if vbos:
            gfx.draw_vbo(vbos.get('triangles'), color=(0, 100, 255))
            gfx.draw_vbo(vbos.get('quads'), color=(0, 255, 100))
            gfx.draw_vbo(vbos.get('walls'), color=(255, 255, 255))

        # --- Draw refinement zone overlays ---
        self._draw_refinement_zones(gfx)

        u = self._unit_names[self._unit_idx]

        # --- Hover Detection ---
        # Read mouse from imgui IO so we do not need an extra pygame call
        mx, my = imgui.get_io().mouse_pos
        mouse_world = camera.screen_to_world((mx, my))
        # Only hover-detect when mouse is NOT over an imgui window
        want_mouse = imgui.get_io().want_capture_mouse

        hovered_line = None
        if not want_mouse:
            for line in self.lines:
                if line.is_mouse_over(mouse_world,camera):
                    hovered_line = line
                    break

        # --- Draw Lines with Colour Coding ---
        for line in self.lines:
            if line is self.selected_line:
                color = (255, 200, 0)    # gold  = selected
            elif line is hovered_line:
                color = (100, 220, 255)  # cyan  = hovered
            else:
                color = (255, 255, 255)  # white = default
            line.draw(gfx, color=color, width=2)

        # --- Canvas Hover Tooltip ---
        if hovered_line and not want_mouse:
            length = self._line_length(hovered_line)
            idx = self.lines.index(hovered_line) + 1
            imgui.begin_tooltip()
            imgui.text(f"Line {idx}")
            imgui.text(f"Length: {length:.3f} {u}")
            imgui.text(f"Type: {hovered_line.boundary_type}")
            imgui.end_tooltip()

        # ---- Main Settings Window ----
        # Auto-fit to content, but cap height so it can't grow past the
        # screen on tall control lists (refinement zones, solver settings).
        max_h = imgui.get_io().display_size[1] * 0.9
        imgui.set_next_window_size_constraints((0, 0), (480, max_h))
        imgui.begin("Mesher Settings", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        changed_u, self._unit_idx = imgui.combo("World units", self._unit_idx, self._unit_names)
        if changed_u:
            self.unit_to_meters = _UNIT_FACTORS[self._unit_names[self._unit_idx]]

        imgui.separator()

        imgui.text("Fluid Properties")
        _, self.density = imgui.input_float("Density [kg/m3]", self.density, step=0.1, format="%.3f")
        _, self.viscosity = imgui.input_float("Dynamic Viscosity [Pa*s]", self.viscosity, format="%.3e")

        # --- New Validation Section ---
        imgui.separator()
        imgui.text("Validation Checks")

        # Let the user define the characteristic length (L)
        _, self.char_length = imgui.input_float(f"Char. Length [{u}]", self.char_length, step=0.1)

        # Calculate Reynolds: Re = (rho * V * L) / mu
        # Note: we multiply char_length by unit_to_meters to keep the math in SI
        world_L = self.char_length * self.unit_to_meters
        reynolds = (self.density * self.inlet_velocity * world_L) / max(self.viscosity, 1e-12)

        imgui.text(f"Reynolds Number: {reynolds:.2e}")

        # DNS Estimator 
        if reynolds > 0:
            # 2D DNS grid scaling is linear with Re due to the enstrophy dissipation scale
            dns_cells = reynolds 
            
            if dns_cells > 5e6: # 5 Million cells is getting heavy for an interactive 2D solver
                imgui.text_colored(f"Est. 2D DNS Cells: {dns_cells:.2e} (High for real-time)", 1.0, 0.4, 0.4)
            else:
                imgui.text(f"Est. 2D DNS Cells: {dns_cells:.2e}")
        imgui.separator()

        opened, _ = imgui.collapsing_header("Boundary layer settings")
        if opened:
            _, self.n_layers         = imgui.input_int(  "N. Boundary layers",           self.n_layers,         step=1,   step_fast=1)
            _, self.growth_factor    = imgui.input_float("Growth factor",                 self.growth_factor,    step=0.05, step_fast=1.0)
            _, self.thickness        = imgui.input_float(f"First layer thickness [{u}]",  self.thickness,        step=0.25, step_fast=5.0)

            # --- Per-BC boundary cell spacing ---
            imgui.separator()
            _, self._spacing_linked = imgui.checkbox("Linked spacing (all BCs)", self._spacing_linked)
            if self._spacing_linked:
                changed, self.boundary_spacing = imgui.input_float(
                    f"Boundary cell spacing [{u}]", self.boundary_spacing,
                    step=0.5, step_fast=10.0)
                if changed:
                    # Propagate global value to all BC types
                    for k in self._bc_spacing:
                        self._bc_spacing[k] = self.boundary_spacing
            else:
                # Show individual per-BC spacing fields
                for bc_type in sorted(self._bc_spacing.keys()):
                    label = f"{bc_type} spacing [{u}]"
                    changed, val = imgui.input_float(label, self._bc_spacing[bc_type],
                                                     step=0.5, step_fast=10.0)
                    if changed:
                        self._bc_spacing[bc_type] = max(0.1, val)
                if imgui.button("Reset all to global"):
                    for k in self._bc_spacing:
                        self._bc_spacing[k] = self.boundary_spacing

        opened2, _ = imgui.collapsing_header("Mesher settings")
        if opened2:
            _, self.r = imgui.input_float(f"Mesh size (min sep.) [{u}]", self.r, step=1.0, step_fast=10.0)

        opened_sm, _ = imgui.collapsing_header("Smoothing settings")
        if opened_sm:
            imgui.push_item_width(160)
            _, self.smooth_passes = imgui.input_int("Passes", self.smooth_passes, step=1)
            self.smooth_passes = max(1, min(self.smooth_passes, 10))
            _, self.smooth_relaxation = imgui.slider_float(
                "Relaxation", self.smooth_relaxation, 0.05, 1.0)
            imgui.pop_item_width()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Relax interior points toward their Delaunay\n"
                                  "neighbour centroid to even out the seam between\n"
                                  "the boundary layer and the interior triangles.\n"
                                  "Used by the Smooth Mesh button below.")

        # --- Refinement Zones Section ---
        imgui.separator()
        opened3, _ = imgui.collapsing_header("Refinement Zones", flags=imgui.TREE_NODE_DEFAULT_OPEN)
        if opened3:
            # Draw the list of existing zones
            to_remove = None
            for i, zone in enumerate(self.refinement_zones):
                x1, y1, x2, y2 = zone['rect']
                rx1, rx2 = min(x1, x2), max(x1, x2)
                ry1, ry2 = min(y1, y2), max(y1, y2)
                w = rx2 - rx1
                h = ry2 - ry1
                # Delete button on the left, always visible
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.8, 0.2, 0.2, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 1.0, 0.3, 0.3, 1.0)
                if imgui.button(f" X ##del_{i}"):
                    to_remove = i
                imgui.pop_style_color(2)
                imgui.same_line()
                imgui.text(f"Zone {i+1}:")
                
                imgui.same_line()
                imgui.push_item_width(140)
                _, zone['factor'] = imgui.input_float(f"##factor_{i}", zone['factor'], step=0.5, step_fast=1.0)
                imgui.pop_item_width()
                zone['factor'] = max(1.1, zone['factor'])
                
                imgui.same_line()
                imgui.text_colored(f"({self.r / zone['factor']:.1f}{u})", 0.6, 1.0, 0.6, 1.0)
                
                # Buffer multiplier per zone
                bm = zone.get('buffer_mult', 5.0)
                imgui.same_line()
                imgui.push_item_width(140)
                _, zone['buffer_mult'] = imgui.input_float(f"##buf_{i}", bm, step=0.5, step_fast=1.0)
                imgui.pop_item_width()
                zone['buffer_mult'] = max(1.0, zone['buffer_mult'])
                
                imgui.same_line()
                local_r = self.r / zone['factor']
                imgui.text_colored(f"buf:{zone['buffer_mult'] * local_r:.1f}{u}", 0.6, 1.0, 0.6, 1.0)

                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Rect: ({rx1:.1f}, {ry1:.1f}) to ({rx2:.1f}, {ry2:.1f}) [{w:.1f}x{h:.1f} {u}]")
            if to_remove is not None:
                self.refinement_zones.pop(to_remove)

            # "Add Refinement Zone" button / drawing mode toggle
            if not self._drawing_refinement:
                if imgui.button("Add Refinement Zone"):
                    self._drawing_refinement = True
                    self._refine_start = None
                    self._refine_current = None
            else:
                imgui.text_colored("Click & drag on canvas to draw a refinement rectangle", 0.2, 1.0, 0.2, 1.0)
                if imgui.button("Cancel"):
                    self._drawing_refinement = False
                    self._refine_start = None
                    self._refine_current = None

            _, self._refine_factor = imgui.input_float("Refinement factor", self._refine_factor, step=0.5, step_fast=1.0)
            self._refine_factor = max(1.1, self._refine_factor)
            _, self._refine_buffer_mult = imgui.input_float("Buffer multiplier", self._refine_buffer_mult, step=0.5, step_fast=1.0)
            self._refine_buffer_mult = max(1.0, self._refine_buffer_mult)

        imgui.separator()

        opened_s, _ = imgui.collapsing_header("Solver Settings")
        if opened_s:
            imgui.push_item_width(160)

            _, self.alpha_u = imgui.slider_float(
                "alpha_u  (velocity relax.)", self.alpha_u, 0.01, 0.99, format="%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Under-relaxation for U/V momentum equations.\n"
                                  "Lower = more stable but slower convergence.\n"
                                  "High-Re or separated flows may need 0.1–0.2.")

            _, self.alpha_p = imgui.slider_float(
                "alpha_p  (pressure relax.)", self.alpha_p, 0.01, 0.99, format="%.2f")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Under-relaxation for pressure correction.\n"
                                  "Typically half of alpha_u. Reduce for stability.")

            _, self.max_iterations = imgui.input_int(
                "Max iterations", self.max_iterations, step=100, step_fast=500)
            self.max_iterations = max(1, self.max_iterations)

            # Tolerance as a log10 slider: "1e-N" where N = 3..10
            tol_exp = int(round(-np.log10(max(self.tolerance, 1e-15))))
            tol_exp = max(3, min(tol_exp, 10))
            changed_tol, tol_exp = imgui.slider_int(
                "Tolerance  (1e-N)", tol_exp, 3, 10)
            if changed_tol:
                self.tolerance = 10.0 ** (-tol_exp)
            imgui.same_line()
            imgui.text_colored(f"= {self.tolerance:.0e}", 0.6, 1.0, 0.6, 1.0)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Continuity RMS residual target.\n"
                                  "1e-4 is typical; 1e-6 for production results.")

            _, self.viz_interval = imgui.input_int(
                "Live viz every N iters", self.viz_interval, step=5)
            self.viz_interval = max(1, self.viz_interval)
            if imgui.is_item_hovered():
                imgui.set_tooltip("How often the background field is updated\n"
                                  "in the Solver Monitor during a solve.\n"
                                  "Higher = less GPU overhead during solving.")

            imgui.pop_item_width()

        imgui.separator()
        mesh_label = "Remesh" if self.has_mesh else "Mesh"
        if imgui.button(mesh_label):
            self.pending_action = PhysicsAction.MESH
        if self.has_mesh:
            imgui.same_line()
            if imgui.button("Solve"):
                self.pending_action = PhysicsAction.SOLVE
        if self.has_mesh and self.mesher is not None:
            imgui.same_line()
            if imgui.button("Smooth Mesh"):
                self.pending_action = PhysicsAction.SMOOTH_MESH
            if imgui.is_item_hovered():
                imgui.set_tooltip("Relax interior points toward their Delaunay\n"
                                  "neighbour centroid to even out the seam between\n"
                                  "the boundary layer and the interior triangles.\n"
                                  "Opt-in and reversible only by re-meshing.\n"
                                  "Passes/relaxation set under Smoothing settings above.")
        imgui.same_line()
        if imgui.button("Save Mesh"):
            self.open_save_dialog()
        imgui.same_line()
        if imgui.button("Load Mesh"):
            self.open_load_dialog()
        imgui.same_line()
        if imgui.button("Load Visualization"):
            self.open_load_visualization_dialog()
        imgui.end()

        # ---- Per-Line Selection Popup ----
        if self.selected_line:
            idx = self.lines.index(self.selected_line) + 1
            length = self._line_length(self.selected_line)
            title = f"Line {idx} settings"

            # Place first-open position in top-right, away from the main menu
            imgui.set_next_window_position(820, 50, imgui.FIRST_USE_EVER)

            imgui.begin(title, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

            imgui.text(f"Length: {length:.3f} {u}")
            imgui.separator()

            self.current_line_idx = self.boundary_types.index(self.selected_line.boundary_type)
            changed, self.current_line_idx = imgui.combo(
                "Condition", self.current_line_idx, self.boundary_types)
            if changed:
                self.selected_line.boundary_type = self.boundary_types[self.current_line_idx]
                print(f"Line {idx} boundary type -> {self.selected_line.boundary_type}")

            if self.selected_line.boundary_type == "Velocity Inlet":
                _, self.inlet_velocity = imgui.input_float(
                    "Inlet velocity [m/s]", self.inlet_velocity, step=0.1, step_fast=1.0)

            if self.selected_line.boundary_type == "Pressure Outlet":
                _, self.outlet_pressure = imgui.input_float(
                    "Outlet pressure [Pa]", self.outlet_pressure, step=0.1, step_fast=1.0)

            imgui.end()

    # ------------------------------------------------------------------
    def _draw_refinement_zones(self, gfx):
        """Draw all refinement zone rectangles as semi-transparent overlays."""
        camera = gfx.camera

        # Existing zones: light blue
        for zone in self.refinement_zones:
            x1, y1, x2, y2 = zone['rect']
            rx1, rx2 = min(x1, x2), max(x1, x2)
            ry1, ry2 = min(y1, y2), max(y1, y2)
            gfx.draw_rect(camera.to_screen((rx1, ry1)),
                          camera.to_screen((rx2, ry2)),
                          fill_rgba=(0.2, 0.6, 1.0, 0.15),
                          outline_rgba=(0.2, 0.6, 1.0, 0.6))

        # The in-progress rectangle while dragging: green
        if self._drawing_refinement and self._refine_start is not None and self._refine_current is not None:
            x1, y1 = self._refine_start.x, self._refine_start.y
            x2, y2 = self._refine_current.x, self._refine_current.y
            rx1, rx2 = min(x1, x2), max(x1, x2)
            ry1, ry2 = min(y1, y2), max(y1, y2)
            gfx.draw_rect(camera.to_screen((rx1, ry1)),
                          camera.to_screen((rx2, ry2)),
                          fill_rgba=(0.2, 1.0, 0.2, 0.12),
                          outline_rgba=(0.2, 1.0, 0.2, 0.8))

    # ------------------------------------------------------------------
    def handle_event(self, event, camera):
        """Handle events for the physics editor, including refinement zone drawing."""
        if self._drawing_refinement:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Start dragging a new refinement rectangle
                world_pos = camera.screen_to_world(event.pos)
                self._refine_start = world_pos
                self._refine_current = world_pos
            elif event.type == pygame.MOUSEMOTION and self._refine_start is not None:
                world_pos = camera.screen_to_world(event.pos)
                self._refine_current = world_pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self._refine_start is not None:
                # Finish the rectangle
                world_pos = camera.screen_to_world(event.pos)
                x1, y1 = self._refine_start.x, self._refine_start.y
                x2, y2 = world_pos.x, world_pos.y
                # Only add if the rectangle has non-zero area
                if abs(x2 - x1) > 0.01 and abs(y2 - y1) > 0.01:
                    self.refinement_zones.append({
                        'rect': (x1, y1, x2, y2),
                        'factor': self._refine_factor,
                        'buffer_mult': self._refine_buffer_mult,
                    })
                    print(f"[Refinement] Added zone: ({x1:.2f},{y1:.2f})→({x2:.2f},{y2:.2f}) ×{self._refine_factor:.1f} buf×{self._refine_buffer_mult:.1f}")
                self._drawing_refinement = False
                self._refine_start = None
                self._refine_current = None
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # Cancel refinement drawing
                self._drawing_refinement = False
                self._refine_start = None
                self._refine_current = None
        else:
            # Normal line selection handling
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not imgui.get_io().want_capture_mouse:
                    self.handle_selection(camera.screen_to_world(event.pos), camera)

    # ------------------------------------------------------------------
    def handle_selection(self, pos, camera):
        """Called on left-click (world coords). Clicking same line deselects it."""
        for line in self.lines:
            if line.is_mouse_over(pos, camera):
                if line is self.selected_line:
                    self.selected_line = None  # toggle off
                else:
                    self.selected_line = line
                return
        # Click on empty space -> deselect
        self.selected_line = None

    # ------------------------------------------------------------------
    def open_save_dialog(self):
        """Opens a native OS file dialog to choose where to save the mesh."""
        import tkinter as tk
        from tkinter import filedialog
        from . import meshIO

        # 1. Initialize a hidden Tkinter root window
        # (Otherwise, a blank, ugly white box pops up alongside the file explorer)
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # Force the file explorer to open on top of Pygame

        # 2. Open the native "Save As" window
        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy Compressed Archive", "*.npz"), ("All Files", "*.*")],
            title="Export Solver Mesh"
        )

        # 3. Destroy the hidden root immediately so it cleans up resources
        root.destroy()

        # 4. If the user didn't click 'Cancel', extract the data and save it!
        if filepath:
            print(f"[UI] User selected save path: {filepath}")
            if self.mesher is not None:
                mesh_data = self.mesher.solver_data_pipeline()
            elif self.loaded_mesh is not None:
                mesh_data = self.loaded_mesh
            else:
                print("[UI] No mesh available to save yet — mesh first.")
                return
            meshIO.save_mesh_for_solver(mesh_data, filepath)

    def open_load_dialog(self):
        """Opens a native OS file dialog to pick a saved mesh (.npz) to load."""
        import tkinter as tk
        from tkinter import filedialog
        from . import meshIO

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        filepath = filedialog.askopenfilename(
            defaultextension=".npz",
            filetypes=[("NumPy Compressed Archive", "*.npz"), ("All Files", "*.*")],
            title="Load Solver Mesh"
        )

        root.destroy()

        if filepath:
            print(f"[UI] User selected load path: {filepath}")
            self.loaded_mesh = meshIO.load_mesh_for_solver(filepath)
            self.pending_action = PhysicsAction.LOAD_MESH
            self.selected_line = None  # Reset selection to avoid crash on stale line object

    def open_load_visualization_dialog(self):
        """Opens a native OS file dialog to pick a saved visualization (.npz
        with solved fields) to load — jumps straight to VISUALIZER, skipping
        the meshing/solving steps entirely."""
        import tkinter as tk
        from tkinter import filedialog
        from . import meshIO

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        filepath = filedialog.askopenfilename(
            defaultextension=".npz",
            filetypes=[("NumPy Compressed Archive", "*.npz"), ("All Files", "*.*")],
            title="Load Visualization"
        )

        root.destroy()

        if filepath:
            print(f"[UI] User selected visualization load path: {filepath}")
            data = meshIO.load_mesh_for_solver(filepath)
            if 'P' not in data or 'U' not in data:
                print("[UI] File has no solved fields — use Load Mesh instead.")
                return
            self.loaded_visualization = data
            self.pending_action = PhysicsAction.LOAD_VISUALIZATION
            self.selected_line = None  # Reset selection to avoid crash on stale line object
