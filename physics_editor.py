import imgui
from imgui.integrations.pygame import PygameRenderer
from line import Line
import math

# Conversion factors to SI (metres)
_UNIT_FACTORS = {"mm": 0.001, "cm": 0.01, "m": 1.0}


class PhysicsEditor:

    def __init__(self, screen, lines, renderer, initial_unit_idx=2):
        self.lines = lines
        self.renderer = renderer
        self.finished = False

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

        # --- Mesh Generation (in world units) ---
        self.r = 4.0

        # --- Boundary Conditions (SI) ---
        self.inlet_velocity = 1.0
        self.outlet_pressure = 0.0

        # --- Line Selection State ---
        self.selected_line = None
        self.boundary_types = ["Wall", "Velocity Inlet", "Pressure Outlet"]
        self.current_line_idx = 0
        
        # --- Re calculator part ---
        self.char_length = 1.0 # Default to 1m, 1mm, etc.

    # ------------------------------------------------------------------
    def _line_length(self, line):
        """Returns length of a line in world units."""
        dx = line.b.x - line.a.x
        dy = line.b.y - line.a.y
        return math.hypot(dx, dy)

    # ------------------------------------------------------------------
    def draw(self, screen, camera):
        imgui.new_frame()

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
            line.draw(screen, camera, color=color, width=2)

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
        imgui.begin("Mesher Settings")

        changed_u, self._unit_idx = imgui.combo("World units", self._unit_idx, self._unit_names)
        if changed_u:
            self.unit_to_meters = _UNIT_FACTORS[self._unit_names[self._unit_idx]]

        imgui.separator()

        imgui.text("Fluid Properties")
        _, self.density = imgui.input_float("Density [kg/m3]", self.density, step=0.1, format="%.3f")
        _, self.viscosity = imgui.input_float("Viscosity [Pa*s]", self.viscosity, format="%.3e")

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

        # DNS Estimator (3D Scaling: N ~ Re^2.25)
        if reynolds > 0:
            dns_cells = math.pow(reynolds, 2.25)
            
            # Visual warning if the number is getting absurd
            if dns_cells > 1e8:
                imgui.text_colored("Est. DNS Cells: >100M (Heavy!)", 1.0, 0.4, 0.4)
            else:
                imgui.text(f"Est. DNS Cells: {dns_cells:.2e}")

            # Tooltip explaining the scaling
            if imgui.is_item_hovered():
                imgui.set_tooltip("Estimated number of cells required for DNS based on \n"
                                "the Kolmogorov scale resolution: N ≈ Re^(9/4)")

        imgui.separator()

        opened, _ = imgui.collapsing_header("Boundary layer settings")
        if opened:
            _, self.n_layers         = imgui.input_int(  "N. Boundary layers",           self.n_layers,         step=1,   step_fast=1)
            _, self.growth_factor    = imgui.input_float("Growth factor",                 self.growth_factor,    step=0.05, step_fast=1.0)
            _, self.thickness        = imgui.input_float(f"First layer thickness [{u}]",  self.thickness,        step=0.25, step_fast=5.0)
            _, self.boundary_spacing = imgui.input_float(f"Boundary cell spacing [{u}]",  self.boundary_spacing, step=0.5, step_fast=10.0)

        opened2, _ = imgui.collapsing_header("Mesher settings")
        if opened2:
            _, self.r = imgui.input_float(f"Mesh size (min sep.) [{u}]", self.r, step=1.0, step_fast=10.0)

        imgui.separator()
        if imgui.button("Proceed to Meshing"):
            self.finish()
        imgui.end()

        # ---- Per-Line Selection Popup ----
        # BUG FIX: Previously set_next_window_position and set_next_window_size were called
        # every frame with no condition, which (a) pinned the window so it could not be
        # moved, and (b) caused it to overlap and hide behind the main settings window.
        # Fix: FIRST_USE_EVER means imgui only applies the hint once (on first open).
        # WINDOW_ALWAYS_AUTO_RESIZE replaces the fixed height so content is never clipped.
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

        imgui.render()
        self.renderer.render(imgui.get_draw_data())

    # ------------------------------------------------------------------
    def handle_selection(self, pos,camera):
        """Called on left-click (world coords). Clicking same line deselects it."""
        for line in self.lines:
            if line.is_mouse_over(pos,camera):
                if line is self.selected_line:
                    self.selected_line = None  # toggle off
                else:
                    self.selected_line = line
                return
        # Click on empty space -> deselect
        self.selected_line = None

    def finish(self):
        self.finished = True