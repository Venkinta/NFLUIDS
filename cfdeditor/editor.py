import pygame
import math
from .line import Line
from .snapengine import SnapEngine
from .point import Point
import imgui


class Editor:
    def __init__(self):
        self.lines = []
        self.snap_engine = SnapEngine(pixel_radius=10)
        self.is_drawing = False
        self.start_pos = None
        self.current_mouse_pos = (0, 0)
        self.finished = False
        
        # --- NEW CAD STATE ---
        self.target_length = 0.0
        self.ortho_mode = False
        
        # --- NEW UNIT STATE (Solves Problem 3) ---
        self.unit_names = ["mm", "cm", "m"]
        self.unit_idx = 0 # Default to mm (unit_idx=0); camera scale=2.0 -> 1mm = 2px
        
        self.snap_step = 1.0 # Set to 0.0 to disable, 1.0 for whole numbers
        self.show_tracking_lines = True
        
    def _apply_constraints(self, start, target):
        """
        Applies CAD constraints to the target position in order:
        1. Ortho lock (snap to nearest 90° axis)
        2. Fixed length (exact line length)
        3. Snap step (round length to nearest step value)
        """
        dx = target.x - start.x
        dy = target.y - start.y

        # 1. Apply Ortho Lock (Snap to nearest 90 deg)
        if self.ortho_mode:
            if abs(dx) > abs(dy):
                dy = 0
            else:
                dx = 0

        # 2. Apply Fixed Length
        if self.target_length > 0:
            length = math.hypot(dx, dy)
            if length > 0.0001:
                dx = (dx / length) * self.target_length
                dy = (dy / length) * self.target_length

        # 3. Apply Snap Step (round length to nearest step)
        length = math.hypot(dx, dy)
        if self.snap_step > 0 and length > 0.001:
            snapped_length = round(length / self.snap_step) * self.snap_step
            scale = snapped_length / length
            dx *= scale
            dy *= scale

        return Point(start.x + dx, start.y + dy)

    # Update signature to accept camera
    def handle_event(self, event, camera):
        if event.type == pygame.MOUSEMOTION:
            self.current_mouse_pos = event.pos

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(event.pos, camera) 

        # --- NEW: Escape Key Logic ---
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._handle_escape()

    def _handle_escape(self):
        if self.is_drawing:
            # 1. If mid-drawing, just stop drawing and "throw away" the start point
            self.is_drawing = False
            self.start_pos = None
            print("Drawing canceled.")
        elif self.lines:
            # 2. If not drawing, remove the last line (Undo)
            removed_line = self.lines.pop()
            print("Last line removed.")

    def check_loops(self):
        """Analyze current lines and return a list of loop statuses.
        Returns: (all_closed: bool, statuses: list of strings)
        """
        remaining = self.lines.copy()
        statuses = []
        all_closed = True
        
        loop_idx = 1
        while remaining:
            first = remaining.pop(0)
            ordered = [first]
            pivot = first.b
            closed = False
            while True:
                if pivot == ordered[0].a:
                    closed = True
                    break
                found = False
                for i, line in enumerate(remaining):
                    if pivot == line.a:
                        pivot = line.b
                        ordered.append(line)
                        remaining.pop(i)
                        found = True
                        break
                    elif pivot == line.b:
                        pivot = line.a
                        ordered.append(line)
                        remaining.pop(i)
                        found = True
                        break
                if not found:
                    break
            
            if closed:
                statuses.append(f"Loop {loop_idx}: OK (Closed)")
            else:
                statuses.append(f"Loop {loop_idx}: OPEN!")
                all_closed = False
            loop_idx += 1
            
        if not self.lines:
            return True, ["No geometry drawn yet."]
        return all_closed, statuses

    def new_loop(self):
        """Finalize the current chain (if any) and start a fresh, disconnected
        loop.  The mesher groups all drawn lines into separate closed loops by
        connectivity, so this just breaks the chain so the next click begins a
        new loop (e.g. a hole inside the domain)."""
        self.is_drawing = False
        self.start_pos = None
        print("New loop started.")

    # Update click logic to convert FIRST
    def _handle_click(self, screen_mouse_pos, camera):
        world_mouse_pos = camera.screen_to_world(screen_mouse_pos)
        snapped_pos, is_vertex_snap = self.snap_engine.get_snapped_pos(
            world_mouse_pos, self.lines, camera.scale, self.start_pos
        )

        if not self.is_drawing:
            self.start_pos = snapped_pos
            self.is_drawing = True
        else:
            # If vertex snapped, use it exactly to guarantee watertight closure
            if is_vertex_snap:
                final_pos = snapped_pos
            else:
                final_pos = self._apply_constraints(self.start_pos, snapped_pos)
            new_line = Line(self.start_pos, final_pos)
            self.lines.append(new_line)
            self.start_pos = final_pos # Chain to next line

    def draw(self, gfx):
        camera = gfx.camera
        imgui.set_next_window_position(50, 50, imgui.ALWAYS)
        imgui.begin("Controls", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        
        # --- NEW CAD CONTROLS ---
        changed_u, self.unit_idx = imgui.combo("Units", self.unit_idx, self.unit_names)
        _, self.ortho_mode = imgui.checkbox("Ortho (90° Snap)", self.ortho_mode)
        _, self.target_length = imgui.input_float("Exact Length (0 = Free)", self.target_length, step=0.1)
        imgui.separator()

        all_closed, loop_statuses = self.check_loops()
        for status in loop_statuses:
            color = (0.2, 1.0, 0.2, 1.0) if "OK" in status else (1.0, 0.8, 0.2, 1.0)
            imgui.text_colored(status, *color)
            
        if not all_closed:
            imgui.text_colored("⚠️ Warning: Open loops detected!", 1.0, 0.4, 0.4, 1.0)
        
        if imgui.button("Finish CAD"):
            self.finish()
        imgui.same_line()
        if imgui.button("New Loop"):
            self.new_loop()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Break the current chain to start a new disconnected shape (e.g. a hole).")
        imgui.end()

        # 3. Draw your CAD lines (World Space)
        for line in self.lines:
            line.draw(gfx)

        if self.is_drawing and self.start_pos:
            world_mouse = camera.screen_to_world(self.current_mouse_pos)
            snapped_pos, is_vertex_snap = self.snap_engine.get_snapped_pos(
                world_mouse, self.lines, camera.scale, self.start_pos
            )
            
            # If vertex snapped, use it exactly (otherwise apply constraints)
            if is_vertex_snap:
                target_world_pos = snapped_pos
            else:
                target_world_pos = self._apply_constraints(self.start_pos, snapped_pos)
            
            p1_screen = camera.to_screen(self.start_pos)
            p2_screen = camera.to_screen(target_world_pos)
            
            gfx.draw_screen_line(p1_screen, p2_screen, (150, 150, 150), 1)
            gfx.draw_circle(p2_screen, 3, (0, 255, 0), 1)

            # --- FIXED SECTION: Use .x and .y instead of [0] and [1] ---
            dx = target_world_pos.x - self.start_pos.x
            dy = target_world_pos.y - self.start_pos.y
            
            # Use your Point class's built-in distance method
            length = self.start_pos.distance_to(target_world_pos)

            # --- Floating ImGui Tooltip ---
            tooltip_x = self.current_mouse_pos[0] + 15
            tooltip_y = self.current_mouse_pos[1] + 15
            
            current_unit = self.unit_names[self.unit_idx] # Grab "mm", "cm", or "m"
            
            imgui.set_next_window_position(tooltip_x, tooltip_y, imgui.ALWAYS)
            imgui.begin("CursorInfo", flags=imgui.WINDOW_NO_TITLE_BAR | 
                                            imgui.WINDOW_ALWAYS_AUTO_RESIZE | 
                                            imgui.WINDOW_NO_MOVE | 
                                            imgui.WINDOW_NO_INPUTS)
            
            imgui.text(f"Length: {length:.4f} {current_unit}")
            imgui.text(f"dx: {dx:.4f} | dy: {dy:.4f}")
            imgui.text(f"Pos: ({target_world_pos.x:.2f}, {target_world_pos.y:.2f})")
            imgui.end()

    def finish(self):
        self.finished = True
