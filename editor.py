import pygame
import math
from line import Line
from snapengine import SnapEngine
from pygame_widgets.button import Button
import pygame_widgets
from camera import Camera
import imgui


class Editor:
    def __init__(self, screen, renderer):
        self.lines = []
        self.snap_engine = SnapEngine(pixel_radius=10)
        self.is_drawing = False
        self.start_pos = None  
        self.current_mouse_pos = (0, 0)
        self.screen = screen
        self.renderer = renderer
        self.finished = False
        
        # --- NEW CAD STATE ---
        self.target_length = 0.0
        self.ortho_mode = False
        
        # --- NEW UNIT STATE (Solves Problem 3) ---
        self.unit_names = ["mm", "cm", "m"]
        self.unit_idx = 2 # Default to meters so 10 = 10m
        
        self.snap_step = 1.0 # Set to 0.0 to disable, 1.0 for whole numbers
        self.show_tracking_lines = True
        
    def _apply_constraints(self, start, target):
        """Overrides the target position based on CAD UI inputs."""
        from point import Point
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

    # Update click logic to convert FIRST
    def _handle_click(self, screen_mouse_pos, camera):
        world_mouse_pos = camera.screen_to_world(screen_mouse_pos)
        snapped_pos = self.snap_engine.get_snapped_pos(
            world_mouse_pos, self.lines, camera.scale, self.start_pos
        )

        if not self.is_drawing:
            self.start_pos = snapped_pos
            self.is_drawing = True
        else:
            # Apply our exact measurements before creating the line
            final_pos = self._apply_constraints(self.start_pos, snapped_pos)
            new_line = Line(self.start_pos, final_pos)
            self.lines.append(new_line)
            self.start_pos = final_pos # Chain to next line

    def _cancel_or_undo(self):
        if self.is_drawing:
            self.is_drawing = False
            self.start_pos = None
        elif self.lines:
            self.lines.pop()
        
        
    def _apply_constraints(self, start, snapped_target):
        """Finalizes the point by rounding the length to the nearest step."""
        from point import Point
        import math
        
        dx = snapped_target.x - start.x
        dy = snapped_target.y - start.y
        length = math.hypot(dx, dy)

        if self.snap_step > 0 and length > 0.001:
            # Round the length to the nearest integer/step
            snapped_length = round(length / self.snap_step) * self.snap_step
            # Scale the vector to the new snapped length
            scale = snapped_length / length
            return Point(start.x + dx * scale, start.y + dy * scale)
        
        return snapped_target

    def draw(self, screen, camera):
        imgui.new_frame()
        
        imgui.set_next_window_position(50, 50, imgui.ALWAYS)
        imgui.begin("Controls", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        
        # --- NEW CAD CONTROLS ---
        changed_u, self.unit_idx = imgui.combo("Units", self.unit_idx, self.unit_names)
        _, self.ortho_mode = imgui.checkbox("Ortho (90° Snap)", self.ortho_mode)
        _, self.target_length = imgui.input_float("Exact Length (0 = Free)", self.target_length, step=0.1)
        imgui.separator()
        
        if imgui.button("Finish CAD"):
            self.finish()
        imgui.end()

        # --- NEW: Draw World Origin (0,0) ---
        origin_screen = camera.to_screen((0, 0))
        # Draw a simple Red/Green crosshair for X/Y axes
        pygame.draw.line(screen, (255, 50, 50), (origin_screen[0] - 15, origin_screen[1]), (origin_screen[0] + 15, origin_screen[1]), 2)
        pygame.draw.line(screen, (50, 255, 50), (origin_screen[0], origin_screen[1] - 15), (origin_screen[0], origin_screen[1] + 15), 2)

        # 3. Draw your CAD lines (World Space)
        for line in self.lines:
            line.draw(screen, camera)

        if self.is_drawing and self.start_pos:
            world_mouse = camera.screen_to_world(self.current_mouse_pos)
            snapped_pos = self.snap_engine.get_snapped_pos(
                world_mouse, self.lines, camera.scale, self.start_pos
            )
            
            # Apply constraints to the preview line
            target_world_pos = self._apply_constraints(self.start_pos, snapped_pos)
            
            p1_screen = camera.to_screen(self.start_pos)
            p2_screen = camera.to_screen(target_world_pos)
            
            camera.draw_screen_line(screen, p1_screen, p2_screen, (150, 150, 150), 1)
            camera.draw_circle(screen, (0, 255, 0), p2_screen, 3, 1)

            # --- FIXED SECTION: Use .x and .y instead of [0] and [1] ---
            dx = target_world_pos.x - self.start_pos.x
            dy = target_world_pos.y - self.start_pos.y
            
            # Use your Point class's built-in distance method
            length = self.start_pos.distance_to(target_world_pos)

            # --- Floating ImGui Tooltip ---
            tooltip_x = self.current_mouse_pos[0] + 15
            tooltip_y = self.current_mouse_pos[1] + 15
            
            imgui.set_next_window_position(tooltip_x, tooltip_y, imgui.ALWAYS)
            imgui.begin("CursorInfo", flags=imgui.WINDOW_NO_TITLE_BAR | 
                                            imgui.WINDOW_ALWAYS_AUTO_RESIZE | 
                                            imgui.WINDOW_NO_MOVE | 
                                            imgui.WINDOW_NO_INPUTS)
            
            imgui.text(f"Length: {length:.4f} m")
            imgui.text(f"dx: {dx:.4f} | dy: {dy:.4f}")
            imgui.text(f"Pos: ({target_world_pos.x:.2f}, {target_world_pos.y:.2f})")
            imgui.end()

        # 5. Render ImGui on top of everything
        imgui.render()
        self.renderer.render(imgui.get_draw_data())
            
    def finish(self):
        self.finished = True
        pass