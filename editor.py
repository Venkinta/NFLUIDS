import pygame
import math
from line import Line
from snapengine import SnapEngine
from pygame_widgets.button import Button
import pygame_widgets
from camera import Camera
import imgui


class Editor:
    def __init__(self, screen, renderer): # Added renderer here
            self.lines = []
            self.snap_engine = SnapEngine(pixel_radius=10)
            
            # State
            self.is_drawing = False
            self.start_pos = None  
            self.current_mouse_pos = (0, 0)
            self.screen = screen
            self.renderer = renderer # Store it!
            self.finished = False
        

        

        
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
        # 1. Convert Screen (Pixels) -> World (Meters)
        # You need to implement screen_to_world in your Camera class!
        world_mouse_pos = camera.screen_to_world(screen_mouse_pos)

        # 2. Snap Logic now happens in World Space (consistent with your Lines)
        snapped_pos = self.snap_engine.get_snapped_pos(
            world_mouse_pos, 
            self.lines, 
            camera.scale,
            self.start_pos
        )

        if not self.is_drawing:
            self.start_pos = snapped_pos
            self.is_drawing = True
        else:
            new_line = Line(self.start_pos, snapped_pos)
            self.lines.append(new_line)
            self.start_pos = snapped_pos

    def _cancel_or_undo(self):
        if self.is_drawing:
            self.is_drawing = False
            self.start_pos = None
        elif self.lines:
            self.lines.pop()
            

    def draw(self, screen, camera):
        # 1. Start ImGui Frame
        imgui.new_frame()
        
        # 2. Define the Main UI
        imgui.set_next_window_position(50, 50, imgui.ALWAYS)
        imgui.begin("Controls", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
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

        # 4. Draw preview line and Floating Tooltip
        if self.is_drawing and self.start_pos:
            world_mouse = camera.screen_to_world(self.current_mouse_pos)
            target_world_pos = self.snap_engine.get_snapped_pos(
                world_mouse, self.lines, camera.scale, self.start_pos
            )
            
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