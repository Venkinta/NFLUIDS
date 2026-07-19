from .point import Point

class Camera:
    """Pure view math: world<->screen transforms and cursor-anchored zoom.
    All drawing lives in renderer.Renderer (which holds this camera)."""
    def __init__(self, scale=2.0, offset=None):
        # scale = pixels per world-unit (world units = mm by default)
        # scale=2.0 means 1mm = 2px  →  1280px screen shows 640mm wide
        self.scale = scale
        self.offset = offset if offset is not None else [0.0, 0.0]

    def to_screen(self, world_point):
        """Converts World (mm) to Screen (Pixels)."""
        px = world_point.x if hasattr(world_point, 'x') else world_point[0]
        py = world_point.y if hasattr(world_point, 'y') else world_point[1]
        
        sx = (px + self.offset[0]) * self.scale
        sy = (py + self.offset[1]) * self.scale
        return (sx, sy)

    def screen_to_world(self, screen_pos):
        """Converts Screen (Pixels) to World (mm)."""
        sx, sy = screen_pos
        wx = (sx / self.scale) - self.offset[0]
        wy = (sy / self.scale) - self.offset[1]
        return Point(wx, wy)

    def handle_zoom(self, mouse_pos, scroll_y):
        mx, my = mouse_pos
        zoom_factor = 1.1 if scroll_y > 0 else 0.9
        
        # Adjust offset to keep mouse point anchored
        self.offset[0] = mx / (self.scale * zoom_factor) - (mx / self.scale - self.offset[0])
        self.offset[1] = my / (self.scale * zoom_factor) - (my / self.scale - self.offset[1])
        
        self.scale *= zoom_factor

