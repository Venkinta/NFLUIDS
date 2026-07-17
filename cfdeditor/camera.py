from OpenGL.GL import *
import math
from .point import Point

class Camera:
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

    # --- DRAWING METHODS (OPENGL) ---

    def draw_line(self, screen, line, color=(255, 255, 255), width=1):
        """Draws a CAD Line object (handles world-to-screen conversion)."""
        p0 = self.to_screen(line.a)
        p1 = self.to_screen(line.b)
        self.draw_screen_line(screen, p0, p1, color, width)

    def draw_screen_line(self, screen, p0, p1, color=(255, 255, 255), width=1):
        """Draws a line using raw screen coordinates (Pixels). Useful for UI/Previews."""
        r, g, b = [c/255.0 for c in color]
        glLineWidth(width)
        glBegin(GL_LINES)
        glColor3f(r, g, b)
        glVertex2f(p0[0], p0[1])
        glVertex2f(p1[0], p1[1])
        glEnd()

    def draw_polygon(self, polygon_vertices, screen, color=(100, 100, 250), width=1):
        """Draws Triangles/Quads (handles world-to-screen conversion)."""
        r, g, b = [c/255.0 for c in color]
        screen_points = [self.to_screen(p) for p in polygon_vertices]

        if width == 0: # Filled
            glBegin(GL_POLYGON)
        else: # Outline
            glLineWidth(width)
            glBegin(GL_LINE_LOOP)
            
        glColor3f(r, g, b)
        for p in screen_points:
            glVertex2f(p[0], p[1])
        glEnd()

    def draw_circle(self, screen, color, center_screen, radius, width=1):
        """Draws a circle using screen coordinates. Useful for snapping points."""
        r, g, b = [c/255.0 for c in color]
        glColor3f(r, g, b)
        glLineWidth(width)
        glBegin(GL_LINE_LOOP)
        for i in range(32):
            angle = 2 * math.pi * i / 32
            x = center_screen[0] + math.cos(angle) * radius
            y = center_screen[1] + math.sin(angle) * radius
            glVertex2f(x, y)
        glEnd()
        
        
    def apply_gl_transform(self):
        """Applies the camera's zoom and pan to the OpenGL matrix pipeline."""
        glPushMatrix()
        glScalef(self.scale, self.scale, 1.0)
        glTranslatef(self.offset[0], self.offset[1], 0.0)

    def remove_gl_transform(self):
        """Pops the camera transform off the stack."""
        glPopMatrix()

    def draw_vbo(self, vbo_id, vertex_count, color=(100, 255, 100), mode=GL_LINES):
        """Draws a VBO using high-performance line rendering."""
        r, g, b = [c/255.0 for c in color]
        glColor3f(r, g, b)

        self.apply_gl_transform()

        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        glVertexPointer(2, GL_FLOAT, 0, None)

        # Use the provided mode (GL_LINES for wireframe)
        glDrawArrays(mode, 0, vertex_count)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        self.remove_gl_transform()