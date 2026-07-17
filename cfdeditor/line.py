from .point import Point
import numpy as np

class Line:
    def __init__(self, a, b):
        # Convert to Point objects if they are passed as tuples/lists
        self.a = a if isinstance(a, Point) else Point(a[0], a[1])
        self.b = b if isinstance(b, Point) else Point(b[0], b[1])
        
        
        self.boundary_type = "Wall"
        self.u_val = 0.0
        self.v_val = 0.0
        self.p_val = 0.0
        
    def draw(self, screen, camera, color=(255, 255, 255), width=1):
        # Corrected method name
        camera.draw_line(screen, self, color, width)
        
    @property   
    def vector(self):
        # Now this safely works because self.a and self.b are Point objects
        return [self.b.x - self.a.x, self.b.y - self.a.y]
    
    def is_mouse_over(self, mouse_world_pos, camera, pixel_threshold=10):
        # Convert fixed pixel sensitivity to dynamic world sensitivity
        world_threshold = pixel_threshold / camera.scale
        
        # Vector from line start to line end (AB)
        A = np.array([self.a.x, self.a.y])
        B = np.array([self.b.x, self.b.y])
        M = np.array([mouse_world_pos.x, mouse_world_pos.y])
        
        AB = B - A
        AM = M - A
        
        line_mag_sq = np.dot(AB, AB)
        if line_mag_sq == 0: return False
        
        t = np.dot(AM, AB) / line_mag_sq
        t = max(0, min(1, t))
        
        nearest_point = A + t * AB
        distance = np.linalg.norm(M - nearest_point)
        
        # Check against the dynamic threshold
        return distance < world_threshold