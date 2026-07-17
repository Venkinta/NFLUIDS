from .point import Point


class Quad:
    def __init__(self, p1, p2, p3, p4):
        # Stored in counter-clockwise order
        self.points = [p1, p2, p3, p4]

    def vertices(self):
        return self.points

    def edges(self):
        # A quad has 4 edges
        return [
            frozenset([self.points[0], self.points[1]]),
            frozenset([self.points[1], self.points[2]]),
            frozenset([self.points[2], self.points[3]]),
            frozenset([self.points[3], self.points[0]])
        ]
    
    @property
    def centroid(self):
        area = 0.0
        cx = 0.0  # <--- Initialize cx
        cy = 0.0  # <--- Initialize cy
        
        pts = self.points
        n = len(pts)
        
        for i in range(n):
            x0, y0 = pts[i].x, pts[i].y
            x1, y1 = pts[(i+1) % n].x, pts[(i+1) % n].y
            
            cross = x0 * y1 - x1 * y0
            area += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        
        area *= 0.5
        
        if abs(area) < 1e-12:
            # error handling for degenerate quads
            avg_x = sum(p.x for p in pts) / n
            avg_y = sum(p.y for p in pts) / n
            return Point(avg_x, avg_y)
        
        # The Shoelace centroid formula requires 6 * area
        cx /= (6.0 * area)
        cy /= (6.0 * area)
        
        return Point(cx, cy)
    
    
    @property
    def area(self):
        pts = self.points
        n = len(pts)
        area = 0.0

        for i in range(n):
            x0, y0 = pts[i].x, pts[i].y
            x1, y1 = pts[(i + 1) % n].x, pts[(i + 1) % n].y
            area += x0 * y1 - x1 * y0

        return abs(area) * 0.5
    
    def draw(self, screen, camera, color=(100, 255, 100), width=1):
        # self.points is already [p1, p2, p3, p4]
        camera.draw_polygon(self.points, screen, color, width)
