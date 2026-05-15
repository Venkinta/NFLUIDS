class Triangulation:
    def __init__(self, triangles=None):
        self.triangles = []
        self.coords = [] # Parallel list for instant NumPy conversion
        if triangles is not None:
            for t in triangles:
                self.add_triangle(t)

    def add_triangle(self, triangle):    
        self.triangles.append(triangle)
        # Store flat floats. This is incredibly cache-friendly.
        self.coords.append((triangle.a.x, triangle.a.y, 
                            triangle.b.x, triangle.b.y, 
                            triangle.c.x, triangle.c.y))

    def remove_triangle(self, triangle):
        try:
            # Pop both lists at the same index to keep them synced
            idx = self.triangles.index(triangle)
            self.triangles.pop(idx)
            self.coords.pop(idx)
        except ValueError:
            pass

    def draw(self, screen, camera, color=(0, 0, 255), width=2):
        for triangle in self.triangles:
            triangle.draw(screen, camera, color, width)