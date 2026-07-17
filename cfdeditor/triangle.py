from .point import Point

class Triangle:
    def __init__(self, a,b,c,neighbours=None):
        if neighbours is None:
            neighbours = []
        self.a = a
        self.b = b
        self.c = c
        self.neighbours = neighbours
        
    def vertices(self):
        return self.a,self.b,self.c
        
    def edges(self):
        return [
            frozenset((self.a, self.b)),
            frozenset((self.b, self.c)),
            frozenset((self.c, self.a)),
        ]
    
    # In triangle.py (or wherever your Triangle class is)
    @property
    def centroid(self):
        cx = (self.a.x + self.b.x + self.c.x) / 3.0
        cy = (self.a.y + self.b.y + self.c.y) / 3.0
        return Point(cx, cy) # Change this from a tuple to a Point object
    
    @property
    def area(self):
        x1, y1 = self.a.x, self.a.y
        x2, y2 = self.b.x, self.b.y
        x3, y3 = self.c.x, self.c.y

        return abs(((x2-x1) * (y3-y1) - (y2-y1)*(x3-x1))*0.5)



    def draw(self, screen, camera, color=(100, 100, 250), width=1):
            # We pass the list of 3 Point objects directly
        camera.draw_polygon([self.a, self.b, self.c], screen, color, width)

