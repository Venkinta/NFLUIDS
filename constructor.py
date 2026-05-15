from triangulation import Triangulation
from triangle import Triangle
from point import Point
import numpy as np 

def create_super_triangle(points):
    """Creates a triangle large enough to contain all points with massive padding."""
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)

    dx = max_x - min_x
    dy = max_y - min_y
    dmax = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    p1 = Point(mid_x - 20 * dmax, mid_y - dmax)
    p2 = Point(mid_x + 20 * dmax, mid_y - dmax)
    p3 = Point(mid_x, mid_y + 20 * dmax)

    return Triangle(p1, p2, p3)

def checkCircumcentre(triangle, point):
    """Standard determinant-based circumcircle check."""
    orientCCW(triangle)
    
    ax, ay = triangle.a.x - point.x, triangle.a.y - point.y
    bx, by = triangle.b.x - point.x, triangle.b.y - point.y
    cx, cy = triangle.c.x - point.x, triangle.c.y - point.y

    det = (
        (ax*ax + ay*ay) * (bx*cy - cx*by) -
        (bx*bx + by*by) * (ax*cy - cx*ay) +
        (cx*cx + cy*cy) * (ax*by - bx*ay)
    )
    return det > 0   

def check_circum_bulk(coords_list, point):
    """Checks all triangles against one point using pure vectorization."""
    if not coords_list:
        return np.array([])
        
    # Instant array creation from flat tuples
    C = np.array(coords_list)
    px, py = point.x, point.y

    # Vectorized extraction and subtraction
    Ax = C[:, 0] - px
    Ay = C[:, 1] - py
    Bx = C[:, 2] - px
    By = C[:, 3] - py
    Cx = C[:, 4] - px
    Cy = C[:, 5] - py

    a2 = Ax**2 + Ay**2
    b2 = Bx**2 + By**2
    c2 = Cx**2 + Cy**2
    
    det = (a2 * (Bx*Cy - Cx*By) -
           b2 * (Ax*Cy - Cx*Ay) +
           c2 * (Ax*By - Bx*Ay))
    
    return det > 0

def orientCCW(triangle):
    a = triangle.a
    b = triangle.b
    c = triangle.c

    cross = (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)

    if cross < 0:
        triangle.b, triangle.c = triangle.c, triangle.b

def updatebadedges(edge_count, triangle):
    for edge in triangle.edges():
        edge_count[edge] = edge_count.get(edge, 0) + 1

#functions meant for mesher.py
def cross2d(u, v):
    return u[0]*v[1] - u[1]*v[0]

def intersect(line1, line2):
    p = np.array([line1.a.x, line1.a.y])
    r = line1.vector
    q = np.array([line2.a.x, line2.a.y])
    s = line2.vector

    rxs = cross2d(r, s)
    if abs(rxs) < 1e-9:
        return None  

    t = cross2d(q - p, s) / rxs
    return p + t * r