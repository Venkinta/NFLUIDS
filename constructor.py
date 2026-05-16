from triangulation import Triangulation
from triangle import Triangle
from point import Point
import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Numba JIT kernels — compiled once, cached to disk (cache=True).
# All inputs/outputs are plain Python floats or NumPy arrays so numba can
# run without the GIL and (for the bulk check) in parallel across cores.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _circum_scalar(ax, ay, bx, by, cx, cy):
    """Scalar circumcircle in-circle test.  Returns True if point is inside."""
    a2 = ax*ax + ay*ay
    b2 = bx*bx + by*by
    c2 = cx*cx + cy*cy
    det = (a2 * (bx*cy - cx*by)
         - b2 * (ax*cy - cx*ay)
         + c2 * (ax*by - bx*ay))
    return det > 0.0

@njit(parallel=True, cache=True)
def _check_circum_bulk_core(C, px, py):
    """Parallel bulk circumcircle test.

    C   : (N, 6) float64 array — each row is (ax,ay, bx,by, cx,cy)
    px, py : coordinates of the candidate point
    Returns a bool array of length N.

    Avoids the 6 temporary NumPy arrays that the vectorised version creates,
    improves cache locality (one row read per iteration), and uses all cores
    via numba's prange.
    """
    n = len(C)
    result = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        ax = C[i, 0] - px
        ay = C[i, 1] - py
        bx = C[i, 2] - px
        by = C[i, 3] - py
        cx = C[i, 4] - px
        cy = C[i, 5] - py
        a2 = ax*ax + ay*ay
        b2 = bx*bx + by*by
        c2 = cx*cx + cy*cy
        det = (a2 * (bx*cy - cx*by)
             - b2 * (ax*cy - cx*ay)
             + c2 * (ax*by - bx*ay))
        result[i] = det > 0.0
    return result

@njit(cache=True)
def _cross2d(ux, uy, vx, vy):
    """2-D cross product of two vectors given as scalar components."""
    return ux * vy - uy * vx

# ---------------------------------------------------------------------------
# Warm-up: trigger JIT compilation at import time so the first real mesh
# run pays no compilation overhead.
# ---------------------------------------------------------------------------
_warmup_C = np.zeros((1, 6), dtype=np.float64)
_check_circum_bulk_core(_warmup_C, 0.0, 0.0)
_circum_scalar(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
_cross2d(1.0, 0.0, 0.0, 1.0)
del _warmup_C


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
    """Standard determinant-based circumcircle check (delegates math to JIT kernel)."""
    orientCCW(triangle)
    ax, ay = triangle.a.x - point.x, triangle.a.y - point.y
    bx, by = triangle.b.x - point.x, triangle.b.y - point.y
    cx, cy = triangle.c.x - point.x, triangle.c.y - point.y
    return bool(_circum_scalar(ax, ay, bx, by, cx, cy))

def check_circum_bulk(coords_list, point):
    """Checks all triangles against one point using the parallel JIT kernel.

    coords_list : list/array of (ax,ay,bx,by,cx,cy) tuples — one per triangle
    point       : Point with .x and .y attributes
    Returns a bool NumPy array of length len(coords_list).
    """
    if not coords_list:
        return np.array([], dtype=bool)
    # np.asarray avoids a copy when coords_list is already an ndarray;
    # forces float64 so the JIT kernel always gets the right dtype.
    C = np.asarray(coords_list, dtype=np.float64)
    return _check_circum_bulk_core(C, float(point.x), float(point.y))

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
    """2-D cross product — thin wrapper around the JIT kernel."""
    return _cross2d(float(u[0]), float(u[1]), float(v[0]), float(v[1]))

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