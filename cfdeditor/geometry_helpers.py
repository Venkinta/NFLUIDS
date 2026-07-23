"""Geometry-builder helpers for scripting/parametric use (see api.py).

Promoted out of test_holes.py / test_force_balance.py, which had duplicated
copies of rect_lines() with contradictory edge-order docstrings. The order
below is the verified one (confirmed against boundary_tags counts on a real
mesh in test_force_balance.py): edges are built bottom, right, top, left.
"""
import numpy as np

from .point import Point
from .line import Line


def rect_lines(x0, y0, x1, y1, bc_map):
    """Build 4 Lines forming a closed rectangle, with per-edge BC types.

    bc_map: dict mapping edge index -> boundary_type string.
        0 = bottom (x0,y0)->(x1,y0)
        1 = right  (x1,y0)->(x1,y1)
        2 = top    (x1,y1)->(x0,y1)
        3 = left   (x0,y1)->(x0,y0)
    """
    bl, br, tr, tl = Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)
    edges = [(bl, br), (br, tr), (tr, tl), (tl, bl)]
    lines = []
    for i, (a, b) in enumerate(edges):
        ln = Line(a, b)
        ln.boundary_type = bc_map[i]
        lines.append(ln)
    return lines


def circle_lines(cx, cy, radius, bc_type="Wall", n_segments=32):
    """Build a closed polygon of Lines approximating a circle.

    A single bc_type applies to every segment — a circular hole (e.g. a
    cylinder in a channel) is normally one boundary type throughout.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    pts = [Point(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
    lines = []
    for i in range(n_segments):
        a, b = pts[i], pts[(i + 1) % n_segments]
        ln = Line(a, b)
        ln.boundary_type = bc_type
        lines.append(ln)
    return lines


def polygon_lines(points, bc_map=None, default_bc="Wall"):
    """Build a closed polygon of Lines from an ordered list of (x, y) points.

    points: list of (x, y) tuples, in order, NOT repeating the first point
        at the end (the polygon is closed automatically).
    bc_map: optional dict mapping edge index -> boundary_type string (edge i
        runs from points[i] to points[(i+1) % len(points)]). Missing indices
        fall back to default_bc.
    """
    n = len(points)
    pts = [Point(x, y) for x, y in points]
    lines = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ln = Line(a, b)
        ln.boundary_type = bc_map.get(i, default_bc) if bc_map else default_bc
        lines.append(ln)
    return lines
