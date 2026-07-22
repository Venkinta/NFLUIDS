import math


def triangle_min_angle(triangle):
    """Smallest interior angle of a triangle, in degrees."""
    a, b, c = triangle.a, triangle.b, triangle.c
    lab = math.hypot(b.x - a.x, b.y - a.y)
    lbc = math.hypot(c.x - b.x, c.y - b.y)
    lca = math.hypot(a.x - c.x, a.y - c.y)

    def angle_opposite(opposite, s1, s2):
        cos_a = (s1 * s1 + s2 * s2 - opposite * opposite) / (2.0 * s1 * s2)
        cos_a = max(-1.0, min(1.0, cos_a))
        return math.degrees(math.acos(cos_a))

    angle_a = angle_opposite(lbc, lab, lca)
    angle_b = angle_opposite(lca, lab, lbc)
    angle_c = 180.0 - angle_a - angle_b
    return min(angle_a, angle_b, angle_c)


def seam_quality(triangulation, ring_points):
    """Worst (smallest) min-angle among true seam triangles.

    A seam triangle straddles the ring/interior boundary: at least one
    vertex is a ring point AND at least one is not. A triangle with all
    three vertices on the ring is a corner artifact of the boundary layer
    itself (frozen geometry, out of reach for Steiner-point smoothing), not
    a seam, and is excluded so it can't dominate the metric.

    Returns the seam's weakest triangle as a single scalar so a smoothing
    pass can be compared before/after. +inf if no seam triangles exist.
    """
    worst = math.inf
    for t in triangulation.triangles:
        verts = (t.a, t.b, t.c)
        ring_count = sum(v in ring_points for v in verts)
        if 0 < ring_count < 3:
            worst = min(worst, triangle_min_angle(t))
    return worst
