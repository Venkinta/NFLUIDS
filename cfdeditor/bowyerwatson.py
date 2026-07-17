from . import constructor as ct
from .triangle import Triangle
from .triangulation import Triangulation
import numpy as np

def Bowyer_watson(input_points):
    # 1. Deduplicate and sort points immediately (improves cache locality)
    points = sorted(list(set(input_points)), key=lambda p: (p.x, p.y)) 
    
    triangulation = Triangulation()
    super_triangle = ct.create_super_triangle(points)
    ct.orientCCW(super_triangle) # Ensure CCW from the start
    triangulation.add_triangle(super_triangle)

    for point in points:
        if triangulation.triangles:
            mask = ct.check_circum_bulk(triangulation.coords, point)
            # np.nonzero runs at C speed — extracts bad indices without any
            # Python loop over all N triangles. We then only iterate the small
            # number of bad ones (typically 3-10), not the full triangulation.
            bad_indices = np.nonzero(mask)[0]
            badTriangles = [triangulation.triangles[i] for i in bad_indices]
        else:
            badTriangles = []

        edge_count = {}

        # Count occurrences of edges 
        for triangle in badTriangles:
            for edge in triangle.edges():
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # The hole's boundary consists of edges appearing exactly once
        polygon = [edge for edge, count in edge_count.items() if count == 1]

        # Remove bad triangles 
        for t in badTriangles:
            triangulation.remove_triangle(t) 

        # Re-triangulate the hole
        for edge in polygon:
            v1, v2 = list(edge)
            newTriangle = Triangle(v1, v2, point)
            
            # Orient CCW immediately so check_circum_bulk math holds up on the next loop
            ct.orientCCW(newTriangle) 
            triangulation.add_triangle(newTriangle)

    # Cleanup: Remove triangles sharing a vertex with the super-triangle.
    # Must use remove_triangle() to keep _tri_to_idx and _coords_np in sync.
    super_verts = set(super_triangle.vertices())
    to_remove = [t for t in triangulation.triangles
                 if any(v in super_verts for v in t.vertices())]
    for t in to_remove:
        triangulation.remove_triangle(t)

    return triangulation