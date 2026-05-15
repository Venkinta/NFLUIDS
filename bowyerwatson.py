import constructor as ct
from point import Point
from triangle import Triangle
from triangulation import Triangulation 
import numpy as np

def Bowyer_watson(input_points):
    # 1. Deduplicate and sort points immediately (improves cache locality)
    points = sorted(list(set(input_points)), key=lambda p: (p.x, p.y)) 
    
    triangulation = Triangulation()
    super_triangle = ct.create_super_triangle(points)
    ct.orientCCW(super_triangle) # Ensure CCW from the start
    triangulation.add_triangle(super_triangle)

    for point in points:
        # 2. Safely check all existing triangles using fast C-speed NumPy math
        # We now pass triangulation.coords instead of triangulation.triangles
        if triangulation.coords: 
            mask = ct.check_circum_bulk(triangulation.coords, point)
            badTriangles = [triangulation.triangles[i] for i, is_bad in enumerate(mask) if is_bad]
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

    # Cleanup: Remove triangles connected to the super-triangle vertices
    super_verts = set(super_triangle.vertices())
    
    triangulation.triangles = [
        t for t in triangulation.triangles 
        if not any(v in super_verts for v in t.vertices())
    ]

    return triangulation