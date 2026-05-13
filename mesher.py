import numpy as np
from line import Line
from matplotlib.path import Path
import numpy as np
from bowyerwatson import Bowyer_watson
from point import Point
import pygame
import constructor as ct
from quad import Quad
from shapely.geometry import Polygon as ShapelyPoly
from shapely.geometry import Point as ShapelyPoint
from triangle import Triangle

import cProfile
import pstats

import imgui
from imgui.integrations.pygame import PygameRenderer


class Mesher:
    def __init__(self, screen, lines, n_layers, growth_factor, thickness, spacing, r, RENDERER,
                 unit_to_meters=0.001):
        """
        unit_to_meters: conversion factor from world-units (CAD coords) to SI metres.
                        0.001 for mm (default), 0.01 for cm, 1.0 for m.
        All geometry parameters (thickness, spacing, r) must be in the same world-unit.
        The solver receives everything converted to metres.
        """
        self.lines = lines
        self.points = None
        self.boundary_points = None
        self.candidate_points = None
        self.triangulation = None
        self.orientation = None
        self.thickness_mask = None

        # Unit conversion
        self.unit_to_meters = unit_to_meters

        # Boundary layers
        self.n_layers = n_layers
        self.growth_factor = growth_factor
        self.thickness = thickness
        self.boundary_spacing = spacing

        # Mesh generation
        self.r = r

        self.renderer = RENDERER
        self.finished = False

    def mesh(self):
        # 1. Get the lines in the correct loop order
        ordered_lines = self.build_polygon()

        # 2. Calculate orientation
        vert_coords = [(line.a.x, line.a.y) for line in ordered_lines]
        vert_array = np.array(vert_coords)
        self.orientation = self.polygon_orientation(vert_array)

        # 3. Create high-res boundary points
        self.create_boundary_points(ordered_lines)

        # 4. Generate Layers
        layers = [self.boundary_points]
        for i in range(self.n_layers):
            current_factor_array = self.thickness_mask * self.thickness
            next_layer = self.boundary_layer(layers[-1], current_factor_array)
            layers.append(next_layer)
            self.thickness *= self.growth_factor

        # 5. Connect them into Quads
        self.boundary_elements = self.connect_layers(layers)

        # --- PHASE 2: Unstructured Interior ---
        inner_ring = layers[-1]
        self.create_steiner_points(inner_ring, self.r)

        inner_ring_pts = [Point(p[0], p[1]) for p in inner_ring]
        steiner_pts = [Point(p[0], p[1]) for p in self.points]
        all_interior_pts = inner_ring_pts + steiner_pts

        self.triangulation = Bowyer_watson(all_interior_pts)
        self.filter_triangles(inner_ring)

        message = 'generated ' + repr(len(self.triangulation.triangles)) + ' cells'
        print(message)

    def create_boundary_points(self, ordered_lines):
        all_points = []
        all_thicknesses = []
        all_bc_tags = []

        bc_map = {"Wall": 0, "Velocity Inlet": 1, "Pressure Outlet": 2, "Inlet": 1, "Outlet": 2}
        num_l = len(ordered_lines)

        for i in range(num_l):
            line = ordered_lines[i]
            next_line = ordered_lines[(i + 1) % num_l]
            prev_line = ordered_lines[(i - 1) % num_l]

            start = np.array([line.a.x, line.a.y])
            end = np.array([next_line.a.x, next_line.a.y])

            line_vec = end - start
            line_length = np.linalg.norm(line_vec)
            if line_length == 0:
                continue

            n_points = max(1, int(np.floor(line_length / self.boundary_spacing)))
            segment_points = np.linspace(start, end, n_points, endpoint=False)
            all_points.append(segment_points)

            if line.boundary_type == "Wall":
                seg_thick = np.ones((n_points, 1))
            else:
                seg_thick = np.zeros((n_points, 1))

            if line.boundary_type != "Wall" and prev_line.boundary_type == "Wall":
                seg_thick[0] = 1.0
            all_thicknesses.append(seg_thick)

            tag = bc_map.get(line.boundary_type, 0)
            seg_tags = np.full(n_points, tag)
            all_bc_tags.append(seg_tags)

        self.boundary_points = np.vstack(all_points)
        self.thickness_mask = np.vstack(all_thicknesses)
        self.point_bc_mask = np.concatenate(all_bc_tags)

    def check_points(self):
        polygon_path = Path(self.boundary_points)
        mask = polygon_path.contains_points(self.points)
        steiner_points = self.points[mask]

    def create_steiner_points(self, boundary_points, r=550, k=30):
        if boundary_points is None or len(boundary_points) < 3:
            raise ValueError("Boundary polygon not defined properly.")

        full_poly = ShapelyPoly(boundary_points)
        safe_zone = full_poly.buffer(-r * 0.8)
        xmin, ymin, xmax, ymax = full_poly.bounds

        w = r / np.sqrt(2)
        cols = int(np.ceil((xmax - xmin) / w))
        rows = int(np.ceil((ymax - ymin) / w))

        if cols * rows > 500000:
            print(f"Warning: Grid too dense ({cols}x{rows}). Increase 'r'.")
            self.points = np.array([])
            return

        grid = np.full((cols, rows), None, dtype=object)
        points = []
        active = []

        def get_grid_coords(p):
            gx = int((p[0] - xmin) / w)
            gy = int((p[1] - ymin) / w)
            return gx, gy

        found_start = False
        attempts = 0
        while not found_start and attempts < 1000:
            attempts += 1
            p0 = np.random.uniform([xmin, ymin], [xmax, ymax])
            if safe_zone.contains(ShapelyPoint(p0)):
                points.append(p0)
                active.append(p0)
                gx, gy = get_grid_coords(p0)
                grid[gx, gy] = p0
                found_start = True

        if not found_start:
            print("Could not find a starting point inside the polygon!")
            self.points = np.array([])
            return

        while active:
            idx = np.random.randint(len(active))
            base_point = active[idx]
            found = False

            for _ in range(k):
                angle = np.random.uniform(0, 2 * np.pi)
                rad = np.random.uniform(r, 2 * r)
                candidate = base_point + rad * np.array([np.cos(angle), np.sin(angle)])

                if not (xmin <= candidate[0] <= xmax and ymin <= candidate[1] <= ymax):
                    continue

                gx, gy = get_grid_coords(candidate)
                is_far_enough = True
                for i in range(max(0, gx - 2), min(cols, gx + 3)):
                    for j in range(max(0, gy - 2), min(rows, gy + 3)):
                        neighbor = grid[i, j]
                        if neighbor is not None:
                            if np.linalg.norm(candidate - neighbor) < r:
                                is_far_enough = False
                                break
                    if not is_far_enough:
                        break

                if not is_far_enough:
                    continue

                if not safe_zone.contains(ShapelyPoint(candidate)):
                    continue

                points.append(candidate)
                active.append(candidate)
                grid[gx, gy] = candidate
                found = True
                break

            if not found:
                active.pop(idx)

        self.points = np.array(points)

    def build_polygon(self):
        remaining = self.lines.copy()
        first = remaining.pop(0)
        ordered_lines = [first]
        pivot = first.b

        while remaining:
            found = False
            for i, line in enumerate(remaining):
                if np.array_equal(pivot, line.a):
                    pivot = line.b
                    ordered_lines.append(line)
                    remaining.pop(i)
                    found = True
                    break
                elif np.array_equal(pivot, line.b):
                    pivot = line.a
                    ordered_lines.append(line)
                    remaining.pop(i)
                    found = True
                    break

            if not found:
                raise ValueError("Lines do not form a closed loop")

        return ordered_lines

    def polygon_orientation(self, polygon_array):
        x = polygon_array[:, 0]
        y = polygon_array[:, 1]
        area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        return area  # area < 0 is CCW, area > 0 is CW

    def draw(self, screen, camera):
        imgui.new_frame()

        if hasattr(self, "lines"):
            for line in self.lines:
                line.draw(screen, camera, color=(255, 255, 255), width=2)

        if hasattr(self, 'boundary_elements'):
            for quad in self.boundary_elements:
                quad.draw(screen, camera)

        if hasattr(self, "triangulation") and self.triangulation:
            self.triangulation.draw(screen, camera)

        imgui.begin("Solver")
        if imgui.button("Proceed to Solving"):
            self.finish()
        imgui.end()

        imgui.render()
        self.renderer.render(imgui.get_draw_data())

    def create_boundary_layers(self, n_layers=1, scaling_factor=4):
        layers = [self.boundary_points]
        for _ in range(n_layers):
            last_layer = layers[-1]
            new_layer = self.boundary_layer(last_layer, scaling_factor=scaling_factor)
            layers.append(new_layer)
        return layers

    def boundary_layer(self, polygon_points, scaling_factor):
        n = len(polygon_points)
        new_points = np.zeros_like(polygon_points)

        next_points = np.roll(polygon_points, -1, axis=0)
        edges = next_points - polygon_points
        edge_lengths = np.linalg.norm(edges, axis=1)[:, np.newaxis]
        unit_edges = edges / edge_lengths

        if self.orientation > 0:
            edge_normals = np.column_stack([-unit_edges[:, 1], unit_edges[:, 0]])
        else:
            edge_normals = np.column_stack([unit_edges[:, 1], -unit_edges[:, 0]])

        prev_edge_normals = np.roll(edge_normals, 1, axis=0)
        vertex_normals = prev_edge_normals + edge_normals
        v_norm = np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]
        vertex_normals = np.where(v_norm > 1e-9, vertex_normals / v_norm, edge_normals)

        cos_theta = np.sum(vertex_normals * edge_normals, axis=1)[:, np.newaxis]
        miter_length = scaling_factor / np.maximum(cos_theta, 0.1)

        new_points = polygon_points + vertex_normals * miter_length
        return new_points

    def connect_layers(self, layers):
        import math
        elements = []

        for i in range(len(layers) - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]
            num_points = len(current_layer)

            for j in range(num_points):
                j_next = (j + 1) % num_points

                p1 = self._to_point(current_layer[j])
                p2 = self._to_point(current_layer[j_next])
                p3 = self._to_point(next_layer[j_next])
                p4 = self._to_point(next_layer[j])

                area = 0.5 * abs((p1.x*p2.y + p2.x*p3.y + p3.x*p4.y + p4.x*p1.y) -
                                  (p2.x*p1.y + p3.x*p2.y + p4.x*p3.y + p1.x*p4.y))
                if area < 1e-4:
                    continue

                tol = 1e-4
                d12 = math.hypot(p1.x - p2.x, p1.y - p2.y)
                d23 = math.hypot(p2.x - p3.x, p2.y - p3.y)
                d34 = math.hypot(p3.x - p4.x, p3.y - p4.y)
                d41 = math.hypot(p4.x - p1.x, p4.y - p1.y)

                if d12 < tol:
                    elements.append(Triangle(p2, p3, p4))
                elif d23 < tol:
                    elements.append(Triangle(p1, p3, p4))
                elif d34 < tol:
                    elements.append(Triangle(p1, p2, p3))
                elif d41 < tol:
                    elements.append(Triangle(p1, p2, p3))
                else:
                    elements.append(Quad(p1, p2, p3, p4))

        return elements

    def _to_point(self, p):
        if isinstance(p, Point):
            return p
        return Point(p[0], p[1])

    def filter_triangles(self, inner_ring_points):
        ring_path = Path(inner_ring_points)
        centroids_coords = [[t.centroid.x, t.centroid.y] for t in self.triangulation.triangles]
        centroids = np.array(centroids_coords, dtype=np.float64)
        mask = ring_path.contains_points(centroids, radius=-1e-5)

        valid_triangles = []
        for is_inside, triangle in zip(mask, self.triangulation.triangles):
            if is_inside:
                valid_triangles.append(triangle)
        self.triangulation.triangles = valid_triangles

    # ------------------------------------------------------------------
    def solver_data_pipeline(self):
        """
        Builds the mesh data dict for the Solver.
        ALL coordinates and distances are converted to SI metres by multiplying
        with self.unit_to_meters.  Areas are multiplied by unit_to_meters².
        """
        print('Beginning data collection for Solver...')
        s  = self.unit_to_meters      # length scale factor
        s2 = s * s                    # area scale factor

        def get_edge_key(p_a, p_b):
            ax = p_a.x if hasattr(p_a, 'x') else p_a[0]
            ay = p_a.y if hasattr(p_a, 'y') else p_a[1]
            bx = p_b.x if hasattr(p_b, 'x') else p_b[0]
            by = p_b.y if hasattr(p_b, 'y') else p_b[1]
            k1 = (round(float(ax), 6), round(float(ay), 6))
            k2 = (round(float(bx), 6), round(float(by), 6))
            return tuple(sorted([k1, k2]))

        # 1. BC lookup (in world units — used only for tagging, no conversion needed)
        bc_lookup = {}
        for i in range(len(self.boundary_points)):
            p1 = self.boundary_points[i]
            p2 = self.boundary_points[(i + 1) % len(self.boundary_points)]
            edge_key = get_edge_key(p1, p2)
            bc_lookup[edge_key] = self.point_bc_mask[i]

        # 2. Gather all cells
        valid_boundary_elements = [c for c in self.boundary_elements if float(c.area) > 1e-8]
        Cells = valid_boundary_elements + self.triangulation.triangles
        Nc = len(Cells)

        # Cell centers and areas (world units → converted to SI below)
        cell_centers_wu = np.array([[c.centroid.x, c.centroid.y] for c in Cells], dtype=np.float64)
        cell_areas_wu   = np.array([float(c.area) for c in Cells], dtype=np.float64)

        # 3. Build Edge Map
        edge_map = {}
        for cell_id, cell in enumerate(Cells):
            for edge in cell.edges():
                if len(edge) != 2:
                    continue
                p_a, p_b = edge
                key = get_edge_key(p_a, p_b)
                if key not in edge_map:
                    edge_map[key] = []
                edge_map[key].append(cell_id)

        # 4. Populate Face Arrays (still in world units at this point)
        Nf = len(edge_map)
        owner    = np.zeros(Nf, dtype=np.int32)
        neighbor = np.full(Nf, -1, dtype=np.int32)
        Sf_wu    = np.zeros((Nf, 2))
        Cf_wu    = np.zeros((Nf, 2))
        df_wu    = np.zeros((Nf, 2))
        magDf_wu = np.zeros(Nf)
        boundary_tags = np.full(Nf, -1)

        print(f"DEBUG: bc_lookup contains {len(bc_lookup)} total boundary edges.")
        print(f"DEBUG: Unique tags found in lookup: {set(bc_lookup.values())}")
        boundary_edge_count = sum(1 for ids in edge_map.values() if len(ids) == 1)
        print(f"DEBUG: edge_map has {boundary_edge_count} boundary candidates.")

        for face_idx, (edge_key, cell_ids) in enumerate(edge_map.items()):
            owner[face_idx] = cell_ids[0]

            if len(cell_ids) > 1:
                neighbor[face_idx] = cell_ids[1]
                boundary_tags[face_idx] = -1
            else:
                p1_raw, p2_raw = edge_key
                face_mid = (np.array(p1_raw) + np.array(p2_raw)) / 2.0

                assigned_tag = 0
                min_dist = float('inf')
                for i in range(len(self.boundary_points)):
                    b1 = self.boundary_points[i]
                    b2 = self.boundary_points[(i + 1) % len(self.boundary_points)]
                    b_mid = (b1 + b2) / 2.0
                    dist = np.linalg.norm(face_mid - b_mid)
                    if dist < min_dist:
                        min_dist = dist
                        assigned_tag = self.point_bc_mask[i]

                # Tolerance stays in world units (e.g. 1 mm) — no conversion needed here
                if min_dist < 1.0:
                    boundary_tags[face_idx] = assigned_tag
                else:
                    boundary_tags[face_idx] = 0

            # Geometry (world units)
            p1_coords, p2_coords = edge_key
            p1 = np.array(p1_coords)
            p2 = np.array(p2_coords)
            face_center = (p1 + p2) / 2.0
            vec    = p2 - p1
            normal = np.array([vec[1], -vec[0]])

            owner_c = cell_centers_wu[owner[face_idx]]
            if np.dot(normal, face_center - owner_c) < 0:
                normal = -normal

            Sf_wu[face_idx] = normal
            Cf_wu[face_idx] = face_center

            if neighbor[face_idx] != -1:
                df_vec = cell_centers_wu[neighbor[face_idx]] - owner_c
            else:
                df_vec = face_center - owner_c

            df_wu[face_idx]    = df_vec
            magDf_wu[face_idx] = np.linalg.norm(df_vec)

        # ----------------------------------------------------------------
        # 5. UNIT CONVERSION  — world units → SI metres
        #    Lengths × s,  Areas × s²,  Normals (Sf) are edge-length vectors × s
        # ----------------------------------------------------------------
        cell_centers_si = cell_centers_wu * s
        cell_areas_si   = cell_areas_wu   * s2
        Sf_si           = Sf_wu   * s
        Cf_si           = Cf_wu   * s
        df_si           = df_wu   * s
        magDf_si        = magDf_wu * s
        magSf_si        = np.linalg.norm(Sf_si, axis=1)

        print(f"Unit conversion applied: ×{s} (lengths), ×{s2:.2e} (areas)")
        print(f"  cell_centers range: [{cell_centers_si.min():.4f}, {cell_centers_si.max():.4f}] m")
        print(f"  cell_areas  range:  [{cell_areas_si.min():.4e}, {cell_areas_si.max():.4e}] m²")

        cells_in_faces = set(owner) | set(neighbor[neighbor != -1])
        all_cells      = set(range(Nc))
        orphan_cells   = all_cells - cells_in_faces
        if orphan_cells:
            print(f"⚠️  MESHER BUG: {len(orphan_cells)} cells have NO faces!")

        return {
            'Nc':           Nc,
            'Nf':           Nf,
            'owner':        owner,
            'neighbor':     neighbor,
            'Sf':           Sf_si,
            'magSf':        magSf_si,
            'Cf':           Cf_si,
            'df':           df_si,
            'magDf':        magDf_si,
            'cell_centers': cell_centers_si,
            'cell_areas':   cell_areas_si,
            'boundary_tags': boundary_tags,
        }

    def finish(self):
        self.finished = True