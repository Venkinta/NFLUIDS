import numpy as np
from .line import Line
from matplotlib.path import Path
import numpy as np
from .bowyerwatson import Bowyer_watson
from .point import Point
from .quad import Quad
from shapely.geometry import Polygon as ShapelyPoly
from shapely.geometry import Point as ShapelyPoint
from .triangle import Triangle
from .renderer import VboHandle
import time
from OpenGL.GL import *


class Mesher:
    def __init__(self, lines, n_layers, growth_factor, thickness, spacing, r,
                 unit_to_meters=0.001, refinement_zones=None, bc_spacing_map=None):
        """
        unit_to_meters: conversion factor from world-units (CAD coords) to SI metres.
                        0.001 for mm (default), 0.01 for cm, 1.0 for m.
        All geometry parameters (thickness, spacing, r) must be in the same world-unit.
        The solver receives everything converted to metres.
        
        refinement_zones: list of (shapely_polygon, factor) tuples. Inside each polygon,
                          the Steiner point separation is r / factor, giving finer mesh.

        bc_spacing_map: dict mapping boundary-type string to per-type boundary spacing
                        (world units).  Falls back to `spacing` for any type not in the map.
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

        # Per-BC spacing (fall back to global spacing for missing keys)
        self.bc_spacing_map = bc_spacing_map if bc_spacing_map is not None else {}

        # Mesh generation
        self.r = r

        # Refinement zones: list of (shapely_polygon, factor)
        self.refinement_zones = refinement_zones if refinement_zones is not None else []

    def mesh(self):
        t_total = time.perf_counter()
        print("\n" + "="*50)
        print("  MESHER  —  starting pipeline")
        print("="*50)

        # 1. Group the CAD lines into separate closed loops (outer + holes)
        t = time.perf_counter()
        self.loops = self.build_polygon()
        n_edges = sum(len(l) for l in self.loops)
        print(f"[1/7] Polygon ordering      {time.perf_counter()-t:6.3f}s  "
              f"({n_edges} edges across {len(self.loops)} loops)")

        # 2. Per-loop orientation; the loop with the largest absolute area is
        #    the outer boundary, the rest are holes.
        t = time.perf_counter()
        self.loop_orientations = []
        for loop in self.loops:
            vert_array = np.array([(line.a.x, line.a.y) for line in loop])
            self.loop_orientations.append(self.polygon_orientation(vert_array))
        abs_areas = [abs(o) for o in self.loop_orientations]
        self.outer_idx = int(np.argmax(abs_areas))
        windings = ["CW" if o > 0 else "CCW" for o in self.loop_orientations]
        print(f"[2/7] Orientation           {time.perf_counter()-t:6.3f}s  "
              f"(outer=loop {self.outer_idx}; " + ", ".join(windings) + ")")

        # 3. High-res boundary points, sampled per loop then concatenated
        t = time.perf_counter()
        self.loop_boundary_points = []
        self.loop_thickness_masks = []
        self.loop_bc_masks        = []
        self.loop_point_counts    = []
        self.boundary_points = None
        self.thickness_mask  = None
        self.point_bc_mask   = None
        for loop in self.loops:
            bp, tm, bm = self.create_boundary_points(loop)
            self.loop_boundary_points.append(bp)
            self.loop_thickness_masks.append(tm)
            self.loop_bc_masks.append(bm)
            self.loop_point_counts.append(len(bp))
            self.boundary_points = (bp if self.boundary_points is None
                                    else np.vstack([self.boundary_points, bp]))
            self.thickness_mask = (tm if self.thickness_mask is None
                                   else np.vstack([self.thickness_mask, tm]))
            self.point_bc_mask = (bm if self.point_bc_mask is None
                                  else np.concatenate([self.point_bc_mask, bm]))
        print(f"[3/7] Boundary points       {time.perf_counter()-t:6.3f}s  "
              f"({len(self.boundary_points)} pts)")

        # 4. Boundary layers — extrude each loop independently.  The outer
        #    loop grows toward its interior (the domain); holes grow toward
        #    their *exterior* (also the domain), so their normal is flipped.
        t = time.perf_counter()
        # Capture the *original* first-layer thickness before the growth loop
        # mutates self.thickness.  solver_data_pipeline() saves this so a
        # reloaded mesh restores the value the user actually entered.
        self._orig_thickness = self.thickness
        full_bc_mask = self.point_bc_mask
        self.loop_layer_stacks = []
        for li, loop_pts in enumerate(self.loop_boundary_points):
            # Scope per-loop state consumed by boundary_layer()
            self.orientation   = self.loop_orientations[li]
            self.point_bc_mask = self.loop_bc_masks[li]
            is_outer = (li == self.outer_idx)
            layers = [loop_pts]
            for _ in range(self.n_layers):
                current_factor_array = self.loop_thickness_masks[li] * self.thickness
                next_layer = self.boundary_layer(
                    layers[-1], current_factor_array,
                    extrude_toward_interior=is_outer)
                layers.append(next_layer)
                self.thickness *= self.growth_factor
            self.loop_layer_stacks.append(layers)
        self.point_bc_mask = full_bc_mask  # restore full concatenated mask
        self.boundary_elements = []
        for stack in self.loop_layer_stacks:
            self.boundary_elements.extend(self.connect_layers(stack))
        print(f"[4/7] Boundary layers       {time.perf_counter()-t:6.3f}s  "
              f"({self.n_layers} layers × {len(self.loops)} loops)")

        # 5. Steiner points for the interior (domain may contain holes)
        t = time.perf_counter()
        inner_rings = [stack[-1] for stack in self.loop_layer_stacks]
        self.create_steiner_points(inner_rings, self.r)
        n_steiner = (len(self.points) if self.points is not None
                     and len(self.points) > 0 else 0)
        print(f"[6/7] Steiner points        {time.perf_counter()-t:6.3f}s  ({n_steiner} interior pts)")

        # 6. Bowyer-Watson triangulation + filter (compound region)
        t = time.perf_counter()
        # Collect inner ring points from ALL loops (outer + holes) to ensure triangles connect to them
        inner_ring_pts = []
        for ring in inner_rings:
            for p in ring:
                inner_ring_pts.append(Point(p[0], p[1]))
        steiner_pts    = ([Point(p[0], p[1]) for p in self.points]
                         if n_steiner else [])
        all_interior_pts = inner_ring_pts + steiner_pts
        n_input = len(all_interior_pts)
        print(f"[7/7] Triangulating {n_input} points …")

        self.triangulation = Bowyer_watson(all_interior_pts)
        n_raw = len(self.triangulation.triangles)

        self.filter_triangles(inner_rings, self.outer_idx)
        n_final = len(self.triangulation.triangles)
        elapsed = time.perf_counter() - t
        print(f"      Bowyer-Watson done     {elapsed:6.3f}s  "
              f"({n_raw} raw → {n_final} kept after filter)")

        print("-"*50)
        print(f"  Total meshing time: {time.perf_counter()-t_total:.3f}s")
        print(f"  Generated {n_final} interior cells  +  {len(self.boundary_elements)} boundary cells")
        print(f"  Grand total: {n_final + len(self.boundary_elements)} cells")
        print("="*50 + "\n")

    def create_boundary_points(self, ordered_lines):
        """Sample boundary points along a single ordered loop.

        Returns (points, thickness_mask, bc_tags) arrays for this loop.  The
        caller concatenates the per-loop results into the instance attributes
        consumed by solver_data_pipeline().
        """

        all_points = []

        all_thicknesses = []

        all_bc_tags = []


        bc_map = {"Wall": 0, "Velocity Inlet": 1, "Pressure Outlet": 2, "Inlet": 1, "Outlet": 2, "Symmetry": 3}

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


            # Use per-BC spacing if available, otherwise fall back to global
            spacing = self.bc_spacing_map.get(line.boundary_type, self.boundary_spacing)
            n_points = max(1, int(np.floor(line_length / spacing)))

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


        return (np.vstack(all_points),
                np.vstack(all_thicknesses),
                np.concatenate(all_bc_tags))

    def _get_local_r(self, base_r, candidate):
        """Return the effective Steiner spacing at a given candidate point.
    
        Only called during the seed-placement phase (a handful of times).
        The hot Poisson-disk loop uses radius_grid instead.
        """
        p_shapely = ShapelyPoint(candidate)

        min_signed_dist = None
        best_factor     = None
        best_buffer_mult = 5.0

        # Containment tier: if the point sits inside one or more zones, the
        # smallest-area one wins outright — this is what makes a small nested
        # zone dominate its own footprint instead of losing to a larger
        # enclosing zone (see the matching logic in _precompute_domain_grids).
        containing = []
        for zone in self.refinement_zones:
            if len(zone) == 3:
                poly, factor, buffer_mult = zone
            else:
                poly, factor = zone
                buffer_mult  = 5.0
            if poly.contains(p_shapely):
                containing.append((poly, max(factor, 1.1), buffer_mult))

        if containing:
            poly, best_factor, best_buffer_mult = min(containing, key=lambda z: z[0].area)
            min_signed_dist = -poly.exterior.distance(p_shapely)
        else:
            # Transition tier: no zone contains the point — fall back to the
            # nearest zone by signed distance, exactly as before.
            for zone in self.refinement_zones:
                if len(zone) == 3:
                    poly, factor, buffer_mult = zone
                else:
                    poly, factor = zone
                    buffer_mult  = 5.0
                safe_factor = max(factor, 1.1)

                signed_dist = poly.distance(p_shapely)

                if min_signed_dist is None or signed_dist < min_signed_dist:
                    min_signed_dist  = signed_dist
                    best_factor      = safe_factor
                    best_buffer_mult = buffer_mult

        if best_factor is None or min_signed_dist is None:
            return base_r
    
        local_refined = base_r / best_factor
        buffer_width  = best_buffer_mult * local_refined
        t             = (min_signed_dist + buffer_width) / (2.0 * buffer_width)
        t             = max(0.0, min(1.0, t))
        smooth_t      = t * t * (3.0 - 2.0 * t)
    
        return local_refined * (1.0 - smooth_t) + base_r * smooth_t
    
    def create_steiner_points(self, inner_rings, r=4.0, k=30):
        """Sample interior Steiner points for a domain that may contain holes.
    
        `inner_rings` is a list of Nx2 arrays, one per loop, where
        `inner_rings[self.outer_idx]` is the outer boundary and the rest are
        holes.  A Shapely polygon-with-holes is built so the Poisson-disk
        rejection test automatically avoids the hole interiors.
    
        If refinement_zones are defined, the Steiner spacing inside each zone
        is r / factor.  A single Poisson-disk pass fills all zones + background
        simultaneously, with spatially-varying spacing.  Each zone is guaranteed
        a seed at its centroid so overlapping or disconnected zone arms are all
        populated.
    
        Performance: `inside_grid` and `radius_grid` are precomputed once over
        the background grid before the Poisson-disk loop begins, replacing the
        per-candidate Shapely calls that dominated runtime at large point counts.
        """
        if not inner_rings or len(inner_rings[self.outer_idx]) < 3:
            raise ValueError("Boundary polygon not defined properly.")
    
        outer_coords = [tuple(p) for p in inner_rings[self.outer_idx]]
        hole_coords  = [[tuple(p) for p in ring]
                        for i, ring in enumerate(inner_rings)
                        if i != self.outer_idx and len(ring) >= 3]
        full_poly = ShapelyPoly(outer_coords, hole_coords)
        if not full_poly.is_valid:
            print("      Warning: domain boundary is self-intersecting after "
                  "boundary-layer extrusion — a hole's boundary layers likely "
                  "overlap the outer wall's (shapes drawn too close together, "
                  "or n_layers/thickness too large for the gap). The interior "
                  "mesh may be split or distorted near that overlap.")

        # Determine the effective minimum r across all refinement zones so the
        # background grid is fine enough for the densest zone.
        min_r = r
        for zone in self.refinement_zones:
            factor  = zone[1]
            local_r = r / max(factor, 1.1)
            if local_r < min_r:
                min_r = local_r
    
        safe_zone        = full_poly.buffer(-min_r * 0.8)
        xmin, ymin, xmax, ymax = full_poly.bounds
    
        w    = min_r / np.sqrt(2)
        cols = int(np.ceil((xmax - xmin) / w))
        rows = int(np.ceil((ymax - ymin) / w))
    
        # Clamp to prevent absurdly large grids from tiny refinement factors
        MAX_GRID = 2000
        if cols > MAX_GRID or rows > MAX_GRID:
            scale = min(MAX_GRID / cols, MAX_GRID / rows)
            cols  = max(1, int(cols * scale))
            rows  = max(1, int(rows * scale))
            w     = (xmax - xmin) / cols if cols > 1 else w
    
        _w_inv = 1.0 / w
    
        def get_grid_coords(p):
            gx = int((p[0] - xmin) * _w_inv)
            gy = int((p[1] - ymin) * _w_inv)
            gx = 0 if gx < 0 else (cols - 1 if gx >= cols else gx)
            gy = 0 if gy < 0 else (rows - 1 if gy >= rows else gy)
            return gx, gy
    
        # ------------------------------------------------------------------
        # Precompute domain grids ONCE — replaces per-candidate Shapely calls
        # ------------------------------------------------------------------
        inside_grid, radius_grid = self._precompute_domain_grids(
            safe_zone, xmin, ymin, w, cols, rows)
    
        # Poisson-disk occupancy grid (None = empty cell)
        grid       = np.full((cols, rows), None, dtype=object)
        all_points = []
        active     = []
    
        # ------------------------------------------------------------------
        # Seed points
        # ------------------------------------------------------------------
    
        # 1. Background: one random point in the safe zone.
        #    Uses inside_grid for the fast accept check.
        found_bg  = False
        attempts  = 0
        while not found_bg and attempts < 1000:
            attempts += 1
            p0       = np.random.uniform([xmin, ymin], [xmax, ymax])
            gx, gy   = get_grid_coords(p0)
            if inside_grid[gx, gy]:
                all_points.append(p0)
                active.append(p0)
                grid[gx, gy] = p0
                found_bg      = True
    
        # 2. Each refinement zone: one seed at its centroid.
        #    _get_local_r is acceptable here — called only once per zone.
        for zone in self.refinement_zones:
            poly     = zone[0]
            centroid = np.array([poly.centroid.x, poly.centroid.y])
            seed     = centroid
    
            # If the centroid is outside the safe zone, try random fallback
            c_gx, c_gy = get_grid_coords(seed)
            if not inside_grid[c_gx, c_gy]:
                found_zone = False
                for _ in range(100):
                    p_rnd    = np.random.uniform([xmin, ymin], [xmax, ymax])
                    r_gx, r_gy = get_grid_coords(p_rnd)
                    if poly.contains(ShapelyPoint(p_rnd)) and inside_grid[r_gx, r_gy]:
                        seed       = p_rnd
                        found_zone = True
                        break
                if not found_zone:
                    print(f"  WARNING: refinement zone at centroid "
                          f"({centroid[0]:.2f}, {centroid[1]:.2f}) got no seed "
                          f"point after 100 attempts — it will receive NO "
                          f"targeted refinement.")
                    continue
    
            gx, gy   = get_grid_coords(seed)
            local_r  = self._get_local_r(r, seed)   # fine here — only a few calls
            r_sq     = local_r * local_r
            is_far   = True
            i0, i1   = max(gx - 2, 0), min(gx + 3, cols)
            j0, j1   = max(gy - 2, 0), min(gy + 3, rows)
            for i in range(i0, i1):
                for j in range(j0, j1):
                    nb = grid[i, j]
                    if nb is not None:
                        dx, dy = seed[0] - nb[0], seed[1] - nb[1]
                        if dx*dx + dy*dy < r_sq:
                            is_far = False
                            break
                if not is_far:
                    break
    
            if is_far:
                all_points.append(seed)
                active.append(seed)
                grid[gx, gy] = seed
    
        # ------------------------------------------------------------------
        # Main Poisson-disk pass — spatially-varying radius
        #
        # Hot-path changes vs original:
        #   • safe_zone.contains(ShapelyPoint(candidate))  →  inside_grid[gx, gy]
        #   • self._get_local_r(r, candidate)              →  radius_grid[gx, gy]
        # Both are now O(1) numpy array lookups.
        # ------------------------------------------------------------------
        while active:
            idx        = np.random.randint(len(active))
            base_point = active[idx]
            found      = False
    
            for _ in range(k):
                angle     = np.random.uniform(0, 2 * np.pi)
                # Use local spacing at base_point for correct step distance
                base_gx, base_gy = get_grid_coords(base_point)
                local_r_base = radius_grid[base_gx, base_gy]
                rad        = np.random.uniform(local_r_base, 2 * local_r_base)
                candidate  = base_point + rad * np.array([np.cos(angle), np.sin(angle)])
    
                # Fast bounding-box reject
                if not (xmin <= candidate[0] <= xmax and ymin <= candidate[1] <= ymax):
                    continue
    
                gx, gy = get_grid_coords(candidate)
    
                # --- O(1) domain membership check (replaces Shapely contains) ---
                if not inside_grid[gx, gy]:
                    continue
    
                # --- O(1) local radius lookup (replaces _get_local_r) ---
                local_r_cand = radius_grid[gx, gy]
                local_r_sq   = local_r_cand * local_r_cand
    
                # Neighbour distance check in the occupancy grid
                is_far_enough = True
                i0 = gx - 2 if gx > 2 else 0
                i1 = gx + 3 if gx + 3 < cols else cols
                j0 = gy - 2 if gy > 2 else 0
                j1 = gy + 3 if gy + 3 < rows else rows
                for i in range(i0, i1):
                    for j in range(j0, j1):
                        nb = grid[i, j]
                        if nb is not None:
                            dx, dy = candidate[0] - nb[0], candidate[1] - nb[1]
                            if dx*dx + dy*dy < local_r_sq:
                                is_far_enough = False
                                break
                    if not is_far_enough:
                        break
    
                if not is_far_enough:
                    continue
    
                # Accept the candidate
                all_points.append(candidate)
                active.append(candidate)
                grid[gx, gy] = candidate
                found         = True
                break
    
            if not found:
                active.pop(idx)
    
        self.points = np.array(all_points)
        
    def build_polygon(self):
        """Group the (possibly disconnected) CAD lines into separate closed
        loops.  Each loop is a list of Line objects ordered head-to-tail.
        The loop with the largest absolute area is treated as the outer
        boundary; all others are holes (handled later via orientation).
        """
        remaining = self.lines.copy()
        loops = []
        while remaining:
            first = remaining.pop(0)
            ordered = [first]
            pivot = first.b
            closed = False
            while True:
                # Loop closed when we return to the start vertex
                if pivot == ordered[0].a:
                    closed = True
                    break
                found = False
                for i, line in enumerate(remaining):
                    if pivot == line.a:
                        pivot = line.b
                        ordered.append(line)
                        remaining.pop(i)
                        found = True
                        break
                    elif pivot == line.b:
                        pivot = line.a
                        ordered.append(line)
                        remaining.pop(i)
                        found = True
                        break
                if not found:
                    break
            if not closed:
                raise ValueError("Lines do not form a closed loop "
                                 f"({len(ordered)} edges, not closed)")
            loops.append(ordered)
        return loops

    def polygon_orientation(self, polygon_array):
        x = polygon_array[:, 0]
        y = polygon_array[:, 1]
        area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        return area  # area < 0 is CCW, area > 0 is CW

    def get_render_data(self):
        """Categorizes mesh elements into bundles for multi-colored wireframe rendering."""
        # Bundle 1: Unstructured Interior (Triangles)
        tri_coords = []
        if hasattr(self, 'triangulation') and self.triangulation:
            for t in self.triangulation.triangles:
                # Store as 3 line segments (6 points) for wireframe
                pts = [t.a, t.b, t.b, t.c, t.c, t.a]
                for pt in pts:
                    tri_coords.extend([pt.x, pt.y])

        # Bundle 2: Boundary Layers (mostly Quads, but connect_layers also
        # emits Triangles where a boundary layer pinches at a sharp corner —
        # see boundary_elements' documented type). Use the polymorphic
        # .vertices() (both Quad and Triangle implement it) rather than the
        # Quad-only .points, and build the wireframe from however many
        # vertices come back. Byte-identical output for Quads.
        quad_coords = []
        if hasattr(self, 'boundary_elements'):
            for q in self.boundary_elements:
                p = q.vertices()
                n = len(p)
                for i in range(n):
                    a, b = p[i], p[(i + 1) % n]
                    quad_coords.extend([a.x, a.y, b.x, b.y])

        # Bundle 3: The original CAD wall lines
        wall_coords = []
        for line in self.lines:
            wall_coords.extend([line.a.x, line.a.y, line.b.x, line.b.y])

        return {
            'triangles': (np.array(tri_coords, dtype=np.float32), len(tri_coords) // 2),
            'quads': (np.array(quad_coords, dtype=np.float32), len(quad_coords) // 2),
            'walls': (np.array(wall_coords, dtype=np.float32), len(wall_coords) // 2)
        }

    @staticmethod
    def upload_wireframe_bundles(old_vbos, bundles):
        """Delete `old_vbos`' GL buffers and upload `bundles`
        (key -> (float32 coords array, point count)) as new VboHandles.

        Shared by the live-mesh path (rebuild_wireframe_vbos, below) and
        the loaded-.npz path in app_state's _apply_loaded_mesh_settings,
        which builds its own bundles dict from stored cell data since it
        has no live Mesher/triangulation to call get_render_data() on.
        """
        for handle in old_vbos.values():
            handle.delete()
        new_vbos = {}
        for key, (data, count) in bundles.items():
            if count > 0:
                handle = VboHandle(components=2)
                handle.upload(data)
                new_vbos[key] = handle
        return new_vbos

    def rebuild_wireframe_vbos(self, old_vbos):
        """Delete `old_vbos` and build fresh wireframe VBOs from this mesh."""
        return Mesher.upload_wireframe_bundles(old_vbos, self.get_render_data())

    def boundary_layer(self, polygon_points, current_thickness_array,
                       extrude_toward_interior=True):
        """
        polygon_points: Nx2 array of the current layer's points
        current_thickness_array: Nx1 array containing the target thickness for each point
        """
        n = len(polygon_points)
        new_points = np.zeros_like(polygon_points)

        # 1. Calculate standard edge vectors and normals
        next_points = np.roll(polygon_points, -1, axis=0)
        edges = next_points - polygon_points
        edge_lengths = np.linalg.norm(edges, axis=1)[:, np.newaxis]
        unit_edges = edges / np.where(edge_lengths > 1e-9, edge_lengths, 1.0)

        # Determine edge normals based on polygon orientation
        if self.orientation > 0:
            edge_normals = np.column_stack([-unit_edges[:, 1], unit_edges[:, 0]])
        else:
            edge_normals = np.column_stack([unit_edges[:, 1], -unit_edges[:, 0]])

        # 2. Compute standard vertex miter normals
        prev_edge_normals = np.roll(edge_normals, 1, axis=0)
        vertex_normals = prev_edge_normals + edge_normals
        v_norm = np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]
        vertex_normals = np.where(v_norm > 1e-9, vertex_normals / v_norm, edge_normals)

        cos_theta = np.sum(vertex_normals * edge_normals, axis=1)[:, np.newaxis]
        miter_lengths = 1.0 / np.maximum(cos_theta, 0.1)

        # 3. --- CONSTRAINED CORNER OVERRIDE ---
        # Look at the point BC tags to find where Wall (0) meets Inlet/Outlet (1 or 2)
        for i in range(n):
            current_tag = self.point_bc_mask[i]
            prev_tag = self.point_bc_mask[(i - 1) % n]
            
            # Case A: Moving from an Inlet/Outlet to a Wall
            if prev_tag != 0 and current_tag == 0:
                # Force the extrusion vector to point exactly along the previous edge (the Inlet),
                # picking whichever of the two tangent directions actually agrees with the wall's
                # own (already correctly-signed, see step 4) interior normal. This used to just
                # assume -unit_edges[(i-1)%n] points into the domain, which only holds at an
                # ordinary convex wall/inlet corner; at a concave one (e.g. an injector's side
                # branch meeting the main channel) that assumption flips and the inlet's own
                # boundary layer came out extruded outward. The old abs() on the projection below
                # masked exactly this sign error instead of catching it.
                raw_tangent = -unit_edges[(i - 1) % n]
                wall_normal = edge_normals[i]
                inlet_vector = raw_tangent if np.dot(raw_tangent, wall_normal) >= 0 else -raw_tangent
                vertex_normals[i] = inlet_vector

                # Adjust length so the normal thickness relative to the wall remains correct
                # For a 90-degree corner, dot product is 1.0, so length multiplier is 1.0
                projection = np.dot(inlet_vector, wall_normal)  # >= 0 by construction above
                miter_lengths[i] = 1.0 / max(projection, 0.1)

            # Case B: Moving from a Wall to an Inlet/Outlet
            elif prev_tag == 0 and current_tag != 0:
                # Force the extrusion vector to point exactly along the current edge (the Inlet);
                # see Case A above for why the sign is derived from wall_normal rather than assumed.
                raw_tangent = unit_edges[i]
                wall_normal = edge_normals[(i - 1) % n]
                inlet_vector = raw_tangent if np.dot(raw_tangent, wall_normal) >= 0 else -raw_tangent
                vertex_normals[i] = inlet_vector

                projection = np.dot(inlet_vector, wall_normal)  # >= 0 by construction above
                miter_lengths[i] = 1.0 / max(projection, 0.1)
                
                # Ensure this point actually moves (it belongs to the inlet patch now, 
                # but it must stretch to accommodate the neighboring wall's thickness)
                current_thickness_array[i] = current_thickness_array[(i - 1) % n]

        # 4. Generate the new layer points.
        # vertex_normals is built purely from this loop's own edge_normals
        # (step 1, signed by self.orientation, which is measured fresh per
        # loop) plus the inlet/outlet corner overrides above — so it already
        # points into *this loop's own interior* everywhere, including at
        # reflex/concave corners: averaging two same-sense edge normals can
        # never flip past either one, it just needs a longer miter (clamped
        # by cos_theta above) at sharp corners. That's true regardless of
        # winding direction or convexity.
        #
        # An outer loop's "own interior" is the fluid domain, so its
        # displacement is used as-is. A hole's "own interior" is the solid
        # obstacle, so the fluid is on the *other* side — a hole's
        # displacement is negated wholesale via extrude_toward_interior.
        #
        # This used to be a per-vertex check against the loop centroid
        # (flip displacement[i] if it pointed away from `centroid -
        # polygon_points[i]`), meant to make the outer/hole flip robust to
        # however the user drew the loop. But "distance to the single
        # centroid point" isn't a valid interior/exterior test on a concave
        # outline — near a reflex corner the straight line to the centroid
        # can leave the domain and re-enter it, so the check would fire on
        # an already-correct displacement and flip it outward (e.g. an
        # ink-injector shape with a concave junction had two walls come out
        # with outward-facing prism layers). Since the per-vertex normal is
        # already correctly signed on its own, no geometric re-check is
        # needed — just apply the static outer/hole sign.
        displacement = vertex_normals * (miter_lengths * current_thickness_array)
        if not extrude_toward_interior:
            displacement = -displacement
        new_points = polygon_points + displacement
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

    def filter_triangles(self, inner_rings, outer_idx):
        """Keep only triangles whose centroid lies inside the domain, i.e.
        inside the outer loop's inner ring but outside every hole's inner
        ring, and make sure they don't cross any boundaries.
        """
        if not inner_rings or len(inner_rings[outer_idx]) < 3:
            return

        # Build fluid domain polygon using Shapely
        outer_coords = [tuple(p) for p in inner_rings[outer_idx]]
        hole_coords = [[tuple(p) for p in ring]
                      for i, ring in enumerate(inner_rings)
                      if i != outer_idx and len(ring) >= 3]
        domain_poly = ShapelyPoly(outer_coords, hole_coords)

        # Snapshot the list before any removal (swap-with-last changes order)
        current_tris = list(self.triangulation.triangles)

        to_remove = []
        for t in current_tris:
            t_poly = ShapelyPoly([(t.a.x, t.a.y), (t.b.x, t.b.y), (t.c.x, t.c.y)])
            # Centroid check
            centroid = ShapelyPoint(t.centroid.x, t.centroid.y)
            if not domain_poly.contains(centroid):
                to_remove.append(t)
                continue

            # Robust overlap check (avoid crossing boundaries)
            try:
                overlap_area = t_poly.intersection(domain_poly).area
                if overlap_area < 0.98 * t_poly.area:
                    to_remove.append(t)
            except Exception as exc:
                # Shapely intersection failed on a topology edge case; the
                # centroid check above already passed, so keep the triangle
                # rather than guess — but say so.
                print(f"  WARNING: overlap check failed for triangle at "
                      f"({t.centroid.x:.2f}, {t.centroid.y:.2f}) ({exc}); "
                      f"keeping it on the centroid-only check.")

        for t in to_remove:
            self.triangulation.remove_triangle(t)

    def solver_data_pipeline(self):
        """
        Builds the mesh data dict for the Solver.
        ALL coordinates and distances are converted to SI metres by multiplying
        with self.unit_to_meters.  Areas are multiplied by unit_to_meters².
        """
        t_pipe = time.perf_counter()
        print("\n" + "="*50)
        print("  DATA PIPELINE  —  mesh → solver handoff")
        print("="*50)
        s  = self.unit_to_meters
        s2 = s * s

        def get_edge_key(p_a, p_b):
            ax = p_a.x if hasattr(p_a, 'x') else p_a[0]
            ay = p_a.y if hasattr(p_a, 'y') else p_a[1]
            bx = p_b.x if hasattr(p_b, 'x') else p_b[0]
            by = p_b.y if hasattr(p_b, 'y') else p_b[1]
            k1 = (round(float(ax), 6), round(float(ay), 6))
            k2 = (round(float(bx), 6), round(float(by), 6))
            return tuple(sorted([k1, k2]))

        # 1. BC lookup — built PER LOOP so the wrap-around does not connect
        #    the last point of one loop to the first of the next (which would
        #    create a spurious boundary edge between the outer loop and a hole).
        bc_lookup = {}
        offset = 0
        for cnt in self.loop_point_counts:
            seg = self.boundary_points[offset:offset + cnt]
            for i in range(cnt):
                p1 = seg[i]
                p2 = seg[(i + 1) % cnt]
                edge_key = get_edge_key(p1, p2)
                bc_lookup[edge_key] = self.point_bc_mask[offset + i]
            offset += cnt

        # Per-loop boundary midpoints (concatenated, aligned with point_bc_mask)
        b_mids_list = []
        offset = 0
        for cnt in self.loop_point_counts:
            seg = self.boundary_points[offset:offset + cnt]
            b_mids_list.append((seg + np.roll(seg, -1, axis=0)) / 2.0)
            offset += cnt
        b_mids = np.vstack(b_mids_list)   # precomputed once

        # 2. Gather all cells
        t = time.perf_counter()
        valid_boundary_elements = [c for c in self.boundary_elements if float(c.area) > 1e-8]
        Cells = valid_boundary_elements + self.triangulation.triangles
        Nc = len(Cells)
        print(f"[1/4] Cell gather           {time.perf_counter()-t:6.3f}s  ({Nc} cells: "
            f"{len(valid_boundary_elements)} boundary + {len(self.triangulation.triangles)} interior)")

        # Cell centers and areas (world units → converted to SI below)
        cell_centers_wu = np.array([[c.centroid.x, c.centroid.y] for c in Cells], dtype=np.float64)
        cell_areas_wu   = np.array([float(c.area) for c in Cells], dtype=np.float64)

        # 3. Build Edge Map
        t = time.perf_counter()
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
        Nf = len(edge_map)
        n_internal = sum(1 for ids in edge_map.values() if len(ids) > 1)
        n_boundary = Nf - n_internal
        print(f"[2/4] Edge map              {time.perf_counter()-t:6.3f}s  ({Nf} faces: "
            f"{n_internal} internal, {n_boundary} boundary)")

        # 4. Populate Face Arrays (still in world units at this point)
        t = time.perf_counter()
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

                diffs    = b_mids - face_mid
                sq_dist  = diffs[:, 0]**2 + diffs[:, 1]**2
                min_idx  = int(np.argmin(sq_dist))
                min_dist = np.sqrt(sq_dist[min_idx])
                assigned_tag = self.point_bc_mask[min_idx]

                # Scale-aware tolerance: boundary points are spaced at
                # `boundary_spacing` world units, so a true boundary face
                # midpoint is always within ~spacing/2 of a boundary midpoint.
                # (Previously a hard-coded 1.0 world unit, which broke at m scale.)
                tol = self.boundary_spacing
                if min_dist < tol:
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

        print(f"[3/4] Face arrays           {time.perf_counter()-t:6.3f}s  (BC tags: "
            f"wall={np.sum(boundary_tags==0)}, "
            f"inlet={np.sum(boundary_tags==1)}, "
            f"outlet={np.sum(boundary_tags==2)}, "
            f"internal={np.sum(boundary_tags==-1)})"
            f"symmetry={np.sum(boundary_tags==3)})")

        # ----------------------------------------------------------------
        # 5. UNIT CONVERSION  — world units → SI metres
        #    Lengths × s,  Areas × s²,  Normals (Sf) are edge-length vectors × s
        # ----------------------------------------------------------------
        t = time.perf_counter()
        cell_centers_si = cell_centers_wu * s
        cell_areas_si   = cell_areas_wu   * s2
        Sf_si           = Sf_wu   * s
        Cf_si           = Cf_wu   * s
        df_si           = df_wu   * s
        magDf_si        = magDf_wu * s
        magSf_si        = np.linalg.norm(Sf_si, axis=1)
        print(f"[4/4] Unit conversion (×{s}) {time.perf_counter()-t:6.3f}s")
        print(f"      cell_areas:  [{cell_areas_si.min():.3e}, {cell_areas_si.max():.3e}] m²")
        print(f"      cell_centers:[{cell_centers_si.min():.4f},  {cell_centers_si.max():.4f}] m")

        cells_in_faces = set(owner) | set(neighbor[neighbor != -1])
        all_cells      = set(range(Nc))
        orphan_cells   = all_cells - cells_in_faces
        if orphan_cells:
            print(f"⚠️  MESHER BUG: {len(orphan_cells)} cells have NO faces!")
        else:
            print(f"      Connectivity OK — no orphan cells")

        print("-"*50)
        print(f"  Pipeline total: {time.perf_counter()-t_pipe:.3f}s  →  solver ready")
        print("="*50 + "\n")

        # ----------------------------------------------------------------
        # 6. CELL VERTEX GEOMETRY  — for the Visualizer (and mesh reload)
        #    The solver ignores these; they ride along so a saved mesh can
        #    be re-displayed without the original Mesher object.
        # ----------------------------------------------------------------
        max_verts = 4  # quads are the largest cell type
        cell_vertices = np.zeros((Nc, max_verts, 2), dtype=np.float64)
        cell_nverts   = np.zeros(Nc, dtype=np.int32)
        cell_types    = np.zeros(Nc, dtype=np.int32)   # 0=triangle, 1=quad
        for ci, cell in enumerate(Cells):
            pts = cell.vertices()
            cell_nverts[ci] = len(pts)
            cell_types[ci]  = 1 if len(pts) == 4 else 0
            for vi, p in enumerate(pts):
                cell_vertices[ci, vi, 0] = p.x
                cell_vertices[ci, vi, 1] = p.y
        cell_vertices *= s   # world units → SI metres

        # ----------------------------------------------------------------
        # 7. CAD LINES  — so a loaded mesh can be remeshed and its boundary
        #    conditions edited.  Each row: [ax, ay, bx, by, bc_type_idx]
        #    bc_type_idx: 0=Wall, 1=Velocity Inlet, 2=Pressure Outlet
        # ----------------------------------------------------------------
        bc_map_rev = {"Wall": 0, "Velocity Inlet": 1, "Pressure Outlet": 2, "Symmetry": 3}
        cad_lines = np.array([
            [line.a.x, line.a.y, line.b.x, line.b.y,
             bc_map_rev.get(line.boundary_type, 0)]
            for line in self.lines
        ], dtype=np.float64)

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
            'cell_vertices': cell_vertices,
            'cell_nverts':   cell_nverts,
            'cell_types':    cell_types,
            'cad_lines':     cad_lines,
            # --- Meshing parameters (ride along for save/load; solver ignores) ---
            # `thickness` is the *original* first-layer value the user entered
            # (mesh() mutates self.thickness via the growth loop, so we saved
            # the captured original in self._orig_thickness).
            'n_layers':          self.n_layers,
            'growth_factor':     self.growth_factor,
            'thickness':         getattr(self, '_orig_thickness', self.thickness),
            'boundary_spacing':  self.boundary_spacing,
            'bc_spacing_map':    self.bc_spacing_map,
            'r':                 self.r,
            'unit_to_meters':    self.unit_to_meters,
            # --- Refinement zones (ride along for save/load; solver ignores) ---
            # Each zone is (shapely_polygon, factor). Serialize the polygon's
            # exterior ring as an Nx2 array so it survives np.savez_compressed.
            'refinement_zones': np.array([
                (np.array(poly.exterior.coords, dtype=np.float64),
                 float(factor),
                 float(buffer_mult))
                for poly, factor, buffer_mult in self.refinement_zones
            ], dtype=object),
        }

    def _precompute_domain_grids(self, safe_zone, xmin, ymin, w, cols, rows):
        """Precompute inside/radius lookup grids at the Poisson-disk grid resolution.
    
        Called once before the hot loop.  Pays the Shapely / per-point cost over
        the fixed (cols × rows) grid cells instead of over every candidate point.
    
        Parameters
        ----------
        safe_zone : shapely Polygon (already eroded by buffer(-min_r * 0.8))
        xmin, ymin : float  — domain bounding-box origin
        w : float           — grid cell side length (= min_r / sqrt(2))
        cols, rows : int    — grid dimensions
    
        Returns
        -------
        inside_grid : bool ndarray (cols, rows)
            True where a candidate at that grid cell should be accepted (i.e. the
            cell centre is inside safe_zone).
        radius_grid : float64 ndarray (cols, rows)
            Local Steiner spacing at each grid cell (= _get_local_r output).
        """
        # Build (cols*rows, 2) array of all grid-cell centres
        gxs = np.arange(cols)
        gys = np.arange(rows)
        GX, GY = np.meshgrid(gxs, gys, indexing='ij')   # both (cols, rows)
        cx = xmin + (GX + 0.5) * w
        cy = ymin + (GY + 0.5) * w
        centers = np.column_stack([cx.ravel(), cy.ravel()])  # (N, 2)
    
        # ------------------------------------------------------------------
        # inside_grid — vectorised point-in-polygon via matplotlib Path
        # matplotlib.path.Path.contains_points is a C extension; it handles
        # one polygon ring at a time, so we check the exterior then subtract holes.
        #
        # Eroding a polygon-with-holes (safe_zone = full_poly.buffer(-min_r*0.8))
        # can split the safe interior into disconnected pieces whenever it
        # pinches to near-zero width somewhere — e.g. a hole drawn close to the
        # outer wall, so their boundary-layer rings almost touch.  When that
        # happens Shapely hands back a MultiPolygon instead of a Polygon, so we
        # iterate over every piece rather than assuming a single exterior ring.
        # ------------------------------------------------------------------
        polys = list(safe_zone.geoms) if safe_zone.geom_type == 'MultiPolygon' else [safe_zone]
        if len(polys) > 1:
            print(f"      Note: safe interior zone split into {len(polys)} pieces "
                  f"(shapes may be close together) — meshing all of them")

        inside_flat = np.zeros(len(centers), dtype=bool)
        for poly in polys:
            if poly.is_empty:
                continue
            ext_path = Path(np.array(poly.exterior.coords))
            piece_inside = ext_path.contains_points(centers)
            for interior in poly.interiors:
                hole_path      = Path(np.array(interior.coords))
                piece_inside  &= ~hole_path.contains_points(centers)
            inside_flat |= piece_inside

        inside_grid = inside_flat.reshape(cols, rows)
    
        # ------------------------------------------------------------------
        # radius_grid — vectorised local spacing (mirrors _get_local_r exactly)
        # ------------------------------------------------------------------
        r = self.r  # background spacing
    
        if not self.refinement_zones:
            radius_grid = np.full((cols, rows), r, dtype=np.float64)
            return inside_grid, radius_grid
    
        # Two-tier zone priority so nested zones behave as expected (a small,
        # more-refined zone dominates its own footprint even near its center,
        # instead of losing to a larger enclosing zone that happens to have a
        # more negative raw distance-to-boundary there):
        #   1. Containment tier — if a cell is inside one or more zones, the
        #      SMALLEST-area containing zone wins, regardless of distance.
        #   2. Transition tier — cells inside no zone fall back to the nearest
        #      zone by signed distance (unchanged from before), so the existing
        #      blend-to-background behavior near zone edges is preserved.
        n_cells          = len(centers)
        best_signed_dist = np.full(n_cells, np.inf, dtype=np.float64)
        best_factor      = np.ones(n_cells,          dtype=np.float64)
        best_buf_mult    = np.full(n_cells, 5.0,     dtype=np.float64)
        claimed          = np.zeros(n_cells, dtype=bool)

        zones_by_area = sorted(self.refinement_zones, key=lambda z: z[0].area)
        zone_cache = []  # (is_inside, signed_dists, safe_factor, buffer_mult), smallest zone first

        # Pass 1 — containment tier, smallest zone first so it claims cells
        # before any larger enclosing zone gets a chance to.
        for zone in zones_by_area:
            poly        = zone[0]
            factor      = zone[1]
            buffer_mult = zone[2] if len(zone) == 3 else 5.0
            safe_factor = max(factor, 1.1)

            # Vectorised inside check for this zone via matplotlib Path
            zone_path = Path(np.array(poly.exterior.coords))
            is_inside = zone_path.contains_points(centers)   # bool (N,)

            # Signed distance: negative inside, positive outside.
            # We approximate boundary distance as distance to the exterior ring,
            # which is exact for convex zones and a good approximation otherwise.
            # Shapely is called per-point here, but only ONCE per zone over the
            # fixed grid — not once per Poisson-disk candidate.
            signed_dists = np.empty(n_cells, dtype=np.float64)
            ext_ring = poly.exterior  # cache the attribute lookup
            for i, (px, py) in enumerate(centers):
                pt = ShapelyPoint(px, py)
                d  = ext_ring.distance(pt)
                signed_dists[i] = -d if is_inside[i] else d

            zone_cache.append((is_inside, signed_dists, safe_factor, buffer_mult))

            newly = is_inside & ~claimed
            best_signed_dist[newly] = signed_dists[newly]
            best_factor[newly]      = safe_factor
            best_buf_mult[newly]    = buffer_mult
            claimed |= is_inside

        # Pass 2 — transition tier, only for cells no zone contains. Nearest
        # zone by signed distance wins, reusing the arrays cached in pass 1.
        unclaimed = ~claimed
        if unclaimed.any():
            for is_inside, signed_dists, safe_factor, buffer_mult in zone_cache:
                better = unclaimed & (signed_dists < best_signed_dist)
                best_signed_dist[better] = signed_dists[better]
                best_factor[better]      = safe_factor
                best_buf_mult[better]    = buffer_mult

        # Smoothstep blending — identical formula to _get_local_r
        local_refined = r / best_factor                         # spacing inside zone
        buffer_width  = best_buf_mult * local_refined           # transition band width
        t             = (best_signed_dist + buffer_width) / (2.0 * buffer_width)
        t             = np.clip(t, 0.0, 1.0)
        smooth_t      = t * t * (3.0 - 2.0 * t)               # C1 smoothstep
        radius_flat   = local_refined * (1.0 - smooth_t) + r * smooth_t
    
        radius_grid = radius_flat.reshape(cols, rows)
        return inside_grid, radius_grid