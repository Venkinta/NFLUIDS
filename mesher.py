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
    def __init__(self,screen,lines,n_layers,growth_factor,thickness,spacing,r,RENDERER):
        self.lines = lines
        self.points = None
        self.boundary_points = None
        self.candidate_points = None
        self.triangulation = None
        self.orientation = None
        self.thickness_mask = None
        
        
        #boundary_layers
        self.n_layers = n_layers 
        self.growth_factor = growth_factor
        self.thickness = thickness
        self.boundary_spacing = spacing
        
        #mesh generation
        self.r = r #20 is good
        
        self.renderer = RENDERER
        self.finished = False
        
        
    def mesh(self):
        # 1. Get the lines in the correct loop order
        ordered_lines = self.build_polygon() 
        
        # 2. Calculate orientation (using the helper we wrote)
        # We pass a numpy array for speed
        vert_coords = [(line.a.x, line.a.y) for line in ordered_lines]
        vert_array = np.array(vert_coords)
        self.orientation = self.polygon_orientation(vert_array)

        # 3. Create high-res boundary points based on the ORDERED vertices
        # Now we have the ordered points and an array with their thicknesses
        self.create_boundary_points(ordered_lines) 
        
        # 4. Generate Layers
        layers = [self.boundary_points]

        
        for i in range(self.n_layers):
            current_factor_array = self.thickness_mask * self.thickness
            next_layer = self.boundary_layer(layers[-1], current_factor_array)
            layers.append(next_layer)
            self.thickness *= self.growth_factor # Geometric stretching
            

        # 5. Connect them into Quads
        self.boundary_elements = self.connect_layers(layers)

        # --- PHASE 2: Unstructured Interior ---
        # The last ring of the boundary layer is the "new" wall for the triangles
        inner_ring = layers[-1]
        
        # Generate Steiner points ONLY inside this inner ring
        self.create_steiner_points(inner_ring, self.r) 
        
        # Convert inner_ring and Steiner points to Point objects for Bowyer-Watson
        inner_ring_pts = [Point(p[0], p[1]) for p in inner_ring]
        steiner_pts = [Point(p[0], p[1]) for p in self.points]
        
        all_interior_pts = inner_ring_pts + steiner_pts
        
        # 4. Triangulate the core
        self.triangulation = Bowyer_watson(all_interior_pts)

        # 5. NEW: Filter out triangles that cross outside the inner ring
        self.filter_triangles(inner_ring)

        message = 'generated ' + repr(len(self.triangulation.triangles)) + ' cells'
        print(message)



        
        
    def create_boundary_points(self, ordered_lines):
        all_points = []
        all_thicknesses = []
        all_bc_tags = []  # <--- NEW: To store our 0, 1, 2 tags
        
        # Mapping strings to the integers our solver expects
        bc_map = {"Wall": 0, "Velocity Inlet": 1, "Pressure Outlet": 2, "Inlet":1,"Outlet":2}
        
        num_l = len(ordered_lines)

        for i in range(num_l):
            line = ordered_lines[i]
            next_line = ordered_lines[(i + 1) % num_l]
            prev_line = ordered_lines[(i - 1) % num_l]
            
            start = np.array([line.a.x, line.a.y])
            end = np.array([next_line.a.x, next_line.a.y])

            line_vec = end - start
            line_length = np.linalg.norm(line_vec)
            if line_length == 0: continue

            n_points = max(1, int(np.floor(line_length / self.boundary_spacing)))
            segment_points = np.linspace(start, end, n_points, endpoint=False)
            all_points.append(segment_points)

            # --- 1. THICKNESS LOGIC (For Geometry) ---
            if line.boundary_type == "Wall":
                seg_thick = np.ones((n_points, 1))
            else:
                seg_thick = np.zeros((n_points, 1))
            
            if line.boundary_type != "Wall" and prev_line.boundary_type == "Wall":
                seg_thick[0] = 1.0
            all_thicknesses.append(seg_thick)

            # --- 2. BC TAG LOGIC (For Physics) --- <--- NEW
            # Get the integer ID (default to 0/Wall if something goes wrong)
            tag = bc_map.get(line.boundary_type, 0)
            # Every point/segment created from this line gets this tag
            seg_tags = np.full(n_points, tag)
            all_bc_tags.append(seg_tags)
        
        self.boundary_points = np.vstack(all_points)
        self.thickness_mask = np.vstack(all_thicknesses)
        # Flatten the list of tags into one long array
        self.point_bc_mask = np.concatenate(all_bc_tags)
        
        
        
        
    def check_points(self):
        #checks if generated points lie inside the boundary
        polygon_path = Path(self.boundary_points)  # your Nx2 array of vertices
        mask = polygon_path.contains_points(self.points)
        steiner_points = self.points[mask]  # only points truly inside polygon
      
      
      
        
    def create_steiner_points(self, boundary_points, r=550, k=30):
        
        
        if boundary_points is None or len(boundary_points) < 3:
            raise ValueError("Boundary polygon not defined properly.")


        
        full_poly = ShapelyPoly(boundary_points)
        safe_zone = full_poly.buffer(-r * 0.8)
        xmin, ymin, xmax, ymax = full_poly.bounds
        

        
        
        
        # --- SAFETY CHECK ---
        # If the grid is going to be massive, stop before we crash
        w = r / np.sqrt(2)
        cols = int(np.ceil((xmax - xmin) / w))
        rows = int(np.ceil((ymax - ymin) / w))
        
        
        if cols * rows > 500000: # Half a million cells limit
            
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



        # --- Step 1: Initial Point (with Safety) ---
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

        # --- Step 2: Expansion ---
        while active:
            
            idx = np.random.randint(len(active))
            base_point = active[idx]
            found = False

            for _ in range(k):
                
                angle = np.random.uniform(0, 2 * np.pi)
                rad = np.random.uniform(r, 2 * r)
                candidate = base_point + rad * np.array([np.cos(angle), np.sin(angle)])

                # 1. Fast Bounding Box Check
                if not (xmin <= candidate[0] <= xmax and ymin <= candidate[1] <= ymax):
                    
                    continue
                
                # 2. Fast Grid Check (The "Secret Sauce")
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

                # 3. Slow Polygon Check (ONLY do this if it passed everything else)
                
                if not safe_zone.contains(ShapelyPoint(candidate)):
                    continue

                # If we got here, the point is valid!
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

                if np.array_equal(pivot,line.a):
                    pivot = line.b
                    ordered_lines.append(line)
                    remaining.pop(i)
                    found = True
                    break

                elif np.array_equal(pivot,line.b):
                    
                    pivot = line.a
                    ordered_lines.append(line)
                    remaining.pop(i)
                    found = True
                    break

            if not found:
                raise ValueError("Lines do not form a closed loop")


        return ordered_lines
  
    def polygon_orientation(self, polygon_array):
        """
        polygon_array: Nx2 numpy array
        """
        x = polygon_array[:, 0]
        y = polygon_array[:, 1]
        
        # Shoelace formula using numpy roll for the (i+1) terms
        # sum of (x_i * y_{i+1} - x_{i+1} * y_i)
        area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        return area # area < 0 is CCW, area > 0 is CW


    def draw(self, screen,camera):
        imgui.new_frame()
        
        # --- Draw boundary lines ---
        if hasattr(self, "lines"):
            for line in self.lines:
                line.draw(screen,camera, color=(255, 255, 255), width=2)
                
        # Draw Boundary Layer Quads
        if hasattr(self, 'boundary_elements'):
            for quad in self.boundary_elements:
                quad.draw(screen,camera)

        # --- Draw triangulation ---
        if hasattr(self, "triangulation") and self.triangulation:
            self.triangulation.draw(screen,camera)
            
            
        imgui.begin("Solver")
        if imgui.button("Proceed to Solving"):
            self.finish()
        imgui.end()
        
      
        imgui.render()
        self.renderer.render(imgui.get_draw_data())
            
            
                    
    def create_boundary_layers(self, n_layers=1, scaling_factor=4):
        layers = [self.boundary_points]  # start with original
        for _ in range(n_layers):
            last_layer = layers[-1]
            new_layer = self.boundary_layer(last_layer, scaling_factor=scaling_factor)
            layers.append(new_layer)
        return layers

                        
                        
    def boundary_layer(self, polygon_points, scaling_factor):
        """
        polygon_points: np.array of shape (N, 2)
        Returns: np.array of shape (N, 2) representing the offset ring
        """
        n = len(polygon_points)
        new_points = np.zeros_like(polygon_points)

        # 1. Get edge vectors and their unit normals
        # We use roll to get the "next" point for each edge
        next_points = np.roll(polygon_points, -1, axis=0)
        edges = next_points - polygon_points
        edge_lengths = np.linalg.norm(edges, axis=1)[:, np.newaxis]
        unit_edges = edges / edge_lengths

        # Normals (perpendicular to edges)
        # If CCW: normal is (-dy, dx)
        if self.orientation > 0:
            edge_normals = np.column_stack([-unit_edges[:, 1], unit_edges[:, 0]])
        else:
            edge_normals = np.column_stack([unit_edges[:, 1], -unit_edges[:, 0]])

        # 2. Calculate vertex normals (Miter vectors)
        # The vertex normal is the average of the normals of the two meeting edges
        prev_edge_normals = np.roll(edge_normals, 1, axis=0)
        vertex_normals = prev_edge_normals + edge_normals
        
        # Normalize the vertex normal
        v_norm = np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]
        # Handle collinear edges to avoid division by zero
        vertex_normals = np.where(v_norm > 1e-9, vertex_normals / v_norm, edge_normals)

        # 3. Calculate Miter Length
        # The length needs to be adjusted so the perpendicular distance remains 'scaling_factor'
        # Length = scaling_factor / cos(theta), where 2*theta is the angle between edges
        # cos(theta) is the dot product of vertex_normal and edge_normal
        cos_theta = np.sum(vertex_normals * edge_normals, axis=1)[:, np.newaxis]
        miter_length = scaling_factor / np.maximum(cos_theta, 0.1) # Clamp to avoid blow-up at sharp spikes

        new_points = polygon_points + vertex_normals * miter_length
        
        return new_points

    def connect_layers(self, layers):
        """
        Connects concentric rings of points into Quad elements.
        Gracefully converts 'pinched' quads at boundary transitions into Triangles.
        """
        import math
        elements = []
        
        # Iterate through each gap between layers
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
                
                # Area check for a quad (Shoelace)
                area = 0.5 * abs((p1.x*p2.y + p2.x*p3.y + p3.x*p4.y + p4.x*p1.y) - 
                                (p2.x*p1.y + p3.x*p2.y + p4.x*p3.y + p1.x*p4.y))
                                
                # If the cell has virtually zero area entirely, skip it
                if area < 1e-4:
                    continue
                
                # Measure the length of all 4 edges
                tol = 1e-4
                d12 = math.hypot(p1.x - p2.x, p1.y - p2.y)
                d23 = math.hypot(p2.x - p3.x, p2.y - p3.y)
                d34 = math.hypot(p3.x - p4.x, p3.y - p4.y)
                d41 = math.hypot(p4.x - p1.x, p4.y - p1.y)
                
                # --- POLYGON CONVERSION LOGIC ---
                # If an edge is pinched (length near 0), spawn a Triangle instead!
                if d12 < tol:
                    elements.append(Triangle(p2, p3, p4))
                elif d23 < tol:
                    elements.append(Triangle(p1, p3, p4))
                elif d34 < tol:
                    elements.append(Triangle(p1, p2, p3))
                elif d41 < tol:
                    elements.append(Triangle(p1, p2, p3))
                else:
                    # All edges have length, spawn a healthy Quad
                    elements.append(Quad(p1, p2, p3, p4))
                    
        return elements

  
    def _to_point(self, p):
        # Helper to handle the numpy vs Point confusion
        if isinstance(p, Point): return p
        return Point(p[0], p[1])
    
    def filter_triangles(self, inner_ring_points):
        """Removes triangles whose centroids fall outside the inner boundary ring."""
        # 1. Create a Path object from the inner ring
        ring_path = Path(inner_ring_points)
        
        # 2. Extract the centroids as a raw NumPy array of [x, y]
        # This converts [Point(x,y), Point(x,y)...] -> [[x, y], [x, y]...]
        centroids_coords = [[t.centroid.x, t.centroid.y] for t in self.triangulation.triangles]
        centroids = np.array(centroids_coords, dtype=np.float64)
        
        # 3. Check which centroids are INSIDE the polygon ring
        # Now Matplotlib receives a NumPy array and will be happy!
        mask = ring_path.contains_points(centroids, radius=-1e-5)
        
        # 4. Rebuild the triangle list keeping only the valid ones
        valid_triangles = []
        for is_inside, triangle in zip(mask, self.triangulation.triangles):
            if is_inside:
                valid_triangles.append(triangle)
                
        # 5. Update the triangulation object
        self.triangulation.triangles = valid_triangles


    def solver_data_pipeline(self):
        print('Beginning data collection for Solver...')

        # --- NEW HELPER FUNCTION ---
        def get_edge_key(p_a, p_b):
            # Handles both objects (like Shapely points with .x) and arrays/lists
            ax = p_a.x if hasattr(p_a, 'x') else p_a[0]
            ay = p_a.y if hasattr(p_a, 'y') else p_a[1]
            bx = p_b.x if hasattr(p_b, 'x') else p_b[0]
            by = p_b.y if hasattr(p_b, 'y') else p_b[1]
            
            # Round to 6 decimals to eliminate float jitter
            k1 = (round(float(ax), 6), round(float(ay), 6))
            k2 = (round(float(bx), 6), round(float(by), 6))
            return tuple(sorted([k1, k2]))
        # ---------------------------
        # 1. Pre-calculate a lookup for boundary points -> BC Type
        # Key: (p1_coords, p2_coords), Value: BC_Type (0, 1, 2)
        # We use sorted tuples so the order of points in the edge doesn't matter
        # 1. Pre-calculate a lookup for boundary points
        bc_lookup = {}
        for i in range(len(self.boundary_points)):
            p1 = self.boundary_points[i]
            p2 = self.boundary_points[(i + 1) % len(self.boundary_points)]
            
            # Use the helper!
            edge_key = get_edge_key(p1, p2)
            bc_lookup[edge_key] = self.point_bc_mask[i]

        # --- 2. Gather all cells ---
        # Filter out boundary layer quads that have collapsed to zero/near-zero area
        valid_boundary_elements = [
            c for c in self.boundary_elements 
            if float(c.area) > 1e-8
        ]

        # Combine the healthy quads with the interior triangles
        Cells = valid_boundary_elements + self.triangulation.triangles
        Nc = len(Cells)
        
        # Extracting Centers: Ensure we get [x, y] for every cell
        cell_centers = np.array([[c.centroid.x, c.centroid.y] for c in Cells], dtype=np.float64)

        # Extracting Areas: Force everything to a float
        cell_areas = np.array([float(c.area) for c in Cells], dtype=np.float64)
                
        # 3. Build Edge Map
        # 3. Build Edge Map
        edge_map = {} 
        for cell_id, cell in enumerate(Cells):
            for edge in cell.edges(): 
                if len(edge) != 2:
                    continue
                    
                p_a, p_b = edge 
                # Use the helper!
                key = get_edge_key(p_a, p_b)
                
                if key not in edge_map:
                    edge_map[key] = []
                edge_map[key].append(cell_id)

        # 4. Populate Face Arrays
        Nf = len(edge_map)
        owner = np.zeros(Nf, dtype=np.int32)
        neighbor = np.full(Nf, -1, dtype=np.int32)
        Sf = np.zeros((Nf, 2))
        Cf = np.zeros((Nf, 2))
        df = np.zeros((Nf, 2))
        magDf = np.zeros(Nf)
        boundary_tags = np.full(Nf, -1) # -1: Internal, 0: Wall, 1: Inlet, 2: Outlet

        # --- DEBUG START ---
        print(f"DEBUG: bc_lookup contains {len(bc_lookup)} total boundary edges.")
        unique_tags_in_lookup = set(bc_lookup.values())
        print(f"DEBUG: Unique tags found in lookup: {unique_tags_in_lookup}")
        
        # Let's count how many edges in edge_map only have 1 cell (meaning they are boundaries)
        boundary_edge_count = sum(1 for ids in edge_map.values() if len(ids) == 1)
        print(f"DEBUG: edge_map has {boundary_edge_count} boundary candidates.")
        # --- DEBUG END ---


        for face_idx, (edge_key, cell_ids) in enumerate(edge_map.items()):
            owner[face_idx] = cell_ids[0]
            
            if len(cell_ids) > 1:
                neighbor[face_idx] = cell_ids[1]
                boundary_tags[face_idx] = -1 # Internal
            else:
                # B. Boundary Tagging - Robust Search
                p1_raw, p2_raw = edge_key
                face_mid = (np.array(p1_raw) + np.array(p2_raw)) / 2.0
                
                assigned_tag = 0 # Default to Wall
                min_dist = float('inf')
                
                # Check against original boundary segments
                for i in range(len(self.boundary_points)):
                    b1 = self.boundary_points[i]
                    b2 = self.boundary_points[(i + 1) % len(self.boundary_points)]
                    b_mid = (b1 + b2) / 2.0
                    
                    dist = np.linalg.norm(face_mid - b_mid)
                    if dist < min_dist:
                        min_dist = dist
                        assigned_tag = self.point_bc_mask[i]
                
                # If the face is reasonably close to a boundary segment, tag it.
                # If it's far away (like an internal hole), it stays a Wall (0).
                if min_dist < 1.0: # 1.0 is a safe tolerance for most mesh sizes
                    boundary_tags[face_idx] = assigned_tag
                else:
                    boundary_tags[face_idx] = 0

            # C. Geometry (Vector math)
            p1_coords, p2_coords = edge_key
            p1 = np.array(p1_coords)
            p2 = np.array(p2_coords)
            
            face_center = (p1 + p2) / 2.0
            vec = p2 - p1
            normal = np.array([vec[1], -vec[0]]) # Normal vector
            
            # Ensure normal points AWAY from owner
            owner_c = cell_centers[owner[face_idx]]
            if np.dot(normal, face_center - owner_c) < 0:
                normal = -normal
            
            Sf[face_idx] = normal
            Cf[face_idx] = face_center

            # D. Distance vector (for diffusion terms)
            if neighbor[face_idx] != -1:
                df_vec = cell_centers[neighbor[face_idx]] - owner_c
            else:
                df_vec = face_center - owner_c # For BCs, d is center-to-face
            
            df[face_idx] = df_vec
            magDf[face_idx] = np.linalg.norm(df_vec)

        print(cell_centers)
        cells_in_faces = set(owner) | set(neighbor[neighbor != -1])
        all_cells = set(range(Nc))
        orphan_cells = all_cells - cells_in_faces

        if orphan_cells:
            print(f"⚠️  MESHER BUG: {len(orphan_cells)} cells have NO faces!")
            print(f"    These cells will cause singular matrix!")
        return {
            'Nc': Nc, #number of cells
            'Nf': Nf, #number of faces
            'owner': owner, #each face has an owner ID
            'neighbor': neighbor, #each face has a neighbour ID
            'Sf': Sf, #face normal vector
            'magSf': np.linalg.norm(Sf, axis=1), # face normal vector magnitude
            'Cf': Cf, #center of face 
            'df': df, #vector from owner to neighbour
            'magDf': magDf, #magnitude of vector (distance)
            'cell_centers': cell_centers, #center of each cell
            'cell_areas': cell_areas, #area of each cell
            'boundary_tags': boundary_tags # Tag of each face type -1: Internal, 0: Wall, 1: Inlet, 2: Outlet
        }
                
        
    def finish(self):
        self.finished = True
        pass