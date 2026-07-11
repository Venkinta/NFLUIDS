from .point import Point

class SnapEngine:
    def __init__(self, pixel_radius=10):
        self.pixel_radius = pixel_radius

    def get_snapped_pos(self, current_world_pos, lines, camera_scale, anchor_pos=None):
        """Returns (snapped_point, is_vertex_snap)
        is_vertex_snap is True only when a vertex was snapped to (exact match).
        """
        wx, wy = current_world_pos.x, current_world_pos.y
        world_radius = self.pixel_radius / camera_scale
        world_sq_radius = world_radius ** 2
        
        # 1. PRIORITY: Vertex Snapping (Exact corners)
        all_points = []
        for line in lines:
            all_points.extend([line.a, line.b])
        
        for pt in all_points:
            dist_sq = (wx - pt.x)**2 + (wy - pt.y)**2
            if dist_sq <= world_sq_radius:
                return pt, True  # (snapped_vertex, is_vertex_snap=True)

        # 2. SECONDARY: Global Alignment & Axis Locking (not a perfect snap)
        snap_x, snap_y = None, None
        
        # Check anchor (for perfect vertical/horizontal lines)
        if anchor_pos:
            if abs(wx - anchor_pos.x) < world_radius: snap_x = anchor_pos.x
            if abs(wy - anchor_pos.y) < world_radius: snap_y = anchor_pos.y

        # Check all other points (for alignment/tracking)
        for pt in all_points:
            if snap_x is None and abs(wx - pt.x) < world_radius:
                snap_x = pt.x
            if snap_y is None and abs(wy - pt.y) < world_radius:
                snap_y = pt.y

        final_x = snap_x if snap_x is not None else wx
        final_y = snap_y if snap_y is not None else wy
        
        return Point(final_x, final_y), False  # (aligned_point, is_vertex_snap=False)