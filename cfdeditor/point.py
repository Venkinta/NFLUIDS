import math
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, other):
        if not isinstance(other, Point): return False
        # Tolerance-based comparison. NOTE: __hash__ is intentionally left as
        # the exact coordinate hash so Bowyer-Watson dedup (set() on Points)
        # still relies on bit-identical coordinates. The tolerance here only
        # affects explicit `==` comparisons (e.g. build_polygon), and is tight
        # enough (1e-9) that it will not collapse distinct mesh vertices.
        return (math.isclose(self.x, other.x, rel_tol=0.0, abs_tol=1e-9) and
                math.isclose(self.y, other.y, rel_tol=0.0, abs_tol=1e-9))

    def __hash__(self):
        # Points are effectively immutable after creation in this codebase.
        # Cache the hash so repeated frozenset/dict lookups (edge_count in
        # Bowyer-Watson, edge_map in solver_data_pipeline) pay no tuple cost.
        try:
            return self._hash
        except AttributeError:
            self._hash = hash((self.x, self.y))
            return self._hash

    def __repr__(self):
        return f"P({self.x:.3f}, {self.y:.3f})"

    # --- Math Operations that return Points ---
    def __add__(self, other):
        dx = other.x if hasattr(other, 'x') else other[0]
        dy = other.y if hasattr(other, 'y') else other[1]
        return Point(self.x + dx, self.y + dy)

    def __sub__(self, other):
        dx = other.x if hasattr(other, 'x') else other[0]
        dy = other.y if hasattr(other, 'y') else other[1]
        return Point(self.x - dx, self.y - dy)

    # --- Utility Methods ---
    def distance_to(self, other):
        """Used by the SnapEngine for threshold checks."""
        dx = self.x - (other.x if hasattr(other, 'x') else other[0])
        dy = self.y - (other.y if hasattr(other, 'y') else other[1])
        return np.sqrt(dx**2 + dy**2)

    def to_tuple(self):
        """Quick conversion for Pygame calls."""
        return (self.x, self.y)