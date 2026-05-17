import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, other):
        if not isinstance(other, Point): return False
        # Using math.isclose is safer for floats, but this works for now
        return self.x == other.x and self.y == other.y

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