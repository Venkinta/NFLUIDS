import numpy as np


class Triangulation:
    """
    Triangulation with O(1) add/remove and a zero-copy coords view.

    Key ideas
    ---------
    * coords is a pre-allocated (cap, 6) float64 numpy array.
      The public `coords` property returns a *view* of the live rows —
      no list conversion, no np.asarray copy, zero allocation.

    * remove_triangle uses a dict  id(triangle) → index  for O(1) lookup,
      then swaps the vacated slot with the last row so the array stays
      compact without shifting anything.

    * add_triangle doubles capacity when full (amortised O(1)).

    These two changes eliminate the two largest profiler hits in the
    previous version: the 23.9 s np.asarray rebuild and the 7.8 s
    list.index scan.
    """

    _INITIAL_CAP = 2048   # triangles; doubles on overflow

    def __init__(self, triangles=None):
        self.triangles: list = []          # Triangle objects; order == _coords rows
        self._tri_to_idx: dict = {}        # id(triangle) -> row index in _coords

        self._cap  = self._INITIAL_CAP
        self._size = 0
        # Pre-allocated backing store.  Row i = (ax, ay, bx, by, cx, cy)
        self._coords_np = np.empty((self._cap, 6), dtype=np.float64)

        if triangles is not None:
            for t in triangles:
                self.add_triangle(t)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def coords(self):
        """Return a (N, 6) float64 *view* — zero copy, always up to date."""
        return self._coords_np[:self._size]

    def add_triangle(self, t):
        """Amortised O(1) insert."""
        if self._size == self._cap:
            self._cap *= 2
            new_arr = np.empty((self._cap, 6), dtype=np.float64)
            new_arr[:self._size] = self._coords_np[:self._size]
            self._coords_np = new_arr

        i = self._size
        # Write directly into the pre-allocated row — no tuple creation
        self._coords_np[i, 0] = t.a.x
        self._coords_np[i, 1] = t.a.y
        self._coords_np[i, 2] = t.b.x
        self._coords_np[i, 3] = t.b.y
        self._coords_np[i, 4] = t.c.x
        self._coords_np[i, 5] = t.c.y
        self.triangles.append(t)
        self._tri_to_idx[id(t)] = i
        self._size += 1

    def remove_triangle(self, t):
        """O(1) removal: dict lookup + swap-with-last."""
        idx = self._tri_to_idx.pop(id(t), None)
        if idx is None:
            return                              # already gone — safe no-op

        last = self._size - 1
        if idx != last:
            # Move the last triangle into the vacated slot
            last_tri = self.triangles[last]
            self.triangles[idx]          = last_tri
            self._coords_np[idx]         = self._coords_np[last]
            self._tri_to_idx[id(last_tri)] = idx

        self.triangles.pop()                    # O(1) — removes from end
        self._size -= 1

    def draw(self, screen, camera, color=(0, 0, 255), width=2):
        for triangle in self.triangles:
            triangle.draw(screen, camera, color, width)