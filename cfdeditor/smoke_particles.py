import math
import numpy as np
from OpenGL.GL import *

# solver_data_pipeline()'s boundary_tags convention (mesher.py bc_map).
_BC_VELOCITY_INLET = 1


class SmokeParticles:
    """Tracer particles advected through a Visualizer's frozen velocity
    snapshot and continuously reseeded, so they read as flowing smoke.

    Holds a back-reference to the owning Visualizer (not copies of its
    centroids/U/tree) so field updates (mid-solve or final) are picked up
    live with no extra wiring.
    """

    def __init__(self, owner, count=800, speed_scale=1.0, point_size=3.0,
                 despawn_multiplier=2.5, limit_lifetime=False, lifetime=6.0,
                 color=(1.0, 1.0, 1.0)):
        self.owner = owner
        self.count = count
        self.speed_scale = speed_scale
        self.point_size = point_size
        self.despawn_multiplier = despawn_multiplier
        self.limit_lifetime = limit_lifetime
        self.lifetime = lifetime
        self.color = color

        self.positions = np.empty((0, 2), dtype=np.float32)
        self.ages = np.empty(0, dtype=np.float32)
        self.max_ages = np.empty(0, dtype=np.float32)

        self._rng = np.random.default_rng()
        self._bbox_min, self._bbox_max = self._compute_domain_bbox()
        self._char_length = self._compute_char_length()
        self._despawn_dist = self._char_length * self.despawn_multiplier
        self._inlet_points = self._compute_inlet_points()

        self._grow(self.count)

        self.particle_vbo = glGenBuffers(1)
        self.vertex_count = 0
        self._update_particle_vbo()

    def _compute_domain_bbox(self):
        xs, ys = [], []
        for cell in self.owner.cell_data_map:
            pts = [(p.x, p.y) for p in cell.vertices()] if hasattr(cell, "vertices") else cell
            for x, y in pts:
                xs.append(x)
                ys.append(y)
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        return (np.array([xs.min(), ys.min()], dtype=np.float32),
                np.array([xs.max(), ys.max()], dtype=np.float32))

    def _compute_char_length(self):
        bbox_area = float(np.prod(self._bbox_max - self._bbox_min))
        nc = len(self.owner.centroids)
        return math.sqrt(bbox_area / max(nc, 1))

    def _compute_inlet_points(self):
        """Velocity-inlet face midpoints, in the Visualizer's display units.

        `mesh_data` (from solver_data_pipeline()) stores boundary_tags/Cf in
        SI metres regardless of the CAD drawing's own units, so this divides
        back out by unit_to_meters to match self.owner.centroids' scale.
        Returns an (M, 2) array, empty if no mesh_data or no tagged inlet
        faces are available (e.g. an older save or a mesh with no BCs yet) —
        callers fall back to random-in-domain seeding in that case.
        """
        mesh_data = getattr(self.owner, "mesh_data", None)
        if not mesh_data:
            return np.empty((0, 2), dtype=np.float32)
        tags = mesh_data.get("boundary_tags")
        cf = mesh_data.get("Cf")
        u2m = mesh_data.get("unit_to_meters")
        if tags is None or cf is None or not u2m:
            return np.empty((0, 2), dtype=np.float32)
        mask = tags == _BC_VELOCITY_INLET
        if not np.any(mask):
            return np.empty((0, 2), dtype=np.float32)
        return (cf[mask] / u2m).astype(np.float32)

    def _seed_at_inlet(self, n):
        """Random point among inlet face midpoints, jittered along the
        boundary by a fraction of the local cell size so respawns don't all
        land on the same handful of discrete points."""
        idx = self._rng.integers(0, len(self._inlet_points), size=n)
        pts = self._inlet_points[idx].copy()
        jitter = self._rng.uniform(-0.5, 0.5, size=(n, 2)).astype(np.float32) * self._char_length
        return pts + jitter

    def _seed_random_bbox(self, n):
        """Rejection-sample n valid in-mesh points within the domain bbox.

        Fallback used when no inlet faces are tagged. A capped retry loop
        plus a guaranteed-valid centroid fallback keeps this bounded even on
        sparse/oddly-shaped domains where random bbox points rarely land
        inside the mesh.
        """
        out = np.empty((n, 2), dtype=np.float32)
        filled = 0
        max_rounds = 30
        accept_dist = self._char_length * 1.5

        for _ in range(max_rounds):
            need = n - filled
            candidates = self._rng.uniform(self._bbox_min, self._bbox_max, size=(need, 2)).astype(np.float32)
            dist, idx = self.owner.tree.query(candidates, k=1)
            ok = dist < accept_dist
            for i in np.nonzero(ok)[0]:
                cell = self.owner.cell_data_map[idx[i]]
                if not self.owner._is_point_in_cell(candidates[i, 0], candidates[i, 1], cell):
                    ok[i] = False
            accepted = candidates[ok]
            take = min(len(accepted), need)
            out[filled:filled + take] = accepted[:take]
            filled += take
            if filled >= n:
                break

        if filled < n:
            fallback_idx = self._rng.integers(0, len(self.owner.centroids), size=n - filled)
            out[filled:] = self.owner.centroids[fallback_idx]

        return out

    def _seed_positions(self, n):
        if len(self._inlet_points) > 0:
            return self._seed_at_inlet(n)
        return self._seed_random_bbox(n)

    def _sample_max_ages(self, n):
        # ±20% spread around the user-set lifetime so particles don't all
        # expire on the same frame even when lifetime limiting is enabled.
        return self._rng.uniform(self.lifetime * 0.8, self.lifetime * 1.2, size=n).astype(np.float32)

    def _grow(self, n):
        """Append n freshly-seeded particles (used at construction and when
        the particle-count slider is increased)."""
        new_pos = self._seed_positions(n)
        new_max_age = self._sample_max_ages(n)
        # Stagger initial ages so a freshly grown batch doesn't all read as
        # "just spawned" on the next frame.
        new_age = self._rng.uniform(0.0, new_max_age).astype(np.float32)

        self.positions = np.concatenate([self.positions, new_pos])
        self.max_ages = np.concatenate([self.max_ages, new_max_age])
        self.ages = np.concatenate([self.ages, new_age])

    def set_count(self, n):
        n = max(0, int(n))
        current = len(self.positions)
        if n == current:
            return
        elif n > current:
            self._grow(n - current)
        else:
            self.positions = self.positions[:n]
            self.ages = self.ages[:n]
            self.max_ages = self.max_ages[:n]
        self.count = n
        self._update_particle_vbo()

    def step(self, dt):
        owner = self.owner

        dist, idx = owner.tree.query(self.positions, k=3)
        eps = 1e-6
        w = 1.0 / (dist + eps)
        w /= w.sum(axis=1, keepdims=True)
        sampled_U = np.einsum('nk,nkd->nd', w, owner.U[idx])

        self.positions += sampled_U * (dt * self.speed_scale)
        self.ages += dt

        respawn_mask = dist[:, 0] > self._despawn_dist
        if self.limit_lifetime:
            respawn_mask |= self.ages >= self.max_ages

        n_respawn = int(respawn_mask.sum())
        if n_respawn > 0:
            self.positions[respawn_mask] = self._seed_positions(n_respawn)
            self.max_ages[respawn_mask] = self._sample_max_ages(n_respawn)
            self.ages[respawn_mask] = 0.0

        self._update_particle_vbo()

    def _update_particle_vbo(self):
        self.vertex_count = len(self.positions)
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.positions.nbytes, self.positions, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, camera):
        camera.apply_gl_transform()
        glEnableClientState(GL_VERTEX_ARRAY)
        glPointSize(self.point_size)
        glColor3f(*self.color)
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        glVertexPointer(2, GL_FLOAT, 0, None)
        glDrawArrays(GL_POINTS, 0, self.vertex_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        camera.remove_gl_transform()

    def destroy(self):
        glDeleteBuffers(1, [self.particle_vbo])
