import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab, LinearOperator, spilu

try:
    import pyamg
    _HAS_PYAMG = True
except ImportError:
    _HAS_PYAMG = False

# ---------------------------------------------------------------------------
# Solver  —  SIMPLE algorithm for incompressible 2-D Navier-Stokes
#
# Key optimisations over the original version
# -------------------------------------------
# 1. spsolve (SuperLU direct, O(N^1.5)) replaced by bicgstab + cached ILU
#    preconditioner (O(N * k)).  For 200k cells this changes the solver from
#    hours to minutes.  Loose tolerance (1e-3) is intentional: SIMPLE's outer
#    pressure-velocity loop is what drives overall convergence; tight inner
#    solves waste time without improving stability.
# 2. np.add.at replaced by np.bincount everywhere.  add.at disables NumPy's
#    fast paths; bincount runs in C with no GIL overhead.
# 3. All topology / index arrays precomputed once in __init__ and reused.
#    Eliminates repeated np.concatenate and list.append inside the hot loop.
# 4. COO row/col arrays for both matrix assemblies are precomputed once.
#    Only the *data* values are recomputed each iteration.
# 5. Last computed grad_P cached so health_check avoids an extra full sweep.
# ---------------------------------------------------------------------------


class Solver:
    """
        Solver receives handout from mesher/physics selector with dictionary of geometric information:
            return {
            'Nc':           int (Scalar)
                            Total number of cells in mesh.
                            
            'Nf':           int (Scalar)
                            Total number of unique faces (edges) in mesh.
                            
            'owner':        numpy.ndarray (shape: (Nf,), dtype: int32)
                            ID of the cell that "owns" each face. Ranges from 0 to Nc-1.
                            The mesher runs a loop which assigns the cell it encounters first (lower id)
                            as the owner, and the other as the neighbor.
                            
            'neighbor':     numpy.ndarray (shape: (Nf,), dtype: int32)
                            ID of the adjacent cell sharing each face. 
                            Set to -1 if the face is an internal face. 
                            (In practice there are no internal faces, only walls)
                            
            'Sf':           numpy.ndarray (shape: (Nf, 2), dtype: float64)
                            Face area vectors in SI units [m]. Normal to the face, 
                            scaled by face length, pointing outward from owner to neighbor.
                            
            'magSf':        numpy.ndarray (shape: (Nf,), dtype: float64)
                            Magnitude of Sf in SI units [m]. Represents the physical 
                            length of each 1D face/edge.
                            
            'Cf':           numpy.ndarray (shape: (Nf, 2), dtype: float64)
                            Coordinates (x, y) of the face midpoints in SI units [m].
                            
            'df':           numpy.ndarray (shape: (Nf, 2), dtype: float64)
                            Distance vectors in SI units [m]. 
                            - Internal faces: Vector from owner cell center to neighbor cell center.
                            - Boundary faces: Vector from owner cell center to face center (Cf).
                            Important to later correct fluxes to account for skewed cells.
                            
            'magDf':        numpy.ndarray (shape: (Nf,), dtype: float64)
                            Magnitude of df in SI units [m]. Scalar straight-line distance 
                            represented by df.
                            
            'cell_centers': numpy.ndarray (shape: (Nc, 2), dtype: float64)
                            Coordinates (x, y) of the geometric centroids for all cells 
                            in SI units [m].
                            
            'cell_areas':   numpy.ndarray (shape: (Nc,), dtype: float64)
                            Physical 2D surface areas of all cells in SI units [m²].
                            
            'boundary_tags': numpy.ndarray (shape: (Nf,), dtype: int64/int32)
                            Boundary identifiers for each face:
                            -1 = Internal face
                             0 = Wall boundary
                             1 = Inlet boundary
                             2 = Outlet boundary
        }
"""
    def __init__(self, mesher_data, inlet_velocity, outlet_pressure, rho, viscosity):
        # ---Solver parameters
        self.alpha_u = 0.3 #relaxation factor for velocity
        self.alpha_p = 0.1 #relaxation factor for pressure
                
        # ---Physical parameters---
        self.inlet_velocity  = np.asarray(inlet_velocity, dtype=np.float64)
        self.outlet_pressure = float(outlet_pressure)
        self.rho             = float(rho)
        self.viscosity       = float(viscosity) #Moving from dynamic viscosity to kinematic 

        # ---Mesher data: number of cells and faces
        mesh = mesher_data
        self.Nc = mesh['Nc']
        self.Nf = mesh['Nf']

        # ---Ownership ID's
        self.owner         = mesh['owner']
        self.neighbor      = mesh['neighbor']
        self.boundary_tags = mesh['boundary_tags']

        # ---Values of cells and faces 
        self.Sf    = mesh['Sf']
        self.magSf = mesh['magSf']
        self.Cf    = mesh['Cf']
        self.df    = mesh['df']
        self.magDf = mesh['magDf']

        self.magDf = np.maximum(self.magDf, 1e-10)
        self.magSf = np.maximum(self.magSf, 1e-10)

        self.cell_centers = mesh['cell_centers']
        self.cell_areas   = mesh['cell_areas']

        #id's
        self.wall_faces     = np.where(self.boundary_tags == 0)[0]
        self.inlet_faces    = np.where(self.boundary_tags == 1)[0]
        self.outlet_faces   = np.where(self.boundary_tags == 2)[0]
        self.internal_faces = np.where(self.boundary_tags == -1)[0]

        print("--- MESH SANITY CHECK ---")
        print(f"Total Cells:    {self.Nc}")
        print(f"Internal Faces: {len(self.internal_faces)}")
        print(f"Inlet Faces:    {len(self.inlet_faces)}")
        print(f"Outlet Faces:   {len(self.outlet_faces)}")
        print(f"Wall Faces:     {len(self.wall_faces)}")
        print("-------------------------")

        self._precompute_topology()

        # Iterative solver state
        self._precond_cache    = {}
        self._precond_interval = 50   # rebuild ILU every N SIMPLE iterations
        self._iteration        = 0
        self._last_grad_P      = None  # cached for health_check

    # ------------------------------------------------------------------
    # One-time topology precomputation
    # ------------------------------------------------------------------

    def _precompute_topology(self):
        """Cache every derived index/slice array used in the hot loop."""
        f_int = self.internal_faces

        # Gradient computation
        self._grad_own = self.owner[f_int]
        self._grad_nei = self.neighbor[f_int]
        self._Sf_int   = self.Sf[f_int]

        # Boundary faces (ordered: inlet, outlet, wall — consistent throughout)
        self._f_bnd  = np.concatenate([self.inlet_faces,
                                        self.outlet_faces,
                                        self.wall_faces])
        self._own_b  = self.owner[self._f_bnd] #grab owner id and Sf for each boundary face
        self._Sf_bnd = self.Sf[self._f_bnd]

        outlet_set = set(self.outlet_faces.tolist())
        self._is_outlet_bnd = np.array(
            [f in outlet_set for f in self._f_bnd], dtype=bool)

        # Per-face-type aliases, just separating everything for ease of use later, 
        # and to reduce repeated computation each iteration (mesh doesn't change...)
        self._own_i   = self._grad_own
        self._nei_i   = self._grad_nei
        self._own_in  = self.owner[self.inlet_faces]
        self._own_w   = self.owner[self.wall_faces]
        self._own_out = self.owner[self.outlet_faces]
        self._Sf_in   = self.Sf[self.inlet_faces]
        self._Sf_w    = self.Sf[self.wall_faces]
        self._Sf_out  = self.Sf[self.outlet_faces]
        self._all_owner = self.owner  # alias, no copy

        # COO index arrays (still needed to define structure)
        mom_rows = np.concatenate([
            self._own_i, self._own_i, self._nei_i, self._nei_i,
            self._own_in, self._own_w, self._own_out,
        ])
        mom_cols = np.concatenate([
            self._own_i, self._nei_i, self._nei_i, self._own_i,
            self._own_in, self._own_w, self._own_out,
        ])
        pcorr_rows = np.concatenate([
            self._own_i, self._own_i, self._nei_i, self._nei_i,
            self._own_in, self._own_out, self._own_w,
        ])
        pcorr_cols = np.concatenate([
            self._own_i, self._nei_i, self._nei_i, self._own_i,
            self._own_in, self._own_out, self._own_w,
        ])

        # Pre-build the CSR structures.  This eliminates coo_tocsr and
        # csr_sort_indices from the hot loop — both appear at ~2s each.
        self._mom_csr   = self._build_csr_template(mom_rows,   mom_cols)
        self._pcorr_csr = self._build_csr_template(pcorr_rows, pcorr_cols)

    # ------------------------------------------------------------------
    # CSR precomputation helpers
    # ------------------------------------------------------------------

    def _build_csr_template(self, rows, cols):
        """
        Precompute the CSR sparsity structure and scatter-add index for a
        matrix whose (row, col) pattern is fixed but whose values change
        every iteration.

        Returns a dict with:
          indptr     : CSR row pointer array (Nc+1,)
          indices    : CSR column index array (nnz_unique,)
          scatter    : for each COO entry k, scatter[k] = position in CSR data
                       where it should be accumulated (handles duplicates)
          n_unique   : number of unique (row, col) pairs = nnz of the matrix
        """
        Nc = self.Nc
        n  = len(rows)

        # Sort COO by (row, col) — this is the sorting that coo_tocsr does
        sort_order = np.lexsort((cols, rows))
        rows_s = rows[sort_order]
        cols_s = cols[sort_order]

        # Identify unique (row, col) pairs and the inverse mapping
        pairs = rows_s.astype(np.int64) * Nc + cols_s.astype(np.int64)
        _, first_occ, inv_idx = np.unique(pairs, return_index=True,
                                           return_inverse=True)
        n_unique   = len(first_occ)
        rows_u     = rows_s[first_occ]
        cols_u     = cols_s[first_occ]

        # Build CSR indptr from unique row assignments
        row_counts  = np.bincount(rows_u, minlength=Nc)
        indptr      = np.zeros(Nc + 1, dtype=np.int32)
        indptr[1:]  = np.cumsum(row_counts)

        # CSR column indices (sorted within each row by construction)
        indices = cols_u.astype(np.int32)

        # scatter[k] = CSR data position for original COO entry k
        # rank[k] = position of original entry k in the sorted COO
        rank          = np.empty(n, dtype=np.int64)
        rank[sort_order] = np.arange(n, dtype=np.int64)
        scatter       = inv_idx[rank]

        return dict(indptr=indptr, indices=indices,
                    scatter=scatter, n_unique=n_unique)

    def _make_csr(self, csr_info, coo_data):
        """
        Fast CSR matrix construction from precomputed structure.

        All sorting and deduplication is precomputed; this runs one bincount
        (fast C loop) instead of the full coo_tocsr + csr_sort_indices pipeline.
        """
        data = np.bincount(csr_info['scatter'], weights=coo_data,
                           minlength=csr_info['n_unique'])
        return csr_matrix((data, csr_info['indices'], csr_info['indptr']),
                          shape=(self.Nc, self.Nc), copy=False)

    # ------------------------------------------------------------------

    def _solve_momentum(self, A, b, cache_key):
        """
        Solves the momentum linear system using BiCGSTAB with a Jacobi preconditioner.
        
        Momentum matrices are strongly diagonally dominant because the implicit 
        under-relaxation step inflates the diagonal entries by 1/alpha. Assuming 
        a standard alpha = 0.2, the diagonal is boosted by 5x, allowing a simple 
        Jacobi preconditioner to achieve rapid convergence.
        
        Applying the Jacobi preconditioner is a trivial element-wise vector divide. 
        This eliminates the heavy triangular-solve overhead of SuperLU discovered 
        in previous profiler bottlenecks.
        
        If the tolerance (rtol=1e-3) fails to converge within 300 iterations, 
        the solver falls back to a relaxed tolerance (rtol=1e-2) to avoid halting 
        the global simulation loop.
        """
        diag = np.abs(A.diagonal())
        np.maximum(diag, 1e-30, out=diag)
        M  = LinearOperator(A.shape, matvec=lambda x: x / diag, dtype=np.float64)
        x0 = b / diag

        x, info = bicgstab(A, b, x0=x0, M=M, rtol=1e-3, atol=0.0, maxiter=300)
        if info != 0:
            x, info = bicgstab(A, b, x0=x0, M=M, rtol=1e-2, atol=0.0, maxiter=200)
            if info != 0:
                print(f"  BiCGSTAB [{cache_key}] stalled (info={info}), "
                      f"res={np.linalg.norm(A @ x - b):.2e}")
        return x

    def _solve_pressure(self, A, b):
        """
        BiCGSTAB + preconditioner for the pressure-correction equation.

        Primary: PyAMG (algebraic multigrid) if installed — optimal O(N) for
                 Laplacian-type problems.  `pip install pyamg` to enable.
        Fallback: ILU with fill_factor=8.  fill_factor=4 (previous default) was
                  too weak for larger meshes — it caused ~82 BiCGSTAB iterations
                  per pressure solve and ultimately stalled SIMPLE convergence.
                  fill_factor=8 brings that back to ~16 iterations.

        Preconditioner is rebuilt every _precond_interval SIMPLE iterations.
        """
        refresh = (
            self._iteration % self._precond_interval == 0 or
            'pressure' not in self._precond_cache
        )
        if refresh:
            if _HAS_PYAMG:
                try:
                    ml = pyamg.ruge_stuben_solver(A, coarse_solver='pinv')
                    self._precond_cache['pressure'] = ml.aspreconditioner()
                except Exception as exc:
                    print(f"  PyAMG setup failed ({exc}); falling back to ILU.")
                    self._precond_cache.pop('pressure', None)
            if not _HAS_PYAMG or 'pressure' not in self._precond_cache:
                try:
                    ilu = spilu(A.tocsc(), fill_factor=8, drop_tol=1e-4)
                    self._precond_cache['pressure'] = LinearOperator(
                        A.shape, matvec=ilu.solve, dtype=np.float64)
                except Exception as exc:
                    print(f"  ILU failed ({exc}); no preconditioner this step.")
                    self._precond_cache.pop('pressure', None)

        M  = self._precond_cache.get('pressure')
        x0 = b / (np.abs(A.diagonal()) + 1e-30)

        x, info = bicgstab(A, b, x0=x0, M=M, rtol=1e-3, atol=0.0, maxiter=300)
        if info != 0:
            x, info = bicgstab(A, b, x0=x0, M=M, rtol=1e-2, atol=0.0, maxiter=200)
            if info != 0:
                print(f"  BiCGSTAB [pressure] stalled (info={info}), "
                      f"res={np.linalg.norm(A @ x - b):.2e}")
        return x

    # ------------------------------------------------------------------
    # SIMPLE loop
    # ------------------------------------------------------------------

    def Solve(self, max_iterations=600, tolerance=1e-6):
        """
        Main SIMPLE loop.

        Convergence is based on the RMS of the continuity residual (mass
        imbalance per cell).  This avoids the mesh‑size dependence of the raw
        L2 norm and matches the per‑cell errors that a visualiser would show.
        """
        self.initialize_conditions()
        a_P_u = a_P_v = None

        # Optional: track initial RMS to report normalised drop later
        initial_cont_rms = None

        for iteration in range(max_iterations):
            self._iteration = iteration
            self.U_old = self.U.copy()

            self.SIMPLE_UPDATE_FACE_FLUX_AND_DIFFUSSION(a_P_u, a_P_v)

            if not np.all(np.isfinite(self.U)):
                print(f"NaN/Inf in U at iteration {iteration}"); break

            A_mom, b_x, b_y, a_P_u = self.assemble_momentum_both()
            a_P_v = a_P_u

            if not (np.all(np.isfinite(b_x)) and np.all(np.isfinite(b_y))):
                print(f"NaN/Inf in RHS at iteration {iteration}"); break

            u_star, v_star = self.GET_VAR_STAR(A_mom, b_x, b_y)

            if not (np.all(np.isfinite(u_star)) and np.all(np.isfinite(v_star))):
                print(f"NaN/Inf in u* at iteration {iteration}"); break

            # Consistent Rhie‑Chow flux for starred velocities
            U_star_2d = np.column_stack((u_star, v_star))
            phi_star = self._compute_rhie_chow_flux(U_star_2d, a_P_u, a_P_v)

            A_p, b_p = self.ASSEMBLE_PRESSURE_CORRECTION(a_P_u, a_P_v, phi_star)

            # Impose p' = 0 at outlet cells inside the linear system
            outlet_cells = np.unique(self._own_out)
            self._impose_dirichlet_on_system(A_p, b_p, outlet_cells, 0.0)

            p_prime = self.GET_VAR_CORRECTED(A_p, b_p)

            if not np.all(np.isfinite(p_prime)):
                print(f"NaN/Inf in p' at iteration {iteration}"); break

            self.CORRECT_PRESSURE_AND_VELOCITY(p_prime, a_P_u, a_P_v, u_star, v_star)

            if not np.all(np.isfinite(self.U)):
                print(f"NaN/Inf in corrected U at iteration {iteration}"); break

            # -----------------------------------------------------------------
            # Residual computation — mesh‑size‑invariant metrics
            # -----------------------------------------------------------------
            # Continuity residual (mass imbalance)
            res_cont_l2 = np.linalg.norm(b_p)
            res_cont_rms = res_cont_l2 / np.sqrt(self.Nc)
            res_cont_max = np.max(np.abs(b_p))

            # Momentum residuals (based on the *current* linearisation — see note)
            r_u = A_mom @ self.U[:, 0] - b_x
            r_v = A_mom @ self.U[:, 1] - b_y
            res_u_l2 = np.linalg.norm(r_u)
            res_v_l2 = np.linalg.norm(r_v)
            res_u_rms = res_u_l2 / np.sqrt(self.Nc)
            res_v_rms = res_v_l2 / np.sqrt(self.Nc)
            res_u_max = np.max(np.abs(r_u))
            res_v_max = np.max(np.abs(r_v))

            if initial_cont_rms is None:
                initial_cont_rms = max(res_cont_rms, 1e-16)

            # -----------------------------------------------------------------
            # Printout (every 10 iterations)
            # -----------------------------------------------------------------
            if iteration % 10 == 0:
                self.health_check(iteration, a_P_u)
                print(f"Iter {iteration:4d}: "
                    f"Cont (RMS)={res_cont_rms:.2e} (max)={res_cont_max:.2e}, "
                    f"U (RMS)={res_u_rms:.2e} (max)={res_u_max:.2e}, "
                    f"V (RMS)={res_v_rms:.2e} (max)={res_v_max:.2e}")

            # -----------------------------------------------------------------
            # Convergence check — RMS continuity residual
            # -----------------------------------------------------------------
            if iteration> 50:
                if res_cont_rms < tolerance:
                    print(f"\nConverged at iteration {iteration}!")
                    print(f"  Cont (RMS) = {res_cont_rms:.2e}")
                    print(f"  U    (RMS) = {res_u_rms:.2e}")
                    print(f"  V    (RMS) = {res_v_rms:.2e}")
                    break
        else:
            print(f"\nDid not converge in {max_iterations} iterations — "
                f"final continuity RMS = {res_cont_rms:.2e}")

        # Per‑cell absolute residuals for the visualiser (already in your code)
        self.final_res_cont = np.abs(b_p)
        res_u_local = r_u
        res_v_local = r_v
        self.final_res_mom = np.sqrt(res_u_local**2 + res_v_local**2)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Helper: consistent Rhie‑Chow flux for arbitrary velocity field
    # ------------------------------------------------------------------
    def _compute_rhie_chow_flux(self, U_2d, a_P_u, a_P_v):
        """
        Compute face mass flux `phi` (rho * U_f · Sf) using the Rhie‑Chow
        interpolation for the given cell‑centred velocity field `U_2d` (Nc × 2).
        The current pressure field `self.P` and its precomputed gradient
        `self._last_grad_P` are used for the pressure‑smoothing term.
        """
        f_int  = self.internal_faces
        own    = self._own_i
        nei    = self._nei_i

        # Central part
        U_interp = 0.5 * (U_2d[own] + U_2d[nei])
        phi_star = self.rho * np.einsum('fj,fj->f', U_interp, self._Sf_int)

        # Rhie‑Chow correction
        # Use cached pressure gradient (computed earlier in this iteration)
        if self._last_grad_P is None:
            # fallback, should not happen
            self._last_grad_P = self.calculate_pressure_gradients()
        gP_f = 0.5 * (self._last_grad_P[own] + self._last_grad_P[nei])

        # Face coefficient: interpolated (1/a_P) * volume face
        # a_P_f = 0.25*(a_P_u[own] + a_P_u[nei] + a_P_v[own] + a_P_v[nei])
        a_P_f = np.maximum(0.25 * (a_P_u[own] + a_P_u[nei] +
                                   a_P_v[own] + a_P_v[nei]), 1e-10)

        magSf_int = self.magSf[f_int]
        n_f       = self._Sf_int / magSf_int[:, None]
        dp_interp = np.einsum('fj,fj->f', gP_f, n_f)
        dp_actual = (self.P[nei] - self.P[own]) / self.magDf[f_int]

        vol_f  = 0.5 * (self.cell_areas[own] + self.cell_areas[nei])
        D_f    = vol_f / a_P_f

        phi_star += self.rho * D_f * (dp_interp - dp_actual) * magSf_int

        # Boundary fluxes
        phi = np.zeros(self.Nf)
        phi[f_int] = phi_star

        # Inlet – prescribed velocity
        phi[self.inlet_faces] = (
            self.rho * np.einsum('fj,j->f', self._Sf_in, self.inlet_velocity))
        # Wall – no penetration
        phi[self.wall_faces] = 0.0
        # Outlet – use the current (starred) velocity at the outlet cells
        phi[self.outlet_faces] = (
            self.rho * np.einsum('fj,fj->f',
                                 U_2d[self._own_out], self._Sf_out))

        return phi
    
    def initialize_conditions(self):
        self.P     = np.full(self.Nc, self.outlet_pressure)
        self.U     = np.zeros((self.Nc, 2))
        self.U_old = self.U.copy()
        self.phi   = np.zeros(self.Nf)
        self.diff  = np.zeros(self.Nf)
        self.SIMPLE_UPDATE_FACE_FLUX_AND_DIFFUSSION()

    # ------------------------------------------------------------------

    def SIMPLE_UPDATE_FACE_FLUX_AND_DIFFUSSION(self, a_P_u=None, a_P_v=None):
        grad_P  = self.calculate_pressure_gradients()
        f_int   = self.internal_faces
        own     = self._own_i
        nei     = self._nei_i

        U_interp = 0.5 * (self.U[own] + self.U[nei])
        phi_star = self.rho * np.einsum('fj,fj->f', U_interp, self._Sf_int)

        if a_P_u is not None:
            a_P_f      = np.maximum(0.25 * (a_P_u[own] + a_P_u[nei] +
                                             a_P_v[own] + a_P_v[nei]), 1e-10)
            gP_f       = 0.5 * (grad_P[own] + grad_P[nei])
            magSf_int  = self.magSf[f_int]
            n_f        = self._Sf_int / magSf_int[:, None]
            dp_interp  = np.einsum('fj,fj->f', gP_f, n_f)
            dp_actual  = (self.P[nei] - self.P[own]) / self.magDf[f_int]
            vol_f      = 0.5 * (self.cell_areas[own] + self.cell_areas[nei])
            D_f        = vol_f / a_P_f
            self.phi[f_int] = (phi_star
                               + self.rho * D_f * (dp_interp - dp_actual) * magSf_int)
        else:
            self.phi[f_int] = phi_star

        # Boundary fluxes
        self.phi[self.inlet_faces] = (
            self.rho * np.einsum('fj,j->f', self._Sf_in, self.inlet_velocity))
        self.phi[self.wall_faces]  = 0.0
        self.phi[self.outlet_faces] = (
            self.rho * np.einsum('fj,fj->f', self.U[self._own_out], self._Sf_out))

        mu = self.viscosity
        self.diff[f_int]                = mu * self.magSf[f_int]              / self.magDf[f_int]
        self.diff[self.inlet_faces]     = mu * self.magSf[self.inlet_faces]   / self.magDf[self.inlet_faces]
        self.diff[self.outlet_faces]    = 0.0
        self.diff[self.wall_faces]      = mu * self.magSf[self.wall_faces]    / self.magDf[self.wall_faces]

    # ------------------------------------------------------------------

    def assemble_momentum_both(self):
        """
        Build the momentum matrix A and both RHS vectors (b_x, b_y) in a
        single pass.  A is identical for u and v — same convective flux F,
        same diffusion D, same boundary contributions — so we build it once
        and return two RHS vectors.

        This halves the matrix assembly cost and, crucially, means the Jacobi
        preconditioner (just diag(A)) is computed once and shared for both
        the u-solve and the v-solve.
        """
        f_int   = self.internal_faces
        own_i   = self._own_i;  nei_i   = self._nei_i
        own_in  = self._own_in; own_w   = self._own_w; own_out = self._own_out

        F     = self.phi[f_int]
        D     = self.diff[f_int]
        F_in  = self.phi[self.inlet_faces]; D_in = self.diff[self.inlet_faces]
        D_w   = self.diff[self.wall_faces]
        F_out = self.phi[self.outlet_faces]

        data = np.concatenate([
            np.maximum( F, 0) + D,
            -(np.maximum(-F, 0) + D),
            np.maximum(-F, 0) + D,
            -(np.maximum( F, 0) + D),
            D_in,
            D_w,
            np.maximum(F_out, 0),
        ])

        A = self._make_csr(self._mom_csr, data)

        # Build p_face once; used for both axes
        p_face = np.empty(self.Nf)
        p_face[f_int]             = 0.5 * (self.P[own_i] + self.P[nei_i])
        p_face[self.inlet_faces]  = self.P[own_in]
        p_face[self.outlet_faces] = self.outlet_pressure
        p_face[self.wall_faces]   = self.P[own_w]

        b_x = np.zeros(self.Nc)
        b_y = np.zeros(self.Nc)

        for axis, b in ((0, b_x), (1, b_y)):
            b -= np.bincount(self._all_owner,
                             weights=p_face * self.Sf[:, axis],
                             minlength=self.Nc)
            b += np.bincount(nei_i,
                             weights=p_face[f_int] * self.Sf[f_int, axis],
                             minlength=self.Nc)
            b += np.bincount(own_in,
                             weights=(D_in - F_in) * self.inlet_velocity[axis],
                             minlength=self.Nc)

        # Under-relaxation — applied to the shared matrix and both RHS
        a_P     = A.diagonal().copy()
        alpha_u = self.alpha_u
        A.setdiag(a_P / alpha_u)
        if hasattr(self, 'U_old'):
            relax = ((1 - alpha_u) / alpha_u) * a_P
            b_x  += relax * self.U_old[:, 0]
            b_y  += relax * self.U_old[:, 1]

        return A, b_x, b_y, a_P


    # ------------------------------------------------------------------

    def GET_VAR_STAR(self, A_mom, b_x, b_y):
        """Solve u* and v* sharing the same matrix and Jacobi preconditioner."""
        u_star = self._solve_momentum(A_mom, b_x, 'mom_u')
        v_star = self._solve_momentum(A_mom, b_y, 'mom_v')
        v_max  = np.linalg.norm(self.inlet_velocity) * 5.0
        return np.clip(u_star, -v_max, v_max), np.clip(v_star, -v_max, v_max)

    # ------------------------------------------------------------------

        # ------------------------------------------------------------------
    # Modified assembly – pressure correction now uses consistent flux
    # ------------------------------------------------------------------
    def ASSEMBLE_PRESSURE_CORRECTION(self, a_P_u, a_P_v, phi_star):
        """
        Assemble pressure‑correction matrix A and RHS b.
        `phi_star` is the face mass flux (rho * U_f · Sf) computed via
        Rhie‑Chow interpolation for the starred velocity field.
        """
        own_i  = self._own_i;  nei_i  = self._nei_i
        own_in = self._own_in; own_out = self._own_out; own_w = self._own_w

        # Pressure‑velocity coupling coefficients (d = Sf² / a_P per face)
        def _d(Sf_slice, own):
            return self.rho*(Sf_slice[:, 0]**2 / a_P_u[own] +
                    Sf_slice[:, 1]**2 / a_P_v[own])

        d_int = _d(self._Sf_int, own_i)
        d_in  = _d(self._Sf_in,  own_in)
        d_out = _d(self._Sf_out, own_out)
        d_w   = _d(self._Sf_w,   own_w)

        data = np.concatenate([
            d_int, -d_int, d_int, -d_int,
            d_in, d_out, d_w,
        ])
        A = self._make_csr(self._pcorr_csr, data)

        # RHS: mass imbalance using the consistent Rhie‑Chow flux
        b = np.zeros(self.Nc)
        # Internal faces
        mass_flux_int = phi_star[self.internal_faces]
        b -= np.bincount(own_i, weights=mass_flux_int, minlength=self.Nc)
        b += np.bincount(nei_i, weights=mass_flux_int, minlength=self.Nc)

        # Inlet (prescribed flow, outflow from owner)
        mass_flux_in = phi_star[self.inlet_faces]
        b -= np.bincount(own_in, weights=mass_flux_in, minlength=self.Nc)

        # Outlet (outflow from domain, subtract from owner)
        mass_flux_out = phi_star[self.outlet_faces]
        b -= np.bincount(own_out, weights=mass_flux_out, minlength=self.Nc)

        # Wall faces have zero flux, no contribution to continuity
        return A, b

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Impose Dirichlet condition (p' = value) on a subset of cells
    # ------------------------------------------------------------------
    def _impose_dirichlet_on_system(self, A, b, cells, value):
        """
        Modify CSR matrix `A` and vector `b` in‑place: for each cell in `cells`
        set the corresponding row to identity (diag = 1, off‑diag = 0)
        and `b[cell] = value`.
        """
        if len(cells) == 0:
            return
        A_diag = A.diagonal()   # view, will be updated
        for i in cells:
            row_start = A.indptr[i]
            row_end   = A.indptr[i+1]
            for j in range(row_start, row_end):
                if A.indices[j] == i:
                    A.data[j] = 1.0
                else:
                    A.data[j] = 0.0
            b[i] = value
            
            
    # ------------------------------------------------------------------
    # No longer zeroes p' after solve – condition is already in the system
    # ------------------------------------------------------------------
    def GET_VAR_CORRECTED(self, A_p, b_p):
        p_prime = self._solve_pressure(A_p, b_p)
        # Outlet condition is now built into the matrix – no extra zeroing
        return p_prime

    # ------------------------------------------------------------------
    # Corrected pressure / velocity update without global pressure shift
    # ------------------------------------------------------------------
    def CORRECT_PRESSURE_AND_VELOCITY(self, p_prime, a_P_u, a_P_v, u_star, v_star):
        alpha_p = self.alpha_p
        self.P += alpha_p * p_prime
        # Removed the global pressure shift that forced the average outlet
        # pressure to match outlet_pressure. The fixed‑pressure outlet BC
        # already sets the pressure level.

        # Compute gradient of p' using the old P as a temporary placeholder
        P_tmp  = self.P.copy()
        self.P = p_prime
        grad_p_prime = self.calculate_pressure_gradients(is_correction=True)
        self.P = P_tmp
        
        self.U[:, 0] = u_star -  (self.cell_areas / a_P_u) * grad_p_prime[:, 0]
        self.U[:, 1] = v_star -  (self.cell_areas / a_P_v) * grad_p_prime[:, 1]
    # ------------------------------------------------------------------

    def calculate_pressure_gradients(self, is_correction=False):
        grad_P = np.zeros((self.Nc, 2))

        own = self._grad_own
        nei = self._grad_nei
        P_f = 0.5 * (self.P[own] + self.P[nei])

        contrib = P_f[:, None] * self._Sf_int       # (Nf_int, 2)
        for i in range(2):
            grad_P[:, i] += np.bincount(own, weights=contrib[:, i], minlength=self.Nc)
            grad_P[:, i] -= np.bincount(nei, weights=contrib[:, i], minlength=self.Nc)

        P_f_b = self.P[self._own_b].copy()
        if is_correction:
            P_f_b[self._is_outlet_bnd] = 0.0
        else:
            P_f_b[self._is_outlet_bnd] = self.outlet_pressure

        contrib_b = P_f_b[:, None] * self._Sf_bnd   # (Nf_bnd, 2)
        for i in range(2):
            grad_P[:, i] += np.bincount(self._own_b,
                                         weights=contrib_b[:, i],
                                         minlength=self.Nc)

        grad_P /= self.cell_areas[:, None]

        if not is_correction:
            self._last_grad_P = grad_P   # cache for health_check

        return grad_P

    # ------------------------------------------------------------------

    def health_check(self, iteration, a_P_u):
        print(f"\n--- Health Check Iteration {iteration} ---")
        print(f"  U range:    [{np.nanmin(self.U):.2e}, {np.nanmax(self.U):.2e}]")
        print(f"  P range:    [{np.nanmin(self.P):.2e}, {np.nanmax(self.P):.2e}]")
        print(f"  Phi range:  [{np.nanmin(self.phi):.2e}, {np.nanmax(self.phi):.2e}]")
        print(f"  Min a_P:    {np.min(a_P_u):.2e}")
        g = (self._last_grad_P if self._last_grad_P is not None
             else self.calculate_pressure_gradients())
        print(f"  Max grad_P: {np.max(np.abs(g)):.2e}")
        print(f"---------------------------------\n")