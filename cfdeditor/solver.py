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
# 6. Non-Orthogonal Correction (NOC) fully integrated into the Rhie-Chow
#    flux calculation and the pressure-correction inner loop.
# 7. Self-correcting geometry wrapper added to immunize the solver against
#    stale/corrupted df and magDf arrays stored in legacy NPZ files.
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
        self.alpha_u = 0.5 #relaxation factor for velocity
        self.alpha_p = 0.3 #relaxation factor for pressure
                
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

        # ---Values of cells and faces (Explicitly copied to guarantee writability)
        self.Sf    = mesh['Sf'].copy()
        self.magSf = mesh['magSf'].copy()
        self.Cf    = mesh['Cf'].copy()
        self.df    = mesh['df'].copy()
        self.magDf = mesh['magDf'].copy()

        self.cell_centers = mesh['cell_centers']
        self.cell_areas   = mesh['cell_areas']

        #id's
        self.wall_faces     = np.where(self.boundary_tags == 0)[0]
        self.symmetry_faces = np.where(self.boundary_tags == 3)[0]
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


        # After pipeline in solver __init__, before _precompute_topology:
        small_cell_area = np.percentile(self.cell_areas, 20)  # bottom 20% = refined cells
        refined_cells = set(np.where(self.cell_areas < small_cell_area)[0])
        suspect_faces = [f for f in self.internal_faces 
                        if self.owner[f] in refined_cells or self.neighbor[f] in refined_cells]
        mistagged = [f for f in suspect_faces if self.boundary_tags[f] != -1]
        print(f"Refined-zone internal faces: {len(suspect_faces)}")
        print(f"Of those, mistagged (not -1): {len(mistagged)}")
        if mistagged:
            print(f"  Tags found: {set(self.boundary_tags[f] for f in mistagged)}")



        self._precompute_topology()

        # Enforce safety floors on geometric metrics post-correction
        self.magDf = np.maximum(self.magDf, 1e-10)
        self.magSf = np.maximum(self.magSf, 1e-10)

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

        # --- GEOMETRIC SELF-CORRECTION LAYER ---
        # Explicitly recalculate true internal face distance vectors directly from cell centers
        # to ensure resilience against stale/corrupted array inputs from old NPZ files.
        cc_own = self.cell_centers[self.owner[f_int]]
        cc_nei = self.cell_centers[self.neighbor[f_int]]
        cf_int = self.Cf[f_int]

        # Overwrite the internal slice of the global mesh distance vectors
        self.df[f_int] = cc_nei - cc_own
        self.magDf[f_int] = np.linalg.norm(self.df[f_int], axis=1)

        # Gradient computation
        self._grad_own = self.owner[f_int]
        self._grad_nei = self.neighbor[f_int]
        self._Sf_int   = self.Sf[f_int]

        # Boundary faces (ordered: inlet, outlet, wall — consistent throughout)
        self._f_bnd  = np.concatenate([self.inlet_faces,
                                        self.outlet_faces,
                                        self.wall_faces,
                                        self.symmetry_faces])
        self._own_b  = self.owner[self._f_bnd] #grab owner id and Sf for each boundary face
        self._Sf_bnd = self.Sf[self._f_bnd]
        
        sym_set = set(self.symmetry_faces.tolist())
        self._is_sym_bnd = np.array([f in sym_set for f in self._f_bnd], dtype=bool)

        outlet_set = set(self.outlet_faces.tolist())
        self._is_outlet_bnd = np.array(
            [f in outlet_set for f in self._f_bnd], dtype=bool)

        # Per-face-type aliases
        self._own_i   = self._grad_own
        self._nei_i   = self._grad_nei
        self._own_in  = self.owner[self.inlet_faces]
        self._own_w   = self.owner[self.wall_faces]
        self._own_out = self.owner[self.outlet_faces]
        self._own_sym = self.owner[self.symmetry_faces]
        self._Sf_in   = self.Sf[self.inlet_faces]
        self._Sf_w    = self.Sf[self.wall_faces]
        self._Sf_out  = self.Sf[self.outlet_faces]
        self._Sf_sym = self.Sf[self.symmetry_faces]
        self._all_owner = self.owner  # alias, no copy

        # --- NON-ORTHOGONAL GEOMETRIC DECOMPOSITION (Over-Relaxed Approach) ---
        df_int = self.df[f_int]
        Sf_int = self.Sf[f_int]
        
        dot_df_Sf = np.sum(df_int * Sf_int, axis=1)
        dot_Sf_Sf = np.sum(Sf_int * Sf_int, axis=1)
        safe_denom = np.where(np.abs(dot_df_Sf) < 1e-12, 1e-12, dot_df_Sf)
        
        # Over-relaxed scaling factor
        self._lambda_int = dot_Sf_Sf / safe_denom
        
        # Tangential non-orthogonal correction vector components
        self._E_int = self._lambda_int[:, np.newaxis] * df_int
        self._T_int = Sf_int - self._E_int

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

        # Pre-build the CSR structures.
        self._mom_csr   = self._build_csr_template(mom_rows,   mom_cols)
        self._pcorr_csr = self._build_csr_template(pcorr_rows, pcorr_cols)

        # --- DISTANCE-WEIGHTED INTERPOLATION FACTOR (Robust Form) ---
        # Distance from owner and neighbor centers to the shared face midpoint respectively
        d_Pf = np.linalg.norm(cf_int - cc_own, axis=1)
        d_Nf = np.linalg.norm(cf_int - cc_nei, axis=1)
        
        # g_x is the geometric weight assigned to the neighbor cell, invariant to total distance bugs
        self._gx_int = d_Pf / np.maximum(d_Pf + d_Nf, 1e-10)

    # ------------------------------------------------------------------
    # CSR precomputation helpers
    # ------------------------------------------------------------------

    def _build_csr_template(self, rows, cols):
        """
        Precompute the CSR sparsity structure and scatter-add index for a
        matrix whose (row, col) pattern is fixed but whose values change
        every iteration.
        """
        Nc = self.Nc
        n  = len(rows)

        sort_order = np.lexsort((cols, rows))
        rows_s = rows[sort_order]
        cols_s = cols[sort_order]

        pairs = rows_s.astype(np.int64) * Nc + cols_s.astype(np.int64)
        _, first_occ, inv_idx = np.unique(pairs, return_index=True,
                                           return_inverse=True)
        n_unique   = len(first_occ)
        rows_u     = rows_s[first_occ]
        cols_u     = cols_s[first_occ]

        row_counts  = np.bincount(rows_u, minlength=Nc)
        indptr      = np.zeros(Nc + 1, dtype=np.int32)
        indptr[1:]  = np.cumsum(row_counts)

        indices = cols_u.astype(np.int32)

        rank          = np.empty(n, dtype=np.int64)
        rank[sort_order] = np.arange(n, dtype=np.int64)
        scatter       = inv_idx[rank]

        return dict(indptr=indptr, indices=indices,
                    scatter=scatter, n_unique=n_unique)

    def _make_csr(self, csr_info, coo_data):
        """Fast CSR matrix construction from precomputed structure."""
        data = np.bincount(csr_info['scatter'], weights=coo_data,
                           minlength=csr_info['n_unique'])
        return csr_matrix((data, csr_info['indices'], csr_info['indptr']),
                          shape=(self.Nc, self.Nc), copy=False)

    # ------------------------------------------------------------------

    def _solve_momentum(self, A, b, cache_key):
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

    def Solve(self, max_iterations=1600, tolerance=1e-10):
        self.initialize_conditions()
        a_P_u = a_P_v = None
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

            # --- INNER NON-ORTHOGONAL PRESSURE CORRECTOR LOOP ---
            n_non_ortho_correctors = 2  # Standard choice for unstructured grid skewness stabilization
            p_prime = np.zeros(self.Nc)
            grad_p_prime = None

            for non_ortho_iter in range(n_non_ortho_correctors):
                if non_ortho_iter > 0:
                    # Calculate gradients of p_prime from previous inner step to apply explicitly
                    P_tmp = self.P.copy()
                    self.P = p_prime.copy()
                    grad_p_prime = self.calculate_pressure_gradients(is_correction=True)
                    self.P = P_tmp

                A_p, b_p = self.ASSEMBLE_PRESSURE_CORRECTION(a_P_u, a_P_v, phi_star, grad_p_prime=grad_p_prime)

                # Impose p' = 0 at outlet cells inside the linear system
                outlet_cells = np.unique(self._own_out)
               #self._impose_dirichlet_on_system(A_p, b_p, outlet_cells, 0.0)

                p_prime = self.GET_VAR_CORRECTED(A_p, b_p)

                if not np.all(np.isfinite(p_prime)):
                    print(f"NaN/Inf in p' at iteration {iteration}"); break

            self.CORRECT_PRESSURE_AND_VELOCITY(p_prime, a_P_u, a_P_v, u_star, v_star)

            if not np.all(np.isfinite(self.U)):
                print(f"NaN/Inf in corrected U at iteration {iteration}"); break

            # -----------------------------------------------------------------
            # Residual computation — mesh‑size‑invariant metrics
            # -----------------------------------------------------------------
            res_cont_l2 = np.linalg.norm(b_p)
            res_cont_rms = res_cont_l2 / np.sqrt(self.Nc)
            res_cont_max = np.max(np.abs(b_p))

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
            if iteration > 50:
                if res_cont_rms < tolerance:
                    print(f"\nConverged at iteration {iteration}!")
                    print(f"  Cont (RMS) = {res_cont_rms:.2e}")
                    print(f"  U    (RMS) = {res_u_rms:.2e}")
                    print(f"  V    (RMS) = {res_v_rms:.2e}")
                    break
        else:
            print(f"\nDid not converge in {max_iterations} iterations — "
                f"final continuity RMS = {res_cont_rms:.2e}")

        self.final_res_cont = np.abs(b_p)
        res_u_local = r_u
        res_v_local = r_v
        self.final_res_mom = np.sqrt(res_u_local**2 + res_v_local**2)

    # ------------------------------------------------------------------
    # Helper: consistent Rhie‑Chow flux for arbitrary velocity field
    # ------------------------------------------------------------------
    def _compute_rhie_chow_flux(self, U_2d, a_P_u, a_P_v):
        f_int  = self.internal_faces
        own    = self._own_i
        nei    = self._nei_i

        # Distance-weighted velocity interpolation
        U_interp = (1.0 - self._gx_int)[:, None] * U_2d[own] + self._gx_int[:, None] * U_2d[nei]
        phi_star = self.rho * np.einsum('fj,fj->f', U_interp, self._Sf_int)

        # Rhie‑Chow correction
        if self._last_grad_P is None:
            self._last_grad_P = self.calculate_pressure_gradients()
            
        # Distance-weighted pressure gradient interpolation
        gP_f = (1.0 - self._gx_int)[:, None] * self._last_grad_P[own] + self._gx_int[:, None] * self._last_grad_P[nei]

        # Distance-weighted diagonal matrix coefficients
        a_P_own = 0.5 * (a_P_u[own] + a_P_v[own])
        a_P_nei = 0.5 * (a_P_u[nei] + a_P_v[nei])
        a_P_f = np.maximum((1.0 - self._gx_int) * a_P_own + self._gx_int * a_P_nei, 1e-10)

        # Distance-weighted cell volume/area interpolation
        vol_f  = (1.0 - self._gx_int) * self.cell_areas[own] + self._gx_int * self.cell_areas[nei]
        D_f    = vol_f / a_P_f

        # --- NON-ORTHOGONAL RHIE-CHOW GEOMETRY UPDATES ---
        dot_gP_df = np.sum(gP_f * self.df[f_int], axis=1)
        dp_actual = self.P[nei] - self.P[own]

        # The over-relaxed approach applies the lambda multiplier to the primary orthogonal pressure delta
        phi_star += self.rho * D_f * self._lambda_int * (dot_gP_df - dp_actual)

        # Boundary fluxes
        phi = np.zeros(self.Nf)
        phi[f_int] = phi_star

        # Inlet – prescribed velocity
        phi[self.inlet_faces] = (
            self.rho * np.einsum('fj,j->f', self._Sf_in, self.inlet_velocity))
        # Wall – no penetration
        phi[self.wall_faces] = 0.0
        
        #Symmetry
        phi[self.symmetry_faces] = 0.0
        
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

        U_interp = (1.0 - self._gx_int)[:, None] * self.U[own] + self._gx_int[:, None] * self.U[nei]
        phi_star = self.rho * np.einsum('fj,fj->f', U_interp, self._Sf_int)

        if a_P_u is not None:
            a_P_own = 0.5 * (a_P_u[own] + a_P_v[own])
            a_P_nei = 0.5 * (a_P_u[nei] + a_P_v[nei])
            a_P_f = np.maximum((1.0 - self._gx_int) * a_P_own + self._gx_int * a_P_nei, 1e-10)
            
            gP_f       = (1.0 - self._gx_int)[:, None] * grad_P[own] + self._gx_int[:, None] * grad_P[nei]
            vol_f      = (1.0 - self._gx_int) * self.cell_areas[own] + self._gx_int * self.cell_areas[nei]
            D_f        = vol_f / a_P_f
            
            dot_gP_df  = np.sum(gP_f * self.df[f_int], axis=1)
            dp_actual  = self.P[nei] - self.P[own]
            self.phi[f_int] = phi_star + self.rho * D_f * self._lambda_int * (dot_gP_df - dp_actual)
        else:
            self.phi[f_int] = phi_star

        # Boundary fluxes
        self.phi[self.inlet_faces] = (
            self.rho * np.einsum('fj,j->f', self._Sf_in, self.inlet_velocity))
        self.phi[self.wall_faces]  = 0.0
        self.phi[self.outlet_faces] = (
            self.rho * np.einsum('fj,fj->f', self.U[self._own_out], self._Sf_out))
        self.phi[self.symmetry_faces] = 0.0

        mu = self.viscosity
        self.diff[f_int]                = mu * self.magSf[f_int]              / self.magDf[f_int]
        self.diff[self.inlet_faces]     = mu * self.magSf[self.inlet_faces]   / self.magDf[self.inlet_faces]
        self.diff[self.outlet_faces]    = 0.0
        self.diff[self.wall_faces]      = mu * self.magSf[self.wall_faces]    / self.magDf[self.wall_faces]
        self.diff[self.symmetry_faces] = 0.0

    # ------------------------------------------------------------------

    def assemble_momentum_both(self):
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
        p_face[f_int]             = (1.0 - self._gx_int) * self.P[own_i] + self._gx_int * self.P[nei_i]
        p_face[self.inlet_faces]  = self.P[own_in]
        p_face[self.outlet_faces] = self.outlet_pressure
        p_face[self.wall_faces]   = self.P[own_w]
        p_face[self.symmetry_faces] = self.P[self._own_sym]
        
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
        u_star = self._solve_momentum(A_mom, b_x, 'mom_u')
        v_star = self._solve_momentum(A_mom, b_y, 'mom_v')
        v_max  = np.linalg.norm(self.inlet_velocity) * 5.0
        return np.clip(u_star, -v_max, v_max), np.clip(v_star, -v_max, v_max)

    # ------------------------------------------------------------------
    # Modified assembly – handles implicit E component and explicit T corrections
    # ------------------------------------------------------------------
    def ASSEMBLE_PRESSURE_CORRECTION(self, a_P_u, a_P_v, phi_star, grad_p_prime=None):
        own_i  = self._own_i;  nei_i  = self._nei_i
        own_in = self._own_in; own_out = self._own_out; own_w = self._own_w

        # Precompute the consistent face-centered isotropic velocity coefficient array
        a_P_own = 0.5 * (a_P_u[own_i] + a_P_v[own_i])
        a_P_nei = 0.5 * (a_P_u[nei_i] + a_P_v[nei_i])
        a_P_f   = np.maximum((1.0 - self._gx_int) * a_P_own + self._gx_int * a_P_nei, 1e-10)
        vol_f   = (1.0 - self._gx_int) * self.cell_areas[own_i] + self._gx_int * self.cell_areas[nei_i]
        D_f     = vol_f / a_P_f
        
        # Implicit coefficient matrix data derived from the parallel component (E)
        d_int = self.rho * D_f * self._lambda_int

        # Retain original consistent mapping for boundary cells
        def _d_bnd(Sf_slice, own):
            return self.rho * (Sf_slice[:, 0]**2 / np.maximum(a_P_u[own], 1e-10) + 
                               Sf_slice[:, 1]**2 / np.maximum(a_P_v[own], 1e-10))

        d_in  = _d_bnd(self._Sf_in,  own_in)
        d_out = _d_bnd(self._Sf_out, own_out)
        d_w   = _d_bnd(self._Sf_w,   own_w)

        data = np.concatenate([
            d_int, -d_int, d_int, -d_int,
            d_in, d_out, d_w,
        ])
        A = self._make_csr(self._pcorr_csr, data)

        # RHS: mass imbalance using the consistent Rhie‑Chow flux
        b = np.zeros(self.Nc)
        
        # Internal faces base starred flux
        mass_flux_int = phi_star[self.internal_faces].copy()
        
        # If an explicit gradient tracker is available from inner loops, apply tangential correction
        if grad_p_prime is not None:
            # Interpolate cell-centered correction gradients to faces
            grad_p_f = (1.0 - self._gx_int)[:, None] * grad_p_prime[own_i] + self._gx_int[:, None] * grad_p_prime[nei_i]
            # Add explicit source component: rho * D_f * (grad(p')_f · T_f)
            explicit_flux_pcorr = self.rho * D_f * np.sum(grad_p_f * self._T_int, axis=1)
            mass_flux_int += explicit_flux_pcorr

        b -= np.bincount(own_i, weights=mass_flux_int, minlength=self.Nc)
        b += np.bincount(nei_i, weights=mass_flux_int, minlength=self.Nc)

        # Inlet
        mass_flux_in = phi_star[self.inlet_faces]
        b -= np.bincount(own_in, weights=mass_flux_in, minlength=self.Nc)

        # Outlet
        mass_flux_out = phi_star[self.outlet_faces]
        b -= np.bincount(own_out, weights=mass_flux_out, minlength=self.Nc)

        return A, b

    # ------------------------------------------------------------------
    
    def _impose_dirichlet_on_system(self, A, b, cells, value):
        if len(cells) == 0:
            return
        A_diag = A.diagonal()   # view
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

    def GET_VAR_CORRECTED(self, A_p, b_p):
        p_prime = self._solve_pressure(A_p, b_p)
        return p_prime

    # ------------------------------------------------------------------

    def CORRECT_PRESSURE_AND_VELOCITY(self, p_prime, a_P_u, a_P_v, u_star, v_star):
        alpha_p = self.alpha_p
        self.P += alpha_p * p_prime

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
        P_f = (1.0 - self._gx_int) * self.P[own] + self._gx_int * self.P[nei]

        contrib = P_f[:, None] * self._Sf_int       # (Nf_int, 2)
        for i in range(2):
            grad_P[:, i] += np.bincount(own, weights=contrib[:, i], minlength=self.Nc)
            grad_P[:, i] -= np.bincount(nei, weights=contrib[:, i], minlength=self.Nc)

        P_f_b = self.P[self._own_b].copy()
        if is_correction:
            P_f_b[self._is_outlet_bnd] = 0.0
            P_f_b[self._is_sym_bnd]    = self.P[self._own_b[self._is_sym_bnd]]  # already owner P, explicit for clarity
        else:
            P_f_b[self._is_outlet_bnd] = self.outlet_pressure
            # symmetry: already self.P[owner], no change needed — it's the zero-gradient condition
         
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
        
        # Track the geometric interpolation weights to ensure self-correction is active
        print(f"  g_x Mean:   {np.mean(self._gx_int):.4f} (Min: {np.min(self._gx_int):.4f}, Max: {np.max(self._gx_int):.4f})")
        print(f"---------------------------------\n")
        
        # In health_check(), add:
        f_int = self.internal_faces
        own, nei = self._own_i, self._nei_i
        area_ratio = self.cell_areas[own] / np.maximum(self.cell_areas[nei], 1e-20)
        print(f"  Cell area ratio (own/nei) at internal faces:")
        print(f"    Mean={area_ratio.mean():.2f}, Max={area_ratio.max():.2f}, "
            f">4x count: {(area_ratio > 4).sum() + (area_ratio < 0.25).sum()}")

        lambda_stats = self._lambda_int
        print(f"  lambda_int: mean={lambda_stats.mean():.2f}, max={lambda_stats.max():.2f}, "
            f">10 count: {(lambda_stats > 10).sum()}")
        
        # Split U stats by refined vs coarse cells
        small_area_threshold = np.percentile(self.cell_areas, 30)
        refined_mask = self.cell_areas < small_area_threshold
        coarse_mask  = ~refined_mask
        print(f"  U mean (refined cells): {np.mean(self.U[refined_mask, 0]):.4f}")
        print(f"  U mean (coarse cells):  {np.mean(self.U[coarse_mask, 0]):.4f}")
        print(f"  a_P mean (refined):     {np.mean(a_P_u[refined_mask]):.4e}")
        print(f"  a_P mean (coarse):      {np.mean(a_P_u[coarse_mask]):.4e}")
        print(f"  area/a_P mean (refined):{np.mean(self.cell_areas[refined_mask]/a_P_u[refined_mask]):.4e}")
        print(f"  area/a_P mean (coarse): {np.mean(self.cell_areas[coarse_mask]/a_P_u[coarse_mask]):.4e}")