import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import scipy.io

class Solver:
    def __init__(self, mesher_data, inlet_velocity, outlet_pressure, rho, viscosity):
        """
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
        """
        
        # Store fluid properties and BCs
        self.inlet_velocity = inlet_velocity
        self.outlet_pressure = outlet_pressure
        self.rho = rho
        self.viscosity = viscosity
        
        # Unpack mesh dictionary
        mesh = mesher_data
        self.Nc = mesh['Nc']
        self.Nf = mesh['Nf']
        
        # Topology arrays (1-D int)
        self.owner = mesh['owner']
        self.neighbor = mesh['neighbor']
        self.boundary_tags = mesh['boundary_tags']
        
        # Geometry arrays (2-D or 1-D float)
        self.Sf = mesh['Sf']
        self.magSf = mesh['magSf']
        self.Cf = mesh['Cf']
        self.df = mesh['df']
        self.magDf = mesh['magDf']
        
        # --- CFD SAFETY CLAMP ---
        # Prevent division by zero for any distance calculations
        self.magDf = np.maximum(self.magDf, 1e-10)
        self.magSf = np.maximum(self.magSf, 1e-10)
        # ------------------------
        
        # Cell geometry
        self.cell_centers = mesh['cell_centers']
        self.cell_areas = mesh['cell_areas']
        
        
        ###############
        
        self.wall_faces = np.where(self.boundary_tags == 0)[0]
        self.inlet_faces = np.where(self.boundary_tags == 1)[0]
        self.outlet_faces = np.where(self.boundary_tags == 2)[0]
        self.internal_faces = np.where(self.boundary_tags == -1)[0] #returns array of the indices where wall faces, inlet faces, outlet faces... are
        
        
        print(f"--- MESH SANITY CHECK ---")
        print(f"Total Cells: {self.Nc}")
        print(f"Internal Faces: {len(self.internal_faces)}")
        print(f"Inlet Faces: {len(self.inlet_faces)}")
        print(f"Outlet Faces: {len(self.outlet_faces)}")
        print(f"Wall Faces: {len(self.wall_faces)}")
        print(f"-------------------------")
        

    def Solve(self, max_iterations=1000, tolerance=1e-6):
        self.initialize_conditions()
        initial_residuals = None
        a_P_u,a_P_v = None, None


        for iteration in range(max_iterations):
            self.U_old = self.U.copy()
            
            # Step 1: Flux update
            self.SIMPLE_UPDATE_FACE_FLUX_AND_DIFFUSSION(a_P_u,a_P_v)
            
            # CHECK 1: Are velocities still finite?
            if not np.all(np.isfinite(self.U)):
                print(f"⚠️ NaN/Inf in U before momentum solve at iteration {iteration}")
                print(f"   U range: [{np.nanmin(self.U)}, {np.nanmax(self.U)}]")
                break
            
            # Step 2: Momentum (This updates a_P_u and a_P_v for the NEXT iteration)
            A_x, b_x, a_P_u = self.assemble_momentum(axis=0)
            A_y, b_y, a_P_v = self.assemble_momentum(axis=1)
            
            # CHECK 2: Is b finite?
            if not np.all(np.isfinite(b_x)) or not np.all(np.isfinite(b_y)):
                print(f"⚠️ NaN/Inf in RHS at iteration {iteration}")
                break
            
            u_star, v_star = self.GET_VAR_STAR(A_x, b_x, A_y, b_y)
            
            # CHECK 3: Is starred velocity finite?
            if not np.all(np.isfinite(u_star)) or not np.all(np.isfinite(v_star)):
                print(f"⚠️ NaN/Inf in starred velocity at iteration {iteration}")
                break
            
            # Step 3: Pressure correction (NOW PASSING u_star AND v_star)
            A_p, b_p = self.ASSEMBLE_PRESSURE_CORRECTION(a_P_u, a_P_v, u_star, v_star)
            p_prime = self.GET_VAR_CORRECTED(A_p, b_p)
            
            # CHECK 4: Is p_prime finite?
            if not np.all(np.isfinite(p_prime)):
                print(f"⚠️ NaN/Inf in pressure correction at iteration {iteration}")
                print(f"   b_p range: [{b_p.min()}, {b_p.max()}]")
                break
            
            # Step 4: Velocity correction
            self.CORRECT_PRESSURE_AND_VELOCITY(p_prime, a_P_u, a_P_v, u_star, v_star)
            
            # CHECK 5: Is corrected velocity finite?
            if not np.all(np.isfinite(self.U)):
                print(f"⚠️ NaN/Inf in corrected U at iteration {iteration}")
                break
            
            # Step 5: Check convergence
            residual_continuity = np.linalg.norm(b_p)
            residual_u = np.linalg.norm(A_x @ self.U[:, 0] - b_x)
            residual_v = np.linalg.norm(A_y @ self.U[:, 1] - b_y)
            
            # Store initial residuals
            if iteration == 0:
                initial_residuals = {
                    'cont': max(residual_continuity, 1e-10),
                    'u': max(residual_u, 1e-10),
                    'v': max(residual_v, 1e-10)
                }
            
            # Normalize by initial values
            norm_cont = residual_continuity / initial_residuals['cont']
            norm_u = residual_u / initial_residuals['u']
            norm_v = residual_v / initial_residuals['v']
            
            max_residual = max(norm_cont, norm_u, norm_v)

            self.health_check(iteration,a_P_u)


            if iteration % 10 == 0:
                print(f"Iteration {iteration}: "
                    f"Cont = {norm_cont:.2e}, "
                    f"U = {norm_u:.2e}, "
                    f"V = {norm_v:.2e}")
            
            if max_residual < tolerance:
                print(f"\n✓ Converged!")
                break
        else:
            print(f"\n⚠ Did not converge in {max_iterations} iterations")
            print(f"Final residuals: {max_residual:.2e}")




        

        
    
    def initialize_conditions(self):
        # Start with a very mild pressure gradient
        self.P = np.ones((self.Nc)) * self.outlet_pressure
        
        # Start with zero velocity (let the pressure drive the flow)
        # This is often more stable than guessing 0.6*inlet
        self.U = np.zeros((self.Nc, 2)) 
        self.U_old = self.U.copy()
        
        self.phi = np.zeros(self.Nf)
        self.diff = np.zeros(self.Nf)
        
        # Important: Call flux update once to initialize self.phi for the first pressure solve
        self.SIMPLE_UPDATE_FACE_FLUX_AND_DIFFUSSION()
        



    def SIMPLE_UPDATE_FACE_FLUX_AND_DIFFUSSION(self, a_P_u=None, a_P_v=None):
        grad_P = self.calculate_pressure_gradients()

        f_int = self.internal_faces
        own   = self.owner[f_int]
        nei   = self.neighbor[f_int]

        U_interp  = (self.U[own] + self.U[nei]) / 2.0
        phi_star  = self.rho * np.sum(U_interp * self.Sf[f_int], axis=1)

        if a_P_u is not None:
            a_P_f         = 0.5 * (a_P_u[own] + a_P_u[nei] + a_P_v[own] + a_P_v[nei]) / 2.0
            a_P_f         = np.maximum(a_P_f, 1e-10)
            gradP_f_interp= 0.5 * (grad_P[own] + grad_P[nei])
            n_f           = self.Sf[f_int] / self.magSf[f_int][:, None]
            dp_interp     = np.sum(gradP_f_interp * n_f, axis=1)
            dp_actual     = (self.P[nei] - self.P[own]) / self.magDf[f_int]
            vol_f         = 0.5 * (self.cell_areas[own] + self.cell_areas[nei])
            D_f           = vol_f / a_P_f
            self.phi[f_int] = phi_star + self.rho * D_f * (dp_interp - dp_actual) * self.magSf[f_int]
        else:
            self.phi[f_int] = phi_star

        self.phi[self.inlet_faces]  = self.rho * np.sum(self.inlet_velocity * self.Sf[self.inlet_faces], axis=1)
        self.phi[self.wall_faces]   = 0.0

        # FIX: Use actual cell velocity for outlet phi — removing the post-hoc global
        # scaling that created a phi/U inconsistency feeding into the pressure correction.
        own_out = self.owner[self.outlet_faces]
        self.phi[self.outlet_faces] = self.rho * np.sum(
            self.U[own_out] * self.Sf[self.outlet_faces], axis=1)

        self.diff[self.internal_faces] = (
            self.viscosity * self.magSf[self.internal_faces] / self.magDf[self.internal_faces])
        self.diff[self.inlet_faces]    = (
            self.viscosity * self.magSf[self.inlet_faces]    / self.magDf[self.inlet_faces])
        self.diff[self.outlet_faces]   = 0.0
        self.diff[self.wall_faces]     = (
            self.viscosity * self.magSf[self.wall_faces]     / self.magDf[self.wall_faces])

        # NOTE: Removed global mass-conservation phi scaling.
        # That scaling set phi[outlet] to satisfy continuity on paper, but left U[outlet]
        # wrong, causing ASSEMBLE_PRESSURE_CORRECTION to see a persistent spurious
        # mass imbalance. The pressure solver should be the thing that enforces continuity.
        
        
    
    def assemble_momentum(self, axis):

        b = np.zeros(self.Nc)

        # --- INTERNAL FACES ---
        f_int = self.internal_faces
        own_i = self.owner[f_int]
        nei_i = self.neighbor[f_int]

        F = self.phi[f_int]
        D = self.diff[f_int]

        rows = []
        cols = []
        data = []

        # Upwind convection + diffusion (unchanged)
        rows.append(own_i);  cols.append(own_i);  data.append(np.maximum(F, 0) + D)
        rows.append(own_i);  cols.append(nei_i);  data.append(-(np.maximum(-F, 0) + D))
        rows.append(nei_i);  cols.append(nei_i);  data.append(np.maximum(-F, 0) + D)
        rows.append(nei_i);  cols.append(own_i);  data.append(-(np.maximum(F, 0) + D))

        # --- INLET (unchanged) ---
        f_in   = self.inlet_faces
        own_in = self.owner[f_in]
        D_in   = self.diff[f_in]
        F_in   = self.phi[f_in]
        rows.append(own_in); cols.append(own_in); data.append(D_in)
        b[own_in] += (D_in - F_in) * self.inlet_velocity[axis]

        # --- WALL (unchanged) ---
        f_w   = self.wall_faces
        own_w = self.owner[f_w]
        D_w   = self.diff[f_w]
        rows.append(own_w); cols.append(own_w); data.append(D_w)

        # --- OUTLET: FIX — add zero-gradient convective stabilisation ---
        # Previously nothing was added here, leaving outlet cells under-constrained.
        # For a zero-gradient (convective) outlet: the outgoing flux adds to the diagonal.
        f_out   = self.outlet_faces
        if len(f_out) > 0:
            own_out = self.owner[f_out]
            F_out   = self.phi[f_out]
            # Outgoing flux (phi > 0) contributes to diagonal; no diffusion, no source.
            rows.append(own_out)
            cols.append(own_out)
            data.append(np.maximum(F_out, 0))

        # --- PRESSURE GRADIENT (unchanged) ---
        p_face          = np.zeros(self.Nf)
        p_face[f_int]   = 0.5 * (self.P[own_i] + self.P[nei_i])
        p_face[f_in]    = self.P[own_in]
        p_face[f_out]   = self.outlet_pressure
        p_face[f_w]     = self.P[own_w]

        np.add.at(b, self.owner,           -p_face * self.Sf[:, axis])
        np.add.at(b, self.neighbor[f_int],  p_face[f_int] * self.Sf[f_int, axis])

        # --- ASSEMBLE (unchanged) ---
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        A      = coo_matrix((data, (rows, cols)), shape=(self.Nc, self.Nc))
        A_csr  = A.tocsr()

        # Under-relaxation (unchanged)
        a_P_original    = A_csr.diagonal().copy()
        alpha_u         = 0.2
        A_csr.setdiag(a_P_original / alpha_u)
        if hasattr(self, 'U_old'):
            b += ((1 - alpha_u) / alpha_u) * a_P_original * self.U_old[:, axis]

        return A_csr, b, a_P_original                                          
            
            
    def GET_VAR_STAR(self, A_u, b_u, A_v, b_v):
        u_star = spsolve(A_u, b_u)
        v_star = spsolve(A_v, b_v)
        
        # --- VELOCITY CLAMPING ---
        # Prevents the "Iteration 2 Explosion" 
        # Don't let the velocity exceed 5x the inlet velocity in early iterations
        v_max = np.linalg.norm(self.inlet_velocity) * 5.0
        u_star = np.clip(u_star, -v_max, v_max)
        v_star = np.clip(v_star, -v_max, v_max)
        
        return u_star, v_star
    
    def ASSEMBLE_PRESSURE_CORRECTION(self, a_P_u, a_P_v, u_star, v_star):
                
        b = np.zeros(self.Nc)
        
        # Combine starred velocities into a 2D array for easy dot products
        U_star_combined = np.column_stack((u_star, v_star))
        
        # --- 1. INTERNAL FACES ---
        f_int = self.internal_faces
        own_int = self.owner[f_int]
        nei_int = self.neighbor[f_int]
        
        d_f_int = (self.Sf[f_int, 0]**2) / a_P_u[own_int] + (self.Sf[f_int, 1]**2) / a_P_v[own_int]
        
        rows = [own_int, own_int, nei_int, nei_int]
        cols = [own_int, nei_int, nei_int, own_int]
        data = [d_f_int, -d_f_int, d_f_int, -d_f_int]
        
        # Mass imbalance using U* (FIXED)
        U_interp = (U_star_combined[own_int] + U_star_combined[nei_int]) / 2.0
        mass_flux_int = self.rho * np.sum(U_interp * self.Sf[f_int], axis=1)
        np.add.at(b, own_int, -mass_flux_int)
        np.add.at(b, nei_int, mass_flux_int)
        
        # --- 2. INLET FACES ---
        f_in = self.inlet_faces
        if len(f_in) > 0:
            own_in = self.owner[f_in]
            d_f_in = (self.Sf[f_in, 0]**2) / a_P_u[own_in] + (self.Sf[f_in, 1]**2) / a_P_v[own_in]
            
            rows.append(own_in)
            cols.append(own_in)
            data.append(d_f_in)
            
            mass_flux_in = self.rho * np.sum(self.inlet_velocity * self.Sf[f_in], axis=1)
            np.add.at(b, own_in, -mass_flux_in)
        
        # --- 3. OUTLET FACES ---
        f_out = self.outlet_faces
        if len(f_out) > 0:
            own_out = self.owner[f_out]
            d_f_out = (self.Sf[f_out, 0]**2) / a_P_u[own_out] + (self.Sf[f_out, 1]**2) / a_P_v[own_out]
            
            rows.append(own_out)
            cols.append(own_out)
            data.append(d_f_out)
            
            # Mass flux using U* (FIXED)
            mass_flux_out = self.rho * np.sum(U_star_combined[own_out] * self.Sf[f_out], axis=1)
            np.add.at(b, own_out, -mass_flux_out)
        
        # --- 4. WALL FACES ---
        f_w = self.wall_faces
        if len(f_w) > 0:
            own_w = self.owner[f_w]
            d_f_w = (self.Sf[f_w, 0]**2) / a_P_u[own_w] + (self.Sf[f_w, 1]**2) / a_P_v[own_w]
            
            rows.append(own_w)
            cols.append(own_w)
            data.append(d_f_w)
        
        # --- 5. CREATE COO MATRIX ---
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        
        A = coo_matrix((data, (rows, cols)), shape=(self.Nc, self.Nc))
        A_csr = A.tocsr()
        
        return A_csr, b
    
    def GET_VAR_CORRECTED(self, A_p, b_p):
        p_prime = spsolve(A_p, b_p)

        # FIX: Enforce p' = 0 at pressure outlet cells.
        # The pressure there is already fixed; the correction must be zero.
        # Without this, grad(p') explodes near the outlet and drives velocities
        # into the velocity clamp, creating the observed fixed-point stall.
        outlet_owners = np.unique(self.owner[self.outlet_faces])
        p_prime[outlet_owners] = 0.0

        return p_prime
    
    
    def CORRECT_PRESSURE_AND_VELOCITY(self, p_prime, a_P_u, a_P_v, u_star, v_star):
        alpha_p = 0.1
        
        # 1. Update Cell Pressure
        self.P += alpha_p * p_prime
        self.P -= (np.mean(self.P[self.owner[self.outlet_faces]]) - self.outlet_pressure)
        
        # 2. Calculate Gradient of p_prime
        P_tmp = self.P.copy()
        self.P = p_prime
        grad_p_prime = self.calculate_pressure_gradients(is_correction=True) # (FIXED)
        self.P = P_tmp 
        
        # 3. Correct Velocity
        self.U[:, 0] = u_star - (self.cell_areas / a_P_u) * grad_p_prime[:, 0]
        self.U[:, 1] = v_star - (self.cell_areas / a_P_v) * grad_p_prime[:, 1]
        


    def calculate_pressure_gradients(self, is_correction=False):
        grad_P = np.zeros((self.Nc, 2))
        
        # Internal
        f_int = self.internal_faces
        own, nei = self.owner[f_int], self.neighbor[f_int]
        P_f = 0.5 * (self.P[own] + self.P[nei])
        
        for i in range(2):
            np.add.at(grad_P[:, i], own, P_f * self.Sf[f_int, i])
            np.add.at(grad_P[:, i], nei, -P_f * self.Sf[f_int, i])
            
        # Boundaries
        f_bnd = np.concatenate([self.inlet_faces, self.outlet_faces, self.wall_faces])
        own_b = self.owner[f_bnd]
        
        P_f_b = self.P[own_b].copy() # Ensure we don't alter the actual P array
        is_outlet = np.isin(f_bnd, self.outlet_faces)
        
        # (FIXED) Force p' to 0 at the boundary if we are calculating correction gradients
        if is_correction:
            P_f_b[is_outlet] = 0.0
        else:
            P_f_b[is_outlet] = self.outlet_pressure
        
        for i in range(2):
            np.add.at(grad_P[:, i], own_b, P_f_b * self.Sf[f_bnd, i])
            
        grad_P /= self.cell_areas[:, None]
        return grad_P
    


    def health_check(self, iteration, a_P_u):
            print(f"\n--- Health Check Iteration {iteration} ---")
            print(f"  U range:    [{np.nanmin(self.U):.2e}, {np.nanmax(self.U):.2e}]")
            print(f"  P range:    [{np.nanmin(self.P):.2e}, {np.nanmax(self.P):.2e}]")
            print(f"  Phi range:  [{np.nanmin(self.phi):.2e}, {np.nanmax(self.phi):.2e}]")
            
            # (FIXED) Use passed a_P_u instead of rebuilding the whole matrix
            diag_min = np.min(a_P_u)
            print(f"  Min a_P:    {diag_min:.2e}")
            
            print(f"  Max grad_P: {np.max(np.abs(self.calculate_pressure_gradients())):.2e}")
            print(f"---------------------------------\n")