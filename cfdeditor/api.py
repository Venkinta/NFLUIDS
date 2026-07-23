"""Scripting entry point for headless parametric studies.

The CFD core (Mesher, Solver) has always been fully headless — this module
just packages the pattern test_holes.py / test_force_balance.py already use
by hand (Mesher -> mesh() -> solver_data_pipeline() -> Solver -> Solve())
into one documented call, so a sweep script doesn't need to hand-copy it.

Example:
    from cfdeditor.geometry_helpers import rect_lines
    from cfdeditor.api import run_solve

    lines = rect_lines(0, 0, 600, 100, {
        0: "Wall", 1: "Pressure Outlet", 2: "Wall", 3: "Velocity Inlet",
    })
    handle = run_solve(
        lines,
        mesh_params=dict(n_layers=3, growth_factor=1.2, thickness=1.0,
                          spacing=5.0, r=4.0, unit_to_meters=0.001),
        solver_params=dict(inlet_velocity=[0.1, 0.0], outlet_pressure=0.0,
                            rho=1000.0, viscosity=1.0),
    )
    print(handle.results.P.mean())
    handle.export_vtu("case1.vtu")
"""
import contextlib
import io
from dataclasses import dataclass

from .mesher import Mesher
from .solver import Solver
from .solver_protocol import SolverResults
from .vtuIO import export_vtu
from .meshIO import save_mesh_for_solver, save_results

_MESH_CTOR_KEYS = (
    'n_layers', 'growth_factor', 'thickness', 'spacing', 'r',
    'unit_to_meters', 'refinement_zones', 'bc_spacing_map',
)

_SOLVER_REQUIRED_KEYS = ('inlet_velocity', 'outlet_pressure', 'rho', 'viscosity')
_SOLVER_OPTIONAL_KEYS = ('alpha_u', 'alpha_p', 'max_iterations', 'tolerance')


class SolveDiverged(RuntimeError):
    """Raised when the solver never completed a single successful step()."""


@dataclass
class SolveHandle:
    """Bundles a solve's results with the mesh data that produced them, so a
    parametric-sweep script can export/save each case without re-running the
    meshing pipeline just to get mesh_data back.
    """
    results: SolverResults
    mesh_data: dict

    def export_vtu(self, filepath):
        export_vtu(self.mesh_data, filepath,
                   P=self.results.P, U=self.results.U,
                   res_cont=self.results.res_cont, res_mom=self.results.res_mom)

    def save_mesh(self, filepath):
        save_mesh_for_solver(self.mesh_data, filepath)

    def save_results(self, filepath):
        save_results(self.results, filepath)


def run_solve(lines, mesh_params, solver_params, smooth_params=None, verbose=True):
    """Mesh `lines` and solve, returning a SolveHandle.

    mesh_params: kwargs for Mesher.__init__ — n_layers, growth_factor,
        thickness, spacing, r, unit_to_meters, refinement_zones,
        bc_spacing_map (see Mesher docstring for each).
    solver_params: kwargs for Solver.__init__ — inlet_velocity,
        outlet_pressure, rho, viscosity (required), plus optional alpha_u,
        alpha_p, max_iterations, tolerance (see Solver docstring for each).
    smooth_params: optional dict of kwargs for Mesher.smooth_mesh() (passes,
        relaxation, tolerance_deg). If None (default), smoothing is skipped.
    verbose: if False, suppresses the mesh/solve stage-by-stage stdout
        output (Mesher and Solver print unconditionally) — useful when
        running many cases in a sweep. Defaults to True (unchanged output).

    Raises SolveDiverged if the solver never completed a single successful
    step() (i.e. diverged on iteration 0) — there is no valid .results in
    that case, so this fails loudly rather than returning empty arrays.
    """
    unknown = set(mesh_params) - set(_MESH_CTOR_KEYS)
    if unknown:
        raise TypeError(f"run_solve: unknown mesh_params keys {sorted(unknown)}")

    missing = set(_SOLVER_REQUIRED_KEYS) - set(solver_params)
    if missing:
        raise TypeError(f"run_solve: missing required solver_params keys {sorted(missing)}")
    unknown = set(solver_params) - set(_SOLVER_REQUIRED_KEYS) - set(_SOLVER_OPTIONAL_KEYS)
    if unknown:
        raise TypeError(f"run_solve: unknown solver_params keys {sorted(unknown)}")

    sink = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()
    with sink:
        mesher = Mesher(lines=lines, **mesh_params)
        mesher.mesh()
        if smooth_params is not None:
            mesher.smooth_mesh(**smooth_params)
        mesh_data = mesher.solver_data_pipeline()

        inlet_velocity = solver_params['inlet_velocity']
        outlet_pressure = solver_params['outlet_pressure']
        rho = solver_params['rho']
        viscosity = solver_params['viscosity']
        solver_kwargs = {k: v for k, v in solver_params.items()
                          if k not in _SOLVER_REQUIRED_KEYS}

        solver = Solver(mesh_data, inlet_velocity, outlet_pressure, rho, viscosity, **solver_kwargs)
        solver.Solve()

    if not hasattr(solver, 'final_res_cont'):
        raise SolveDiverged(
            "solver diverged before completing a single step() — no valid "
            "results (finalize() never ran). Check inlet_velocity/rho/"
            "viscosity/mesh quality.")

    return SolveHandle(results=solver.results, mesh_data=mesh_data)
