import numpy as np

from .solver_protocol import SolverResults


def save_mesh_for_solver(mesh_data, filepath):
    """Dumps the solver dictionary to a compressed .npz file.

    `mesh_data` is the dict produced by Mesher.solver_data_pipeline() (which
    now also carries cell-vertex geometry for the Visualizer).  The Solver
    only reads the fields it needs; the extra arrays are ignored on load.
    """
    np.savez_compressed(filepath, **mesh_data)
    print(f"[IO] Mesh successfully exported to {filepath}")


def load_mesh_for_solver(filepath):
    """Loads a saved mesh file directly into a clean Python dictionary for the solver."""
    # allow_pickle=True is required because refinement_zones is stored as
    # an inhomogeneous Object array.
    with np.load(filepath, allow_pickle=True) as loaded:
        # Reconstruct the dictionary
        data = {key: loaded[key] for key in loaded.files}

    # Note: np.savez converts Python scalars into 0-dimensional arrays.
    # Let's cast Nc and Nf back to normal Python integers for solver sanity.
    data['Nc'] = int(data['Nc'])
    data['Nf'] = int(data['Nf'])

    print(f"[IO] Mesh successfully loaded from {filepath} ({data['Nc']} cells, {data['Nf']} faces)")
    return data


def save_results(results, filepath):
    """Dumps a SolverResults (U, P, res_cont, res_mom, extra) to a compressed
    .npz file, for persisting per-case results in a parametric study.
    """
    np.savez_compressed(
        filepath,
        U=results.U, P=results.P,
        res_cont=results.res_cont, res_mom=results.res_mom,
        extra=np.array(results.extra, dtype=object),
    )
    print(f"[IO] Results successfully exported to {filepath}")


def load_results(filepath):
    """Loads a SolverResults previously saved with save_results()."""
    with np.load(filepath, allow_pickle=True) as loaded:
        return SolverResults(
            U=loaded['U'], P=loaded['P'],
            res_cont=loaded['res_cont'], res_mom=loaded['res_mom'],
            extra=loaded['extra'].item(),
        )