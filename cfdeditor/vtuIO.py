import numpy as np

# VTK cell-type codes (VTK file format spec) — maps directly from
# mesher.py's cell_types (0=triangle, 1=quad).
_VTK_TRIANGLE = 5
_VTK_QUAD = 9


def _floats_to_str(arr):
    return " ".join(f"{v:.10g}" for v in np.asarray(arr, dtype=np.float64).ravel())


def _ints_to_str(arr):
    return " ".join(str(int(v)) for v in np.asarray(arr).ravel())


def export_vtu(mesh_data, filepath, P=None, U=None, res_cont=None, res_mom=None):
    """Writes an ASCII VTK XML UnstructuredGrid (.vtu) file from a mesh_data
    dict (Mesher.solver_data_pipeline() output, live or loaded from .npz),
    optionally with solved fields as CellData — for cross-validating the
    solver against other CFD codes (e.g. opening the result in ParaView).

    P/U/residuals are cell-centered (finite volume) here, which maps
    directly onto VTU's CellData rather than PointData — so points don't
    need to be deduplicated/shared across cells. Each cell writes its own
    private vertex copy (straight from cell_vertices) with sequential
    connectivity; no global vertex-merge step is needed.
    """
    Nc = int(mesh_data['Nc'])
    cell_vertices = np.asarray(mesh_data['cell_vertices'])   # (Nc, 4, 2), SI metres
    cell_nverts = np.asarray(mesh_data['cell_nverts'])       # (Nc,), 3 or 4
    cell_types = np.asarray(mesh_data['cell_types'])         # (Nc,), 0=tri, 1=quad

    points = []
    connectivity = []
    offsets = []
    types = []
    running = 0
    for ci in range(Nc):
        n = int(cell_nverts[ci])
        for vi in range(n):
            x, y = cell_vertices[ci, vi]
            points.append((x, y, 0.0))
        connectivity.extend(range(running, running + n))
        running += n
        offsets.append(running)
        types.append(_VTK_QUAD if cell_types[ci] == 1 else _VTK_TRIANGLE)

    n_points = len(points)
    points_arr = np.array(points, dtype=np.float64)

    cell_data_blocks = []
    if P is not None:
        cell_data_blocks.append(
            '<DataArray type="Float64" Name="Pressure" format="ascii">\n'
            f'{_floats_to_str(P)}\n</DataArray>'
        )
    if U is not None:
        U = np.asarray(U, dtype=np.float64)
        U3 = np.zeros((Nc, 3), dtype=np.float64)
        U3[:, :2] = U
        cell_data_blocks.append(
            '<DataArray type="Float64" Name="Velocity" NumberOfComponents="3" format="ascii">\n'
            f'{_floats_to_str(U3)}\n</DataArray>'
        )
    if res_cont is not None:
        cell_data_blocks.append(
            '<DataArray type="Float64" Name="ContinuityResidual" format="ascii">\n'
            f'{_floats_to_str(res_cont)}\n</DataArray>'
        )
    if res_mom is not None:
        cell_data_blocks.append(
            '<DataArray type="Float64" Name="MomentumResidual" format="ascii">\n'
            f'{_floats_to_str(res_mom)}\n</DataArray>'
        )
    cell_data_xml = "\n".join(cell_data_blocks)

    xml = f'''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{n_points}" NumberOfCells="{Nc}">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
{_floats_to_str(points_arr)}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
{_ints_to_str(connectivity)}
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
{_ints_to_str(offsets)}
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
{_ints_to_str(types)}
        </DataArray>
      </Cells>
      <CellData>
{cell_data_xml}
      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
'''
    with open(filepath, 'w') as f:
        f.write(xml)
    print(f"[IO] Exported VTU: {filepath} ({Nc} cells, {n_points} points)")
