"""
mesh_audit.py -- standalone geometric-integrity audit for NFLUIDS solver meshes.

Runs entirely on a saved mesh .npz (the file written by "Save mesh" /
meshIO.save_mesh_for_solver) -- no solver run needed.

Usage:
    python mesh_audit.py mesh_3k.npz [mesh_10k.npz ...]

Checks:
  1. PHANTOM BOUNDARY FACES -- boundary-tagged faces whose midpoint does NOT
     lie on any CAD boundary line. These are holes in the cell fabric
     (dropped sliver quads / filtered triangles / unpaired edges) that the
     pipeline silently tagged as walls (or worse, inlets). One phantom wall
     inside a channel blocks mass flux through that face and forces a local
     pressure spike.
  2. BOUNDARY LENGTH AUDIT -- sum(|Sf|) per tag vs total CAD length per tag.
     Inflated wall length = phantom walls; deficit = missing boundary faces.
  3. CELL CLOSURE -- max |sum_f (+/-)Sf| per cell, normalized by perimeter.
     Nonzero closure = a cell is missing a face entirely.
  4. NON-ORTHOGONALITY / LAMBDA -- cos(theta) = df.Sf/(|df||Sf|) and
     lambda = |Sf|^2/(df.Sf) for internal faces. lambda <= 0 breaks the
     M-matrix property of the pressure-correction system; huge lambda
     amplifies the Rhie-Chow correction at that face.
  5. INTERPOLATION WEIGHTS / AREA RATIOS -- gx extremes and owner/neighbor
     area ratios (refinement-transition quality).
"""
import sys
import numpy as np


TAG_NAMES = {-1: "internal", 0: "wall", 1: "inlet", 2: "outlet", 3: "symmetry"}


def point_to_segment_dist(pts, a, b):
    """Distance from each point in pts (N,2) to segment a-b (each (2,))."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-30:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def audit(path):
    print("=" * 72)
    print(f"  MESH AUDIT: {path}")
    print("=" * 72)
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}

    Nc, Nf = int(d['Nc']), int(d['Nf'])
    owner, neighbor = d['owner'], d['neighbor']
    Sf, Cf = d['Sf'], d['Cf']
    magSf = d['magSf']
    tags = d['boundary_tags']
    cc, areas = d['cell_centers'], d['cell_areas']
    s = float(d['unit_to_meters']) if 'unit_to_meters' in d else 1.0

    internal = np.where(tags == -1)[0]
    boundary = np.where(tags != -1)[0]
    print(f"cells={Nc}  faces={Nf}  internal={len(internal)}  "
          + "  ".join(f"{TAG_NAMES[t]}={np.sum(tags == t)}"
                      for t in (0, 1, 2, 3)))

    # ------------------------------------------------------------------
    # 1. Phantom boundary faces (needs cad_lines, world units -> SI)
    # ------------------------------------------------------------------
    n_flagged = 0
    if 'cad_lines' in d and len(d['cad_lines']) > 0:
        cad = np.asarray(d['cad_lines'], dtype=np.float64)
        segs = [(cad[i, :2] * s, cad[i, 2:4] * s, int(cad[i, 4]))
                for i in range(len(cad))]

        mids = Cf[boundary]
        dist_any = np.full(len(boundary), np.inf)
        for a, b, _tag in segs:
            dist_any = np.minimum(dist_any, point_to_segment_dist(mids, a, b))

        # A genuine boundary face midpoint sits ON a CAD line (they are all
        # straight); allow 5% of the local face length for float noise.
        tol = 0.05 * magSf[boundary] + 1e-9
        flagged = boundary[dist_any > tol]
        n_flagged = len(flagged)

        print("\n[1] Phantom boundary faces (midpoint off every CAD line):")
        if n_flagged == 0:
            print("    PASS -- every boundary face lies on a CAD line.")
        else:
            print(f"    *** FLAG: {n_flagged} boundary faces are NOT on any "
                  f"CAD line -- holes in the cell fabric. ***")
            for t in (0, 1, 2, 3):
                n_t = int(np.sum(tags[flagged] == t))
                if n_t:
                    print(f"      tagged {TAG_NAMES[t]:8s}: {n_t}")
            worst = flagged[np.argsort(
                dist_any[np.isin(boundary, flagged)])[::-1][:10]]
            for f in worst:
                print(f"      face {f:7d} tag={TAG_NAMES[int(tags[f])]:8s} "
                      f"mid=({Cf[f, 0]:.6f},{Cf[f, 1]:.6f}) m "
                      f"|Sf|={magSf[f]:.3e}")
    else:
        print("\n[1] Phantom-face check SKIPPED -- no cad_lines in this npz "
              "(old save?). Re-save the mesh to include them.")

    # ------------------------------------------------------------------
    # 2. Boundary length audit
    # ------------------------------------------------------------------
    print("\n[2] Boundary length: sum(|Sf|) per tag vs CAD length per tag:")
    if 'cad_lines' in d and len(d['cad_lines']) > 0:
        for t in (0, 1, 2, 3):
            mesh_len = float(np.sum(magSf[tags == t]))
            cad_len = sum(np.linalg.norm(b - a)
                          for a, b, tg in segs if tg == t)
            if mesh_len == 0.0 and cad_len == 0.0:
                continue
            ratio = mesh_len / cad_len if cad_len > 0 else np.inf
            mark = "" if abs(ratio - 1.0) < 0.02 else "   <-- MISMATCH"
            print(f"    {TAG_NAMES[t]:8s}: mesh {mesh_len:.6f} m | "
                  f"CAD {cad_len:.6f} m | ratio {ratio:.4f}{mark}")
    else:
        print("    SKIPPED (no cad_lines).")

    # ------------------------------------------------------------------
    # 3. Cell closure: sum of outward Sf over each cell's faces = 0
    # ------------------------------------------------------------------
    closure = np.zeros((Nc, 2))
    for i in range(2):
        closure[:, i] += np.bincount(owner, weights=Sf[:, i], minlength=Nc)
        has_n = neighbor >= 0
        closure[:, i] -= np.bincount(neighbor[has_n],
                                     weights=Sf[has_n, i], minlength=Nc)
    # Normalize by an estimate of cell perimeter
    perim = np.bincount(owner, weights=magSf, minlength=Nc)
    has_n = neighbor >= 0
    perim += np.bincount(neighbor[has_n], weights=magSf[has_n], minlength=Nc)
    rel_closure = np.linalg.norm(closure, axis=1) / np.maximum(perim, 1e-30)
    n_open = int(np.sum(rel_closure > 1e-6))
    print(f"\n[3] Cell closure |sum Sf|/perimeter: max={rel_closure.max():.2e}"
          f"  cells>1e-6: {n_open}"
          + ("   PASS" if n_open == 0 else "   *** FLAG: open cells ***"))
    if n_open:
        worst = np.argsort(rel_closure)[::-1][:5]
        for c in worst:
            print(f"      cell {c:7d} closure={rel_closure[c]:.3e} "
                  f"center=({cc[c, 0]:.6f},{cc[c, 1]:.6f}) m")

    # ------------------------------------------------------------------
    # 4. Non-orthogonality / lambda on internal faces
    #    (df recomputed from cell centers, exactly as the solver does)
    # ------------------------------------------------------------------
    df_int = cc[neighbor[internal]] - cc[owner[internal]]
    Sf_int = Sf[internal]
    mag_df = np.linalg.norm(df_int, axis=1)
    dot = np.einsum('fj,fj->f', df_int, Sf_int)
    cos_t = dot / np.maximum(mag_df * magSf[internal], 1e-30)
    lam = np.einsum('fj,fj->f', Sf_int, Sf_int) / np.where(
        np.abs(dot) < 1e-30, 1e-30, dot)
    lam_ref = magSf[internal] / np.maximum(mag_df, 1e-30)  # orthogonal value
    lam_rel = lam / lam_ref

    n_neg = int(np.sum(cos_t <= 0.0))
    n_bad = int(np.sum(cos_t < 0.17))  # >~80 deg non-orthogonality
    print(f"\n[4] Non-orthogonality: cos(theta) min={cos_t.min():.4f}  "
          f"p1={np.percentile(cos_t, 1):.4f}  "
          f"faces<0.17: {n_bad}  faces<=0 (lambda flips sign!): {n_neg}"
          + ("   PASS" if n_neg == 0 and n_bad == 0 else "   *** FLAG ***"))
    print(f"    lambda/lambda_orth: p50={np.percentile(lam_rel, 50):.2f} "
          f"p99={np.percentile(lam_rel, 99):.2f} max={lam_rel.max():.2f}")
    if n_neg or n_bad:
        worst = internal[np.argsort(cos_t)[:5]]
        for f in worst:
            print(f"      face {f:7d} cos={cos_t[np.searchsorted(internal, f)]:.4f} "
                  f"mid=({Cf[f, 0]:.6f},{Cf[f, 1]:.6f}) m")

    # ------------------------------------------------------------------
    # 5. Interpolation weights and area ratios
    # ------------------------------------------------------------------
    d_Pf = np.linalg.norm(Cf[internal] - cc[owner[internal]], axis=1)
    d_Nf = np.linalg.norm(Cf[internal] - cc[neighbor[internal]], axis=1)
    gx = d_Pf / np.maximum(d_Pf + d_Nf, 1e-30)
    ratio = areas[owner[internal]] / np.maximum(areas[neighbor[internal]], 1e-30)
    ratio = np.maximum(ratio, 1.0 / ratio)
    print(f"\n[5] gx: min={gx.min():.3f} max={gx.max():.3f} "
          f"outside [0.05,0.95]: {int(np.sum((gx < 0.05) | (gx > 0.95)))}")
    print(f"    area ratio across faces: p99={np.percentile(ratio, 99):.2f} "
          f"max={ratio.max():.2f}  >4x: {int(np.sum(ratio > 4.0))}")

    print()
    return n_flagged


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    total = 0
    for p in sys.argv[1:]:
        total += audit(p)
    sys.exit(0 if total == 0 else 2)
