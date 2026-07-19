"""Characterization test for cfdeditor.app_state's AppState transition graph.

Written as the v1.8.0 refactor's milestone 6/7 safety net (see CLAUDE.md's
"Big refactors need a design pass first"): this exercises the real
per-state update handlers in cfdeditor/app_state.py using bare stand-in
objects (types.SimpleNamespace) instead of real Editor/PhysicsEditor/
Solver/SolverPanel/Visualizer instances, so it runs anywhere without a
pygame window or bound OpenGL context.

update_editor, update_solving, and update_visualizer are exercised for
real — none of their code paths need a GL context (confirmed:
PhysicsEditor's constructor does no GL work; the other two only call
methods on whatever object is passed in, no construction). update_physics's
MESH/LOAD_MESH/LOAD_VISUALIZATION/SOLVE branches, however, construct real
Mesher/Solver/SolverPanel/Visualizer objects — Visualizer.__init__ and
Mesher.rebuild_wireframe_vbos() call glGenBuffers() directly, which
requires a bound GL context, and SolverPanel.__init__ spawns a background
thread. Building fixtures heavy enough to exercise those safely would be
more test infrastructure than the code being tested, so `physics_next_state`
below mirrors just the state-transition outcome for those four branches
(not their side effects) — the "no pending action" branch has no such
side effects and is exercised via the real update_physics() instead.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from types import SimpleNamespace
from cfdeditor.app_state import AppState, AppContext, update_editor, update_physics, update_solving, update_visualizer
from cfdeditor.physics_editor import PhysicsAction


def physics_next_state(ctx):
    """Mirrors update_physics()'s next-state decision (cfdeditor/app_state.py)
    for the four PhysicsAction branches, without constructing the real
    Mesher/Solver/SolverPanel/Visualizer objects those branches build.
    See module docstring for why."""
    action = ctx.physicseditor.pending_action
    if action == PhysicsAction.LOAD_VISUALIZATION:
        return AppState.VISUALIZER
    if action == PhysicsAction.SOLVE:
        return AppState.SOLVING
    return AppState.PHYSICS


def _ctx(**kwargs):
    kwargs.setdefault("camera", None)
    kwargs.setdefault("editor", None)
    return AppContext(**kwargs)


CASES = [
    ("EDITOR stays EDITOR while drawing",
     update_editor,
     _ctx(editor=SimpleNamespace(finished=False)),
     AppState.EDITOR),

    ("EDITOR -> PHYSICS when CAD finished",
     update_editor,
     _ctx(editor=SimpleNamespace(finished=True, lines=[], unit_idx=0)),
     AppState.PHYSICS),

    ("PHYSICS stays PHYSICS with no pending action",
     update_physics,
     _ctx(physicseditor=SimpleNamespace(pending_action=None)),
     AppState.PHYSICS),

    ("PHYSICS stays PHYSICS after Mesh/Remesh",
     physics_next_state,
     _ctx(physicseditor=SimpleNamespace(pending_action=PhysicsAction.MESH)),
     AppState.PHYSICS),

    ("PHYSICS stays PHYSICS after Load Mesh",
     physics_next_state,
     _ctx(physicseditor=SimpleNamespace(pending_action=PhysicsAction.LOAD_MESH)),
     AppState.PHYSICS),

    ("PHYSICS -> VISUALIZER on Load Visualization (skips SOLVING)",
     physics_next_state,
     _ctx(physicseditor=SimpleNamespace(pending_action=PhysicsAction.LOAD_VISUALIZATION)),
     AppState.VISUALIZER),

    ("PHYSICS -> SOLVING on Solve",
     physics_next_state,
     _ctx(physicseditor=SimpleNamespace(pending_action=PhysicsAction.SOLVE)),
     AppState.SOLVING),

    ("SOLVING stays SOLVING while running",
     update_solving,
     _ctx(solver_panel=SimpleNamespace(finished=False, viz_snapshot=None)),
     AppState.SOLVING),

    ("SOLVING -> VISUALIZER when solver_panel finished",
     update_solving,
     _ctx(solver_panel=SimpleNamespace(finished=True),
          solver=SimpleNamespace(results=SimpleNamespace(P=None, U=None, res_cont=None, res_mom=None)),
          live_field=SimpleNamespace(update_fields=lambda *a, **kw: None,
                                     update_vbo_colors=lambda: None)),
     AppState.VISUALIZER),

    ("VISUALIZER stays VISUALIZER while viewing",
     update_visualizer,
     _ctx(visualizer=SimpleNamespace(finished=False)),
     AppState.VISUALIZER),

    ("VISUALIZER -> PHYSICS on Back to Physics",
     update_visualizer,
     _ctx(visualizer=SimpleNamespace(finished=True, destroy=lambda: None)),
     AppState.PHYSICS),
]


def main():
    failures = []
    for desc, predicate, ctx, expected in CASES:
        actual = predicate(ctx)
        ok = actual == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}: got {actual}, expected {expected}")
        if not ok:
            failures.append(desc)

    ok = not failures
    print("\n=== RESULT ===")
    print(f"  {len(CASES) - len(failures)}/{len(CASES)} transition edges hold")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
