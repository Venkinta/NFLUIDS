import threading
import queue
import time
import numpy as np
import imgui

from .solver_protocol import SolverProtocol


_MAX_PLOT_POINTS = 800   # rolling window length for residual history


class SolverPanel:
    """Threading wrapper and live-monitor UI for any SolverProtocol solver.

    Owns the solver thread lifecycle, shared communication channels, and
    the ImGui control panel. main.py creates one of these when a solve is
    requested, then calls draw() every frame while in the SOLVING state.
    When self.finished is True, main.py reads solver.P / solver.U /
    solver.final_res_* and transitions to VISUALIZER.

    Controls exposed in the panel
    ------------------------------
    Stop        — signal thread to halt after the current step.
    Pause       — suspend the loop between iterations.
    Resume      — continue from a paused state.
    +1 Step     — advance exactly one iteration then pause again.
    Run to N    — run until iteration N then pause.
    Viz interval — how often (in iterations) a live field snapshot is pushed.
    Open Visualizer — available when DONE or PAUSED; commits result and exits.
    """

    def __init__(self, solver: SolverProtocol, renderer,
                 max_iterations: int = 1600,
                 viz_interval: int = 10):
        assert isinstance(solver, SolverProtocol), \
            "solver must implement SolverProtocol (initialize, step, finalize, field_snapshot)"

        self.solver          = solver
        self.renderer        = renderer
        self.max_iterations  = max_iterations
        self.viz_interval    = viz_interval

        # --- Thread control ---
        self._thread       = None
        self._stop_event   = threading.Event()
        self._pause_event  = threading.Event()
        # Run until this iteration index, then pause. Starts as max so it
        # doesn't pause until the user requests it.
        self._run_to       = max_iterations

        # --- Communication channels (solver thread → main thread) ---
        # Unbounded: residuals are small and we want every data point plotted.
        self._res_queue  = queue.Queue()
        # Size-1: only the latest snapshot matters; older ones are dropped.
        self._viz_queue  = queue.Queue(maxsize=1)

        # --- Residual history (main-thread only, written by _drain_queues) ---
        self._hist_cont: list[float] = []
        self._hist_u:    list[float] = []
        self._hist_v:    list[float] = []

        # Latest scalar values for inline readout
        self._last_cont = float('nan')
        self._last_u    = float('nan')
        self._last_v    = float('nan')

        # ImGui plot_lines buffers — log10 float32 arrays rebuilt each drain
        self._plot_cont = np.zeros(0, dtype=np.float32)
        self._plot_u    = np.zeros(0, dtype=np.float32)
        self._plot_v    = np.zeros(0, dtype=np.float32)

        # --- Public state ---
        self.current_iteration = 0
        # IDLE | RUNNING | PAUSED | DONE | DIVERGED
        self.state    = "IDLE"
        # Set True when the thread is fully done and the panel is ready to hand
        # its result to main.py for Visualizer creation.
        self.finished = False

        # Latest live field snapshot owned by main thread
        self.viz_snapshot = None   # dict{'U': ndarray, 'P': ndarray} or None

        # ImGui input buffer for "Run to" field
        self._run_to_input = max_iterations

        # Auto-start immediately
        self._start()

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def _start(self):
        self._stop_event.clear()
        self._pause_event.clear()
        self.state = "RUNNING"
        self.solver.initialize_conditions()
        self._thread = threading.Thread(target=self._solve_loop, daemon=True)
        self._thread.start()

    def _solve_loop(self):
        """Body of the solver thread.  Runs one step per loop iteration,
        pushes results to shared queues, and responds to control events."""
        state       = {}          # opaque solver state dict
        last_result = None

        for iteration in range(self.max_iterations):

            # 1. Hard stop check
            if self._stop_event.is_set():
                break

            # 2. Pause / run-to check (evaluated BEFORE running the step so
            #    "paused at iteration N" means step N has not been run yet)
            if self._pause_event.is_set() or iteration >= self._run_to:
                self._pause_event.set()
                self._res_queue.put({'type': 'paused', 'iteration': iteration})
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.04)
                if self._stop_event.is_set():
                    break

            # 3. One solver iteration
            state['iteration'] = iteration
            result = self.solver.step(**state)

            if result is None:
                # Diverged / NaN — abort the loop
                self._res_queue.put({'type': 'diverged', 'iteration': iteration})
                # Still try to finalize with whatever we have so the residual
                # arrays exist even if they're garbage.
                if last_result is not None:
                    self.solver.finalize(**last_result)
                self._res_queue.put({'type': 'done'})
                return

            state       = result
            last_result = result

            # 4. Push residuals to main thread
            self._res_queue.put({
                'type':      'residuals',
                'iteration': iteration,
                'residuals': result['residuals'],
            })

            # 5. Periodic live field snapshot (drop oldest if main is behind)
            if iteration % self.viz_interval == 0:
                snap = self.solver.field_snapshot
                try:
                    self._viz_queue.put_nowait(snap)
                except queue.Full:
                    try:    self._viz_queue.get_nowait()
                    except queue.Empty: pass
                    try:    self._viz_queue.put_nowait(snap)
                    except queue.Full:  pass

            # 6. Convergence check
            if result['converged']:
                self._res_queue.put({'type': 'converged', 'iteration': iteration})
                break

        else:
            self._res_queue.put({'type': 'max_iters', 'iteration': self.max_iterations})

        # Finalize and push a final snapshot (thread is done; no data race)
        if last_result is not None:
            self.solver.finalize(**last_result)

        final_snap = self.solver.field_snapshot
        try:
            self._viz_queue.put_nowait(final_snap)
        except queue.Full:
            try:    self._viz_queue.get_nowait()
            except queue.Empty: pass
            try:    self._viz_queue.put_nowait(final_snap)
            except queue.Full:  pass

        self._res_queue.put({'type': 'done'})

    # ------------------------------------------------------------------
    # Control methods (called from main thread / ImGui callbacks)
    # ------------------------------------------------------------------

    def stop(self):
        """Signal the thread to stop after the current step completes."""
        self._stop_event.set()
        self._pause_event.clear()   # unblock if currently paused

    def pause(self):
        self._pause_event.set()

    def resume(self):
        """Unpause and run until self._run_to then pause again."""
        self.state = "RUNNING"
        self._pause_event.clear()

    def step_one(self):
        """Advance exactly one iteration from a paused state, then pause."""
        # current_iteration is the next-to-run index (set from 'paused' msg)
        self._run_to = self.current_iteration + 1
        self.resume()

    def run_to(self, target: int):
        """Run until `target` iteration (inclusive), then pause."""
        self._run_to = max(self.current_iteration, target)
        self.resume()

    # ------------------------------------------------------------------
    # Main-thread update: drain queues, rebuild plot buffers
    # ------------------------------------------------------------------

    def _drain_queues(self):
        """Called once per frame from draw(). Must only run on the main thread."""
        while True:
            try:
                msg = self._res_queue.get_nowait()
            except queue.Empty:
                break

            mtype = msg['type']

            if mtype == 'residuals':
                res = msg['residuals']
                self.current_iteration = msg['iteration']
                self._last_cont = res.get('cont_rms', float('nan'))
                self._last_u    = res.get('u_rms',    float('nan'))
                self._last_v    = res.get('v_rms',    float('nan'))

                self._hist_cont.append(self._last_cont)
                self._hist_u.append(self._last_u)
                self._hist_v.append(self._last_v)

                # Rolling window
                if len(self._hist_cont) > _MAX_PLOT_POINTS:
                    self._hist_cont = self._hist_cont[-_MAX_PLOT_POINTS:]
                    self._hist_u    = self._hist_u[-_MAX_PLOT_POINTS:]
                    self._hist_v    = self._hist_v[-_MAX_PLOT_POINTS:]

                # Rebuild log10 arrays for imgui.plot_lines
                def _log(lst):
                    return np.log10(np.maximum(np.array(lst, dtype=np.float32), 1e-15))

                self._plot_cont = _log(self._hist_cont)
                self._plot_u    = _log(self._hist_u)
                self._plot_v    = _log(self._hist_v)

            elif mtype == 'paused':
                self.current_iteration = msg['iteration']
                if self.state == "RUNNING":
                    self.state = "PAUSED"

            elif mtype == 'converged':
                self.state = "DONE"
                print(f"[SolverPanel] Converged at iteration {msg['iteration']}.")

            elif mtype == 'max_iters':
                self.state = "DONE"
                print(f"[SolverPanel] Max iterations ({self.max_iterations}) reached.")

            elif mtype == 'diverged':
                self.state = "DIVERGED"
                print(f"[SolverPanel] Diverged at iteration {msg['iteration']}.")

            elif mtype == 'done':
                # Thread has fully exited; flip RUNNING/PAUSED → DONE if not
                # already resolved by a converged/max_iters/diverged message
                # (covers Stop being clicked while paused).
                if self.state in ("RUNNING", "PAUSED"):
                    self.state = "DONE"

        # Grab the latest viz snapshot if one arrived
        try:
            self.viz_snapshot = self._viz_queue.get_nowait()
        except queue.Empty:
            pass

    # ------------------------------------------------------------------
    # ImGui draw — called every frame by main.py in the SOLVING state
    # ------------------------------------------------------------------

    def draw(self, screen, camera, live_field=None):
        """live_field: optional live-preview Visualizer (has .vars/.var_idx/
        update_vbo_colors()) — if given, a "Show" combo lets the user switch
        which field is painted on the mesh while solving. Kept optional so
        SolverPanel has no hard dependency on Visualizer."""
        self._drain_queues()

        imgui.set_next_window_position(10, 10, imgui.ALWAYS)
        imgui.set_next_window_size(370, 0)
        imgui.begin("Solver Monitor",
                    flags=(imgui.WINDOW_NO_MOVE |
                           imgui.WINDOW_ALWAYS_AUTO_RESIZE |
                           imgui.WINDOW_NO_COLLAPSE))

        # --- Status badge + progress ---
        _COLORS = {
            "IDLE":     (0.55, 0.55, 0.55, 1.0),
            "RUNNING":  (0.20, 0.85, 0.40, 1.0),
            "PAUSED":   (1.00, 0.75, 0.10, 1.0),
            "DONE":     (0.20, 0.70, 1.00, 1.0),
            "DIVERGED": (1.00, 0.20, 0.20, 1.0),
        }
        col = _COLORS.get(self.state, (1, 1, 1, 1))
        imgui.text_colored(f"  {self.state}  ", *col)
        imgui.same_line()
        imgui.text(f"Iter {self.current_iteration} / {self.max_iterations}")

        imgui.separator()

        # --- Residual plots ---
        pw = imgui.get_content_region_available_width()

        if len(self._plot_cont) > 1:
            ov_c = f"{self._last_cont:.2e}" if np.isfinite(self._last_cont) else "---"
            ov_u = f"U {self._last_u:.2e}"  if np.isfinite(self._last_u)    else "U ---"
            ov_v = f"V {self._last_v:.2e}"  if np.isfinite(self._last_v)    else "V ---"

            imgui.text("Continuity RMS  (log10)")
            imgui.plot_lines("##cont", self._plot_cont,
                             overlay_text=ov_c,
                             graph_size=(pw, 72))

            imgui.text("Momentum RMS  (log10)")
            imgui.plot_lines("##u", self._plot_u,
                             overlay_text=ov_u,
                             graph_size=(pw, 48))
            imgui.plot_lines("##v", self._plot_v,
                             overlay_text=ov_v,
                             graph_size=(pw, 48))
        else:
            imgui.text_colored("Waiting for first iteration...", 0.5, 0.5, 0.5, 1.0)
            imgui.dummy(pw, 180)

        imgui.separator()

        # --- Controls ---
        is_running = self.state == "RUNNING"
        is_paused  = self.state == "PAUSED"
        is_done    = self.state in ("DONE", "DIVERGED")

        if is_running or is_paused:
            if imgui.button("Stop"):
                self.stop()

        if is_running:
            imgui.same_line()
            if imgui.button("Pause"):
                self.pause()

        elif is_paused:
            imgui.same_line()
            if imgui.button("Resume"):
                self._run_to = self.max_iterations
                self.resume()
            imgui.same_line()
            if imgui.button("+1 Step"):
                self.step_one()

            # Run-to control
            imgui.push_item_width(110)
            _, self._run_to_input = imgui.input_int("##runto", self._run_to_input, step=10)
            imgui.pop_item_width()
            self._run_to_input = max(self.current_iteration + 1,
                                     min(self._run_to_input, self.max_iterations))
            imgui.same_line()
            if imgui.button("Run to iter"):
                self.run_to(self._run_to_input)

        imgui.separator()

        # Viz interval (editable at any time; takes effect next snapshot)
        imgui.push_item_width(80)
        changed, vi = imgui.input_int("Viz interval", self.viz_interval, step=5)
        imgui.pop_item_width()
        if changed:
            self.viz_interval = max(1, vi)

        if live_field is not None:
            changed_var, live_field.var_idx = imgui.combo(
                "Show", live_field.var_idx, live_field.vars)
            if changed_var:
                live_field.update_vbo_colors()

        imgui.separator()

        # --- Open Visualizer ---
        if is_done or is_paused:
            imgui.push_style_color(imgui.COLOR_BUTTON,         0.12, 0.55, 0.90, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.20, 0.65, 1.00, 1.0)
            if imgui.button("Open Visualizer", width=-1):
                self.stop()
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=1.5)
                self._drain_queues()   # pick up final snapshot
                self.finished = True
            imgui.pop_style_color(2)

        imgui.end()
