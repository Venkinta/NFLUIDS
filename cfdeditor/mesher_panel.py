import threading
import queue
import traceback
import imgui

from .mesher import MeshingCancelled


class MesherPanel:
    """Threading wrapper and live-build monitor UI for Mesher.mesh() /
    Mesher.smooth_mesh(). Mirrors SolverPanel's pattern: a background
    thread produces plain data (get_render_data() bundles, GL-free),
    the main thread drains a queue each frame and is the sole GL/VBO
    consumer.

    Controls exposed in the panel
    ------------------------------
    Cancel — signal the thread to stop at the next stage/pass boundary.
    """

    def __init__(self, mesher, mode, smooth_kwargs=None):
        assert mode in ('mesh', 'smooth'), f"unknown MesherPanel mode: {mode!r}"

        self.mesher = mesher
        self.mode = mode
        self.smooth_kwargs = smooth_kwargs or {}

        # --- Thread control ---
        self._thread = None
        self._stop_event = threading.Event()

        # --- Communication channels (worker thread -> main thread) ---
        # Size-1: only the latest stage/preview matters, older ones dropped.
        self._progress_queue = queue.Queue(maxsize=1)
        # Unbounded, but carries at most one terminal message per run.
        self._status_queue = queue.Queue()

        # --- Public state (main-thread only, written by _drain()) ---
        self.state = "RUNNING"   # RUNNING | DONE | CANCELLED | ERROR
        self.stage_label = "starting..."
        self.error_message = None
        self.latest_bundles = None   # consumed by update_meshing() each frame

        self._start()

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def _start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Body of the meshing thread."""
        try:
            if self.mode == 'mesh':
                self.mesher.mesh(progress_queue=self._progress_queue,
                                  stop_event=self._stop_event)
            else:
                self.mesher.smooth_mesh(progress_queue=self._progress_queue,
                                         stop_event=self._stop_event,
                                         **self.smooth_kwargs)
        except MeshingCancelled:
            self._status_queue.put({'type': 'cancelled'})
            return
        except Exception as exc:
            print(f"[MesherPanel] {self.mode} failed:\n{traceback.format_exc()}")
            self._status_queue.put({'type': 'error', 'message': str(exc)})
            return
        self._status_queue.put({'type': 'done'})

    def stop(self):
        """Signal the thread to stop at the next stage/pass boundary."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Main-thread update: drain queues
    # ------------------------------------------------------------------

    def _drain(self):
        """Called once per frame from draw(). Must only run on the main thread."""
        try:
            msg = self._progress_queue.get_nowait()
            self.stage_label = msg['label']
            if msg['bundles'] is not None:
                self.latest_bundles = msg['bundles']
        except queue.Empty:
            pass

        try:
            status = self._status_queue.get_nowait()
            if status['type'] == 'done':
                self.state = 'DONE'
            elif status['type'] == 'cancelled':
                self.state = 'CANCELLED'
            elif status['type'] == 'error':
                self.state = 'ERROR'
                self.error_message = status['message']
        except queue.Empty:
            pass

    # ------------------------------------------------------------------
    # ImGui draw — called every frame by main.py in the MESHING state
    # ------------------------------------------------------------------

    def draw(self):
        self._drain()

        imgui.set_next_window_position(10, 10, imgui.ALWAYS)
        imgui.set_next_window_size(320, 0)
        imgui.begin("Mesher",
                    flags=(imgui.WINDOW_NO_MOVE |
                           imgui.WINDOW_ALWAYS_AUTO_RESIZE |
                           imgui.WINDOW_NO_COLLAPSE))

        _COLORS = {
            "RUNNING":   (0.20, 0.85, 0.40, 1.0),
            "DONE":      (0.20, 0.70, 1.00, 1.0),
            "CANCELLED": (1.00, 0.75, 0.10, 1.0),
            "ERROR":     (1.00, 0.20, 0.20, 1.0),
        }
        col = _COLORS.get(self.state, (1, 1, 1, 1))
        title = "Smoothing" if self.mode == 'smooth' else "Meshing"
        imgui.text_colored(f"  {self.state}  ", *col)
        imgui.same_line()
        imgui.text(title)

        imgui.separator()
        imgui.text_wrapped(self.stage_label)

        if self.state == 'ERROR':
            imgui.text_colored(self.error_message or "unknown error", 1.0, 0.3, 0.3, 1.0)

        imgui.separator()

        if self.state == 'RUNNING':
            if imgui.button("Cancel"):
                self.stop()

        imgui.end()
