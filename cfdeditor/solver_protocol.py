from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SolverResults:
    """Post-solve fields, returned by SolverProtocol.results.

    This is the only surface main.py is allowed to depend on after a solve
    finishes — it must not reach into solver-specific attributes directly.

    `extra` is a forward-compat escape hatch for fields a future solver adds
    (e.g. a temperature solver's `extra['T']`) without requiring another
    breaking change to SolverProtocol itself.
    """
    U: np.ndarray
    P: np.ndarray
    res_cont: np.ndarray
    res_mom: np.ndarray
    extra: dict = field(default_factory=dict)


class SolverProtocol(ABC):
    """Abstract base class for all solvers.

    Any solver that wants to work with SolverPanel must implement these four
    methods. This keeps SolverPanel completely solver-agnostic: a future LES,
    compressible, or multi-phase solver just inherits from this and the rest of
    the pipeline requires zero changes.

    State convention:
        step() receives and returns an opaque **state dict. SolverPanel passes
        it back unchanged. Only 'residuals' (dict of named scalars) and
        'converged' (bool) are read by SolverPanel.

    Thread safety:
        step() and field_snapshot are called from the solver thread.
        initialize_conditions() and finalize() bracket the loop.
    """

    @abstractmethod
    def initialize_conditions(self) -> None:
        """Set initial field values (U=0, P=outlet_pressure, etc.).
        Called exactly once before the first step() call.
        """

    @abstractmethod
    def step(self, **state):
        """Run exactly one outer iteration.

        Parameters: **state — opaque state from the previous step() call.
                    Empty {} on the very first call.

        Returns a dict containing at minimum:
            'residuals': dict[str, float]  named scalar metrics for the panel.
            'converged': bool              True if convergence criterion met.
            Plus any solver-specific keys, passed back unchanged next call.

        Returns None to signal fatal divergence (NaN/Inf detected).
        """

    @abstractmethod
    def finalize(self, **final_state) -> None:
        """Compute post-solve diagnostics from the last iteration's raw data.
        Expected side-effect: populate self.final_res_cont and
        self.final_res_mom (cell-level arrays) for Visualizer.
        """

    @property
    @abstractmethod
    def field_snapshot(self) -> dict:
        """Return *copies* of the current field arrays for live visualisation.
        Must return a fresh dict with copied arrays on every call so the caller
        owns the data. At minimum: {'U': ndarray(Nc,2), 'P': ndarray(Nc,)}.
        """

    @property
    @abstractmethod
    def results(self) -> SolverResults:
        """Return the final post-solve fields as a SolverResults.
        Valid only after finalize() has been called.
        """
