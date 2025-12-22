from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence
import numpy as np

EqFn = Callable[[np.ndarray, Dict[str, Any]], np.ndarray]
JacFn = Callable[[np.ndarray, Dict[str, Any]], np.ndarray]


@dataclass
class EquilibriumSpec:
    f: EqFn
    y_guess: np.ndarray
    params: Dict[str, Any]
    param_name: str
    jac: Optional[JacFn] = None
    state_names: Optional[Sequence[str]] = None


@dataclass
class ContinuationConfig:
    # domain
    mu_start: float
    mu_stop: float

    # base step (will become adaptive)
    ds: float = 0.05
    max_steps: int = 400

    # Newton/corrector
    newton_max_iter: int = 20
    newton_tol: float = 1e-10
    newton_damp: bool = True
    damp_factor: float = 0.5
    damp_max_backtracks: int = 12

    # finite differences
    fd_eps_y: float = 1e-7
    fd_eps_mu: float = 1e-7

    # diagnostics thresholds
    svd_sing_tol: float = 1e-10
    hopf_imag_min: float = 1e-6

    # --- event localization ---
    hopf_mu_tol: float = 1e-3
    hopf_max_refine: int = 25

    fold_mu_tol: float = 1e-3
    fold_max_refine: int = 25

    # --- adaptive ds ---
    ds_min: float = 1e-4
    ds_max: float = 0.2
    ds_shrink: float = 0.5
    ds_grow: float = 1.2

    # retry logic for failed/slow correctors
    step_retry_max: int = 6
    iter_bad: int = 10       # if corrector iterations exceed -> shrink ds
    iter_good: int = 3       # if corrector iterations low for many steps -> grow ds
    good_streak_for_grow: int = 5
