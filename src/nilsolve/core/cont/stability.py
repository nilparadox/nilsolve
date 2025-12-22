from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
from scipy.linalg import eigvals

from .spec import EquilibriumSpec, ContinuationConfig
from .numdiff import fd_jacobian_y


@dataclass
class StabilityInfo:
    eigvals: np.ndarray
    max_real: float
    stable: bool


def stability_at_equilibrium(
    spec: EquilibriumSpec,
    cfg: ContinuationConfig,
    y: np.ndarray,
    params: Dict[str, Any],
) -> StabilityInfo:
    if spec.jac is not None:
        J = np.asarray(spec.jac(y, params), dtype=float)
    else:
        J = fd_jacobian_y(spec.f, y, params, cfg.fd_eps_y)

    ev = eigvals(J)
    max_real = float(np.max(np.real(ev))) if ev.size else float("nan")
    stable = bool(max_real < 0.0)
    return StabilityInfo(eigvals=ev, max_real=max_real, stable=stable)
