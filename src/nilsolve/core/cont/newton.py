from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve

from .spec import EquilibriumSpec, ContinuationConfig
from .numdiff import fd_jacobian_y


@dataclass
class NewtonResult:
    y: np.ndarray
    success: bool
    n_iter: int
    residual_norm: float
    message: str


def newton_equilibrium(
    spec: EquilibriumSpec,
    cfg: ContinuationConfig,
    y0: np.ndarray,
    params: Dict[str, Any],
) -> NewtonResult:
    """
    Solve F(y; params)=0 with damped Newton.
    """
    y = np.asarray(y0, dtype=float).copy()

    for it in range(1, cfg.newton_max_iter + 1):
        F = np.asarray(spec.f(y, params), dtype=float)
        r = float(norm(F))
        if r <= cfg.newton_tol:
            return NewtonResult(y=y, success=True, n_iter=it - 1, residual_norm=r, message="converged")

        if spec.jac is not None:
            J = np.asarray(spec.jac(y, params), dtype=float)
        else:
            J = fd_jacobian_y(spec.f, y, params, cfg.fd_eps_y)

        try:
            step = solve(J, -F, assume_a="gen")
        except Exception as e:
            return NewtonResult(y=y, success=False, n_iter=it - 1, residual_norm=r, message=f"linear solve failed: {e}")

        if not cfg.newton_damp:
            y = y + step
            continue

        # Backtracking line search on residual norm
        alpha = 1.0
        ok = False
        for _ in range(cfg.damp_max_backtracks):
            y_try = y + alpha * step
            F_try = np.asarray(spec.f(y_try, params), dtype=float)
            if float(norm(F_try)) < r:
                y = y_try
                ok = True
                break
            alpha *= cfg.damp_factor

        if not ok:
            # If damping fails to reduce residual, still take a small step to avoid stagnation
            y = y + alpha * step

    F = np.asarray(spec.f(y, params), dtype=float)
    r = float(norm(F))
    return NewtonResult(y=y, success=False, n_iter=cfg.newton_max_iter, residual_norm=r, message="max_iter reached")
