from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve

from .spec import EquilibriumSpec, ContinuationConfig
from .numdiff import fd_jacobian_y, fd_dF_dmu


@dataclass
class CorrectorResult:
    y: np.ndarray
    mu: float
    success: bool
    n_iter: int
    residual_norm: float
    message: str


def initial_tangent(
    spec: EquilibriumSpec,
    cfg: ContinuationConfig,
    y: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    """
    Compute tangent (dy/ds, dmu/ds) using:
      J dy + dF_dmu dmu = 0, choose dmu=1, dy = -J^{-1} dF_dmu, then normalize.
    """
    if spec.jac is not None:
        J = np.asarray(spec.jac(y, params), dtype=float)
    else:
        J = fd_jacobian_y(spec.f, y, params, cfg.fd_eps_y)

    dFmu = fd_dF_dmu(spec.f, y, params, spec.param_name, cfg.fd_eps_mu)  # (d,)
    try:
        dy = solve(J, -dFmu, assume_a="gen")
        dmu = 1.0
    except Exception:
        # fallback: least squares
        dy, *_ = np.linalg.lstsq(J, -dFmu, rcond=None)
        dmu = 1.0

    vec = np.concatenate([dy, np.array([dmu], dtype=float)])
    s = float(norm(vec))
    if s == 0.0 or not np.isfinite(s):
        # last resort
        dy = np.zeros_like(y)
        dmu = 1.0
        s = 1.0
    dy /= s
    dmu /= s
    return dy, float(dmu)


def corrector_pseudo_arclength(
    spec: EquilibriumSpec,
    cfg: ContinuationConfig,
    y_pred: np.ndarray,
    mu_pred: float,
    dy_ds: np.ndarray,
    dmu_ds: float,
    y_init: np.ndarray,
    mu_init: float,
) -> CorrectorResult:
    """
    Solve augmented system:
      F(y, mu) = 0
      (y - y_pred)·dy_ds + (mu - mu_pred)*dmu_ds = 0
    via Newton on z = [y; mu].
    """
    y = np.asarray(y_init, dtype=float).copy()
    mu = float(mu_init)

    for it in range(1, cfg.newton_max_iter + 1):
        params = dict(spec.params)
        params[spec.param_name] = mu

        F = np.asarray(spec.f(y, params), dtype=float)
        g = float(np.dot((y - y_pred), dy_ds) + (mu - mu_pred) * dmu_ds)

        res = np.concatenate([F, np.array([g], dtype=float)])
        r = float(norm(res))
        if r <= cfg.newton_tol:
            return CorrectorResult(y=y, mu=mu, success=True, n_iter=it - 1, residual_norm=r, message="converged")

        if spec.jac is not None:
            J = np.asarray(spec.jac(y, params), dtype=float)
        else:
            J = fd_jacobian_y(spec.f, y, params, cfg.fd_eps_y)

        dFmu = fd_dF_dmu(spec.f, y, params, spec.param_name, cfg.fd_eps_mu).reshape(-1, 1)

        # Augmented Jacobian:
        # [ J   dF/dmu ]
        # [ dy^T  dmu  ]
        d = y.size
        A = np.zeros((d + 1, d + 1), dtype=float)
        A[:d, :d] = J
        A[:d, d:] = dFmu
        A[d, :d] = dy_ds.reshape(1, -1)
        A[d, d] = dmu_ds

        try:
            step = solve(A, -res, assume_a="gen")
        except Exception as e:
            return CorrectorResult(y=y, mu=mu, success=False, n_iter=it - 1, residual_norm=r, message=f"aug solve failed: {e}")

        dy = step[:d]
        dmu = float(step[d])

        if not cfg.newton_damp:
            y = y + dy
            mu = mu + dmu
            continue

        # Damped Newton on augmented residual norm
        alpha = 1.0
        ok = False
        for _ in range(cfg.damp_max_backtracks):
            y_try = y + alpha * dy
            mu_try = mu + alpha * dmu
            params_try = dict(spec.params)
            params_try[spec.param_name] = mu_try
            F_try = np.asarray(spec.f(y_try, params_try), dtype=float)
            g_try = float(np.dot((y_try - y_pred), dy_ds) + (mu_try - mu_pred) * dmu_ds)
            res_try = np.concatenate([F_try, np.array([g_try], dtype=float)])
            if float(norm(res_try)) < r:
                y = y_try
                mu = mu_try
                ok = True
                break
            alpha *= cfg.damp_factor

        if not ok:
            y = y + alpha * dy
            mu = mu + alpha * dmu

    params = dict(spec.params)
    params[spec.param_name] = mu
    F = np.asarray(spec.f(y, params), dtype=float)
    g = float(np.dot((y - y_pred), dy_ds) + (mu - mu_pred) * dmu_ds)
    res = np.concatenate([F, np.array([g], dtype=float)])
    r = float(norm(res))
    return CorrectorResult(y=y, mu=mu, success=False, n_iter=cfg.newton_max_iter, residual_norm=r, message="max_iter reached")
