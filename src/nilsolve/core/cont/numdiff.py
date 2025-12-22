from __future__ import annotations

from typing import Any, Dict, Callable
import numpy as np

from .spec import EqFn


def fd_jacobian_y(f: EqFn, y: np.ndarray, params: Dict[str, Any], eps: float) -> np.ndarray:
    """
    Finite-difference Jacobian dF/dy at (y, params).
    """
    y = np.asarray(y, dtype=float)
    f0 = np.asarray(f(y, params), dtype=float)
    d = y.size
    J = np.zeros((d, d), dtype=float)

    for j in range(d):
        step = eps * (1.0 + abs(y[j]))
        y1 = y.copy()
        y1[j] += step
        f1 = np.asarray(f(y1, params), dtype=float)
        J[:, j] = (f1 - f0) / step
    return J


def fd_dF_dmu(
    f: EqFn,
    y: np.ndarray,
    params: Dict[str, Any],
    param_name: str,
    eps: float
) -> np.ndarray:
    """
    Finite-difference derivative dF/dmu at (y, params).
    Returns vector of shape (d,).
    """
    y = np.asarray(y, dtype=float)
    mu0 = float(params[param_name])
    f0 = np.asarray(f(y, params), dtype=float)

    step = eps * (1.0 + abs(mu0))
    params2 = dict(params)
    params2[param_name] = mu0 + step
    f1 = np.asarray(f(y, params2), dtype=float)
    return (f1 - f0) / step
