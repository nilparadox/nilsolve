from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .spec import ProblemSpec


def fhn_ode(mu: float = 1.0, a: float = 0.7, b: float = 0.8, tau: float = 12.5) -> ProblemSpec:
    """
    FitzHugh–Nagumo ODE (dimension 2) in a common form:

        v' = v - v^3/3 - w + I
        w' = (v + a - b w)/tau

    Here we use params:
      - I (input current) represented via mu for convenience
      - a, b, tau standard parameters

    This is a representative nonlinear excitable/oscillatory system used in NLD labs.
    """

    def rhs(t: float, y: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
        v, w = float(y[0]), float(y[1])
        I = float(p["I"])
        a_ = float(p["a"])
        b_ = float(p["b"])
        tau_ = float(p["tau"])
        dv = v - (v**3) / 3.0 - w + I
        dw = (v + a_ - b_ * w) / tau_
        return np.array([dv, dw], dtype=float)

    y0 = np.array([-1.0, 1.0], dtype=float)
    params = {"I": float(mu), "a": float(a), "b": float(b), "tau": float(tau)}
    return ProblemSpec(rhs=rhs, t_span=(0.0, 100.0), y0=y0, params=params, require_finite=True, require_nonnegative=False)
