from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------- Type aliases ----------
RHSFn = Callable[[float, np.ndarray, Dict[str, Any]], np.ndarray]
JacFn = Callable[[float, np.ndarray, Dict[str, Any]], np.ndarray]
InvFn = Callable[[float, np.ndarray, Dict[str, Any]], float]


# ---------- Core problem specification ----------
@dataclass(frozen=True)
class ProblemSpec:
    """
    Deterministic specification of a dynamical system.

    This is the ONLY required input for the solver engine.
    AI / UI layers will later generate this object.
    """
    rhs: RHSFn
    t_span: Tuple[float, float]
    y0: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)

    # Optional numerical helpers
    jac: Optional[JacFn] = None
    invariants: List[InvFn] = field(default_factory=list)

    # Hard constraints
    require_finite: bool = True
    require_nonnegative: bool = False


# ---------- Goal declaration ----------
@dataclass(frozen=True)
class GoalSpec:
    """
    Declares what the user wants from the solve.

    v0.1.0 uses this minimally, but it is future-proofed
    for bifurcation, continuation, and statistics.
    """
    objective: str = "trajectory"


# ---------- Time-scale / tolerance policy ----------
@dataclass(frozen=True)
class PolicySpec:
    """
    Numerical time-scale policy.

    Adaptive solvers interpret this as tolerance + step ceilings.
    """
    name: str
    rtol: float
    atol: float
    max_step: Optional[float] = None
