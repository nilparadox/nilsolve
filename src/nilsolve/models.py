from dataclasses import dataclass
from typing import Any, Dict, Tuple, Callable, List, Optional, Sequence
import numpy as np

RhsFn = Callable[[float, np.ndarray, Dict[str, Any]], np.ndarray]


@dataclass
class ProblemSpec:
    rhs: RhsFn
    t_span: Tuple[float, float]
    y0: np.ndarray
    params: Dict[str, Any]

    # Optional physical / mathematical constraints
    state_names: Optional[Sequence[str]] = None

    require_nonnegative: bool = False
    nonnegative_indices: Optional[List[int]] = None

    # Optional invariants: functions I(t, y) -> float
    invariants: Optional[List[Callable[[float, np.ndarray], float]]] = None


@dataclass
class GoalSpec:
    objective: str = "trajectory"


@dataclass
class CandidateResult:
    method: str
    policy_name: str
    rtol: float
    atol: float
    max_step: Optional[float]

    success: bool
    status: int
    message: str

    runtime_s: float
    nfev: int
    njev: int
    nlu: int

    # Legacy sanity flags (kept for backward compatibility)
    finite_ok: bool
    nonnegative_ok: bool

    # NEW: formal quality gate outputs
    quality_status: Optional[str]
    quality_metrics: Optional[Dict[str, float]]
    quality_notes: Optional[List[str]]

    # Reserved for future ranking upgrades
    invariant_drift: Optional[float]
    dt_halving_relerr: Optional[float]

    score: float
