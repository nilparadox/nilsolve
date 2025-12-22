from typing import List, Optional, Tuple
import time
import numpy as np

from nilsolve.models import ProblemSpec, CandidateResult
from nilsolve.core.evaluate_proc import solve_one_candidate


# Stiff-first ordering is critical
SOLVER_POOL_STANDARD = [
    ("BDF",   "loose",    1e-4, 1e-7),
    ("Radau", "standard", 1e-6, 1e-9),
    ("BDF",   "standard", 1e-6, 1e-9),
    ("RK45",  "loose",    1e-4, 1e-7),
    ("DOP853","loose",    1e-4, 1e-7),
    ("RK45",  "standard", 1e-6, 1e-9),
    ("DOP853","standard", 1e-6, 1e-9),
]

SOLVER_POOL_THOROUGH = SOLVER_POOL_STANDARD + [
    ("Radau", "tight",    1e-8, 1e-11),
    ("BDF",   "tight",    1e-8, 1e-11),
    ("DOP853","tight",    1e-8, 1e-11),
    ("RK45",  "tight",    1e-8, 1e-11),
]


def _pool_for_budget(budget: str):
    b = (budget or "standard").lower().strip()
    if b in ("thorough", "deep", "full"):
        return SOLVER_POOL_THOROUGH
    return SOLVER_POOL_STANDARD


def _endpoint_relerr(a: CandidateResult, b: CandidateResult) -> float:
    """
    Cheap accuracy proxy: relative difference between endpoint states.
    """
    try:
        ya = np.asarray(a.quality_metrics.get("y_final"))  # may not exist
        yb = np.asarray(b.quality_metrics.get("y_final"))
        if ya.size == 0 or yb.size == 0:
            return float("inf")
        num = np.linalg.norm(ya - yb)
        den = np.linalg.norm(yb) + a.atol
        return float(num / den)
    except Exception:
        return float("inf")


def rank_methods(
    problem: ProblemSpec,
    budget: str = "standard",
    max_wall_s: float = 20.0,
    pilot_cap: float = 5.0,
    verbose: bool = True,
) -> Tuple[List[CandidateResult], Optional[CandidateResult]]:

    pool = _pool_for_budget(budget)
    leaderboard: List[CandidateResult] = []
    start = time.time()

    ACC_ALPHA = 10.0  # accuracy penalty weight

    for method, policy, rtol, atol in pool:
        if (time.time() - start) > max_wall_s:
            break

        # First (baseline) pilot
        res = solve_one_candidate(
            problem=problem,
            method=method,
            policy_name=policy,
            rtol=rtol,
            atol=atol,
            max_wall_s=pilot_cap,
        )

        # Default: no accuracy info
        res.dt_halving_relerr = None

        # Accuracy proxy only if baseline is usable
        if res.success and res.quality_status in ("PASS", "WARN"):
            tight = solve_one_candidate(
                problem=problem,
                method=method,
                policy_name=policy + "_halfstep",
                rtol=rtol * 0.5,
                atol=atol * 0.5,
                max_wall_s=pilot_cap,
            )

            if tight.success:
                # Compare endpoints using stored quality metrics
                try:
                    ya = np.asarray(tight.quality_metrics.get("y_final"))
                    yb = np.asarray(res.quality_metrics.get("y_final"))
                    num = np.linalg.norm(ya - yb)
                    den = np.linalg.norm(ya) + atol
                    relerr = float(num / den)
                except Exception:
                    relerr = float("inf")
            else:
                relerr = float("inf")

            res.dt_halving_relerr = relerr

            # Penalize score
            if np.isfinite(relerr):
                res.score *= (1.0 + ACC_ALPHA * relerr)
            else:
                res.score = float("inf")

        leaderboard.append(res)

    leaderboard.sort(key=lambda r: r.score)
    return leaderboard, (leaderboard[0] if leaderboard else None)


def solve_best(
    problem: ProblemSpec,
    budget: str = "standard",
    max_wall_s: float = 20.0,
    pilot_cap: float = 5.0,
) -> CandidateResult:

    leaderboard, selected = rank_methods(
        problem=problem,
        budget=budget,
        max_wall_s=max_wall_s,
        pilot_cap=pilot_cap,
        verbose=False,
    )
    if selected is None:
        raise RuntimeError("No solver candidates produced a result.")
    return selected
