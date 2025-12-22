from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .api import solve_best
from .metrics import estimate_period_from_peaks
from .spec import GoalSpec, ProblemSpec


@dataclass
class Scan2DConfig:
    mu1: str
    start1: float
    stop1: float
    n1: int

    mu2: str
    start2: float
    stop2: float
    n2: int

    component: int = 0
    t_transient: float = 0.0
    t_window: float = 0.0
    n_eval: int = 1500
    continuation_row: bool = True


def _linspace(start: float, stop: float, n: int) -> np.ndarray:
    if n < 2:
        return np.array([float(start)], dtype=float)
    return np.linspace(float(start), float(stop), int(n), dtype=float)


def _apply_override(problem: ProblemSpec, key: str, val: Any) -> ProblemSpec:
    params = dict(problem.params)
    params[key] = val
    return ProblemSpec(
        rhs=problem.rhs,
        t_span=problem.t_span,
        y0=problem.y0,
        params=params,
        jac=problem.jac,
        invariants=problem.invariants,
        require_finite=problem.require_finite,
        require_nonnegative=problem.require_nonnegative,
    )


def _replace_y0(problem: ProblemSpec, y0: np.ndarray) -> ProblemSpec:
    return ProblemSpec(
        rhs=problem.rhs,
        t_span=problem.t_span,
        y0=np.asarray(y0, dtype=float),
        params=problem.params,
        jac=problem.jac,
        invariants=problem.invariants,
        require_finite=problem.require_finite,
        require_nonnegative=problem.require_nonnegative,
    )


def _window_indices(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return np.where((t >= t0) & (t <= t1))[0]


def scan2d(problem: ProblemSpec, goal: Optional[GoalSpec], cfg: Scan2DConfig) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    goal = goal or GoalSpec(objective="trajectory")

    mu1_vals = _linspace(cfg.start1, cfg.stop1, cfg.n1)
    mu2_vals = _linspace(cfg.start2, cfg.stop2, cfg.n2)

    t0, t1 = map(float, problem.t_span)
    T = t1 - t0
    if T <= 0:
        raise ValueError("t_span must satisfy t1 > t0")

    t_trans = float(cfg.t_transient)
    t_win = float(cfg.t_window)
    if t_trans < 0 or t_win < 0:
        raise ValueError("t_transient and t_window must be >= 0")
    if t_win == 0.0:
        t_win = 0.2 * T

    t_win_start = max(t0 + t_trans, t1 - t_win)
    t_win_end = t1
    t_eval = np.linspace(t0, t1, int(cfg.n_eval), dtype=float)

    rows: List[Dict[str, Any]] = []

    for v1 in mu1_vals:
        row_problem_base = _apply_override(problem, cfg.mu1, float(v1))
        last_y_end: Optional[np.ndarray] = None

        for v2 in mu2_vals:
            p = _apply_override(row_problem_base, cfg.mu2, float(v2))

            if cfg.continuation_row and last_y_end is not None:
                p = _replace_y0(p, last_y_end)

            res = solve_best(problem=p, goal=goal)
            sol = res["sol"]

            if not sol.success:
                rows.append(
                    {
                        "mu1": float(v1),
                        "mu2": float(v2),
                        "success": False,
                        "method": res.get("best_method"),
                        "policy": getattr(res.get("best_policy"), "name", None),
                        "message": str(sol.message),
                    }
                )
                continue

            last_y_end = np.asarray(sol.y[:, -1], dtype=float)

            if sol.sol is not None:
                y_eval = sol.sol(t_eval)
                t_used = t_eval
            else:
                t_used = np.asarray(sol.t, dtype=float)
                y_eval = np.asarray(sol.y, dtype=float)

            idx = _window_indices(t_used, t_win_start, t_win_end)
            if idx.size < 10:
                idx = np.arange(max(0, len(t_used) - 50), len(t_used))

            ycomp = np.asarray(y_eval[cfg.component, idx], dtype=float)
            tcomp = np.asarray(t_used[idx], dtype=float)

            y_min = float(np.min(ycomp))
            y_max = float(np.max(ycomp))
            amp = float(y_max - y_min)
            y_mean = float(np.mean(ycomp))

            pm = estimate_period_from_peaks(tcomp, ycomp)

            rows.append(
                {
                    "mu1": float(v1),
                    "mu2": float(v2),
                    "success": True,
                    "method": res["best_method"],
                    "policy": res["best_policy"].name,
                    "rtol": float(res["best_policy"].rtol),
                    "atol": float(res["best_policy"].atol),
                    "max_step": None if res["best_policy"].max_step is None else float(res["best_policy"].max_step),
                    "runtime_s": float(res["runtime_s"]),
                    "y_min": y_min,
                    "y_max": y_max,
                    "amp": amp,
                    "y_mean": y_mean,
                    "n_peaks": int(pm.n_peaks),
                    "period_est": None if pm.period_est is None else float(pm.period_est),
                    "freq_est": None if pm.freq_est is None else float(pm.freq_est),
                    "t_window_start": float(t_win_start),
                    "t_window_end": float(t_win_end),
                    "cont_row": bool(cfg.continuation_row),
                }
            )

    return mu1_vals, mu2_vals, rows


def save_scan2d_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
