# src/nilsolve/core/quality.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

QualityStatus = str  # "PASS" | "WARN" | "FAIL"


@dataclass(frozen=True)
class QualityConfig:
    # Finite-ness
    fail_on_nan: bool = True

    # Monotonic time
    fail_on_nonmonotonic_t: bool = True

    # Non-negativity checks
    nonneg_warn_tol: float = 1e-12
    nonneg_fail_tol: float = 1e-6

    # Invariant drift checks (relative drift)
    inv_warn_rel_tol: float = 1e-8
    inv_fail_rel_tol: float = 1e-4

    # Defect (residual) checks (scaled RMS defect)
    defect_enabled: bool = True
    defect_warn_rel: float = 1e2
    defect_fail_rel: float = 1e4


@dataclass(frozen=True)
class QualityResult:
    status: QualityStatus
    metrics: Dict[str, float]
    notes: List[str]


def _status_rank(status: QualityStatus) -> int:
    if status == "PASS":
        return 0
    if status == "WARN":
        return 1
    return 2  # "FAIL"


def _combine_status(cur: QualityStatus, new: QualityStatus) -> QualityStatus:
    return new if _status_rank(new) > _status_rank(cur) else cur


def _is_finite(arr: np.ndarray) -> bool:
    return bool(np.isfinite(arr).all())


def _monotonic_increasing(t: np.ndarray) -> bool:
    if t.size <= 1:
        return True
    return bool(np.all(np.diff(t) > 0))


def _as_vec(x: Union[float, np.ndarray], d: int) -> np.ndarray:
    if np.isscalar(x):
        return np.full((d,), float(x), dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full((d,), float(arr[0]), dtype=float)
    if arr.size != d:
        raise ValueError(f"Expected scalar or length-{d} vector, got length {arr.size}")
    return arr


def _scale(atol: Union[float, np.ndarray], rtol: Union[float, np.ndarray], y: np.ndarray) -> np.ndarray:
    """
    Per-component scale used for weighting:
      s_i(t) = atol_i + rtol_i * |y_i(t)|
    We return a (n,d) array.
    """
    n, d = y.shape
    atol_v = _as_vec(atol, d)
    rtol_v = _as_vec(rtol, d)
    return atol_v[None, :] + rtol_v[None, :] * np.abs(y)


def _finite_diff_derivative(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    dy/dt approx on returned grid.
    y: (n,d), t: (n,)
    """
    n, d = y.shape
    dydt = np.zeros((n, d), dtype=float)
    if n <= 1:
        return dydt

    dt0 = t[1] - t[0]
    dtn = t[-1] - t[-2]
    if dt0 > 0:
        dydt[0] = (y[1] - y[0]) / dt0
    else:
        dydt[0] = np.nan
    if dtn > 0:
        dydt[-1] = (y[-1] - y[-2]) / dtn
    else:
        dydt[-1] = np.nan

    for i in range(1, n - 1):
        dt = t[i + 1] - t[i - 1]
        if dt > 0:
            dydt[i] = (y[i + 1] - y[i - 1]) / dt
        else:
            dydt[i] = np.nan
    return dydt


def assess_solution_quality(
    *,
    t: np.ndarray,
    y: np.ndarray,
    f_rhs: Callable[[float, np.ndarray], np.ndarray],
    atol: Union[float, np.ndarray],
    rtol: Union[float, np.ndarray],
    state_names: Optional[Sequence[str]] = None,
    nonnegative: Optional[Sequence[Union[int, str]]] = None,
    invariants: Optional[Sequence[Callable[[float, np.ndarray], float]]] = None,
    config: Optional[QualityConfig] = None,
) -> QualityResult:
    """
    Solver-agnostic quality gate.
    Returns PASS/WARN/FAIL + metrics + notes.
    """
    cfg = config or QualityConfig()
    notes: List[str] = []
    metrics: Dict[str, float] = {}
    status: QualityStatus = "PASS"

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Shape sanity
    if t.ndim != 1 or y.ndim != 2 or y.shape[0] != t.shape[0]:
        return QualityResult(
            status="FAIL",
            metrics={"shape_ok": 0.0},
            notes=[f"Invalid shapes: t{t.shape}, y{y.shape}"],
        )
    n, d = y.shape
    metrics["n_points"] = float(n)
    metrics["n_state"] = float(d)
    metrics["shape_ok"] = 1.0

    # Finite checks
    finite_ok = _is_finite(t) and _is_finite(y)
    metrics["finite_ok"] = 1.0 if finite_ok else 0.0
    if not finite_ok:
        notes.append("Non-finite values detected in t or y (NaN/Inf).")
        status = _combine_status(status, "FAIL" if cfg.fail_on_nan else "WARN")

    # Monotonic time
    mono_ok = _monotonic_increasing(t)
    metrics["monotonic_t_ok"] = 1.0 if mono_ok else 0.0
    if not mono_ok:
        notes.append("Time grid is not strictly increasing.")
        status = _combine_status(status, "FAIL" if cfg.fail_on_nonmonotonic_t else "WARN")

    # Non-negativity constraints
    if nonnegative:
        if state_names is None:
            state_names = [f"x{i}" for i in range(d)]
        name_to_idx = {str(nm): i for i, nm in enumerate(state_names)}

        idxs: List[int] = []
        for item in nonnegative:
            if isinstance(item, int):
                if 0 <= item < d:
                    idxs.append(item)
            else:
                key = str(item)
                if key in name_to_idx:
                    idxs.append(name_to_idx[key])

        idxs = sorted(set(idxs))
        if idxs:
            min_vals = np.min(y[:, idxs], axis=0)
            min_overall = float(np.min(min_vals))
            metrics["nonneg_min"] = min_overall

            # "neg_mass": integral of negative parts (rough trapezoid)
            neg_mass = 0.0
            if n >= 2:
                dt = np.diff(t)
                for j, k in enumerate(idxs):
                    neg = np.maximum(0.0, -y[:, k])
                    neg_mass += float(np.sum(0.5 * (neg[:-1] + neg[1:]) * dt))
            metrics["nonneg_neg_mass"] = float(neg_mass)

            if min_overall < -cfg.nonneg_fail_tol:
                notes.append(f"Non-negativity violated (min={min_overall:.3e} < -{cfg.nonneg_fail_tol:g}).")
                status = _combine_status(status, "FAIL")
            elif min_overall < -cfg.nonneg_warn_tol:
                notes.append(f"Non-negativity slightly violated (min={min_overall:.3e}).")
                status = _combine_status(status, "WARN")
        else:
            metrics["nonneg_min"] = float("nan")
            metrics["nonneg_neg_mass"] = float("nan")
            notes.append("Non-negativity constraints provided but no valid indices/names matched.")
            status = _combine_status(status, "WARN")

    # Invariant drift
    if invariants:
        inv_rel_drifts: List[float] = []
        inv_abs_drifts: List[float] = []
        for k, inv in enumerate(invariants):
            try:
                i0 = float(inv(float(t[0]), y[0].copy()))
                vals = np.array([float(inv(float(tt), yy.copy())) for tt, yy in zip(t, y)], dtype=float)
                drift_abs = float(np.max(np.abs(vals - i0)))
                denom = max(1.0, abs(i0))
                drift_rel = drift_abs / denom
                metrics[f"invariant_{k}_drift_abs"] = drift_abs
                metrics[f"invariant_{k}_drift_rel"] = drift_rel
                inv_abs_drifts.append(drift_abs)
                inv_rel_drifts.append(drift_rel)
            except Exception as e:
                notes.append(f"Invariant {k} evaluation failed: {e}")
                status = _combine_status(status, "WARN")

        if inv_rel_drifts:
            worst_rel = float(np.max(inv_rel_drifts))
            worst_abs = float(np.max(inv_abs_drifts))
            metrics["invariant_drift_rel_max"] = worst_rel
            metrics["invariant_drift_abs_max"] = worst_abs

            if worst_rel > cfg.inv_fail_rel_tol:
                notes.append(f"Invariant drift too large (rel={worst_rel:.3e} > {cfg.inv_fail_rel_tol:g}).")
                status = _combine_status(status, "FAIL")
            elif worst_rel > cfg.inv_warn_rel_tol:
                notes.append(f"Invariant drift noticeable (rel={worst_rel:.3e}).")
                status = _combine_status(status, "WARN")

    # Defect (residual) check
    if cfg.defect_enabled and n >= 3 and finite_ok and mono_ok:
        try:
            dydt_fd = _finite_diff_derivative(t, y)  # (n,d)
            f_eval = np.zeros_like(y, dtype=float)
            for i in range(n):
                f_eval[i] = np.asarray(f_rhs(float(t[i]), y[i].copy()), dtype=float).reshape(d)
            defect = dydt_fd - f_eval  # (n,d)

            sc = _scale(atol, rtol, y)  # (n,d)
            sc = np.maximum(sc, 1e-30)

            # scaled RMS defect
            rms = float(np.sqrt(np.mean((defect / sc) ** 2)))
            metrics["defect_scaled_rms"] = rms

            if rms > cfg.defect_fail_rel:
                notes.append(f"Defect too large (scaled RMS={rms:.3e} > {cfg.defect_fail_rel:g}).")
                status = _combine_status(status, "FAIL")
            elif rms > cfg.defect_warn_rel:
                notes.append(f"Defect noticeable (scaled RMS={rms:.3e}).")
                status = _combine_status(status, "WARN")
        except Exception as e:
            notes.append(f"Defect check failed: {e}")
            status = _combine_status(status, "WARN")

    metrics["quality_status_rank"] = float(_status_rank(status))
    return QualityResult(status=status, metrics=metrics, notes=notes)
