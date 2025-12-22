from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svdvals

from .spec import EquilibriumSpec, ContinuationConfig
from .newton import newton_equilibrium
from .arclength import initial_tangent, corrector_pseudo_arclength
from .stability import stability_at_equilibrium
from .numdiff import fd_jacobian_y


def _jac_smin(spec: EquilibriumSpec, cfg: ContinuationConfig, y: np.ndarray, params: Dict[str, Any]) -> float:
    if spec.jac is not None:
        J = np.asarray(spec.jac(y, params), dtype=float)
    else:
        J = fd_jacobian_y(spec.f, y, params, cfg.fd_eps_y)
    s = svdvals(J)
    return float(np.min(s)) if s.size else float("nan")


def _equilibrium_stab_tangent(
    spec: EquilibriumSpec,
    cfg: ContinuationConfig,
    mu: float,
    y_guess: np.ndarray,
) -> Dict[str, Any]:
    """
    Solve equilibrium at mu, then compute:
      - stability (eig_max_real)
      - jacobian smallest singular value (jac_smin)
      - tangent (dy/ds, dmu/ds)
    """
    params = dict(spec.params)
    params[spec.param_name] = float(mu)

    newt = newton_equilibrium(spec, cfg, y_guess, params)
    if not newt.success:
        return {
            "success": False,
            "mu": float(mu),
            "y": None,
            "newton_iter": int(newt.n_iter),
            "residual_norm": float(newt.residual_norm),
            "message": str(newt.message),
        }

    y = newt.y
    stab = stability_at_equilibrium(spec, cfg, y, params)
    smin = _jac_smin(spec, cfg, y, params)
    dy_ds, dmu_ds = initial_tangent(spec, cfg, y, params)

    return {
        "success": True,
        "mu": float(mu),
        "y": y,
        "newton_iter": int(newt.n_iter),
        "residual_norm": float(newt.residual_norm),
        "eig_max_real": float(stab.max_real),
        "stable": bool(stab.stable),
        "jac_smin": float(smin),
        "dy_ds": dy_ds,
        "dmu_ds": float(dmu_ds),
        "eigvals": stab.eigvals,
    }


def _localize_hopf(
    spec: EquilibriumSpec,
    cfg: ContinuationConfig,
    lo: Dict[str, Any],
    hi: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Localize Hopf by bisection on sign change of eig_max_real (largest real-part eigenvalue).
    Stores y at mu_est.
    """
    mu_lo, mu_hi = float(lo["mu"]), float(hi["mu"])
    y_lo, y_hi = lo["y"], hi["y"]
    r_lo, r_hi = float(lo["eig_max_real"]), float(hi["eig_max_real"])

    for k in range(cfg.hopf_max_refine):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        y_guess = 0.5 * (y_lo + y_hi)

        mid = _equilibrium_stab_tangent(spec, cfg, mu_mid, y_guess)
        if not mid["success"]:
            break

        r_mid = float(mid["eig_max_real"])
        # keep the bracket that contains the sign change
        if np.sign(r_mid) == np.sign(r_lo):
            mu_lo, y_lo, r_lo = mu_mid, mid["y"], r_mid
        else:
            mu_hi, y_hi, r_hi = mu_mid, mid["y"], r_mid

        if abs(mu_hi - mu_lo) < cfg.hopf_mu_tol:
            break

    mu_est = 0.5 * (mu_lo + mu_hi)
    y_guess = 0.5 * (y_lo + y_hi)
    est = _equilibrium_stab_tangent(spec, cfg, mu_est, y_guess)

    return {
        "type": "HOPF",
        "mu_est": float(mu_est),
        "mu_lo": float(mu_lo),
        "mu_hi": float(mu_hi),
        "refine_steps": int(k + 1),
        "success": bool(est.get("success", False)),
        "y_est": est.get("y", None),
        "eig_max_real_est": float(est.get("eig_max_real", float("nan"))),
        "jac_smin_est": float(est.get("jac_smin", float("nan"))),
        "dmu_ds_est": float(est.get("dmu_ds", float("nan"))),
    }


def _localize_fold(
    spec: EquilibriumSpec,
    cfg: ContinuationConfig,
    lo: Dict[str, Any],
    hi: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Localize fold by bisection on root of dmu/ds between opposite signs.
    Stores y at mu_est.
    """
    mu_lo, mu_hi = float(lo["mu"]), float(hi["mu"])
    y_lo, y_hi = lo["y"], hi["y"]
    q_lo, q_hi = float(lo["dmu_ds"]), float(hi["dmu_ds"])

    # Require sign change for meaningful bisection
    if np.sign(q_lo) == np.sign(q_hi):
        mu_est = 0.5 * (mu_lo + mu_hi)
        est = _equilibrium_stab_tangent(spec, cfg, mu_est, 0.5 * (y_lo + y_hi))
        return {
            "type": "FOLD",
            "mu_est": float(mu_est),
            "mu_lo": float(mu_lo),
            "mu_hi": float(mu_hi),
            "refine_steps": 0,
            "success": bool(est.get("success", False)),
            "y_est": est.get("y", None),
            "dmu_ds_est": float(est.get("dmu_ds", float("nan"))),
            "jac_smin_est": float(est.get("jac_smin", float("nan"))),
            "eig_max_real_est": float(est.get("eig_max_real", float("nan"))),
            "note": "no sign change in dmu/ds",
        }

    for k in range(cfg.fold_max_refine):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        y_guess = 0.5 * (y_lo + y_hi)

        mid = _equilibrium_stab_tangent(spec, cfg, mu_mid, y_guess)
        if not mid["success"]:
            break

        q_mid = float(mid["dmu_ds"])
        if np.sign(q_mid) == np.sign(q_lo):
            mu_lo, y_lo, q_lo = mu_mid, mid["y"], q_mid
        else:
            mu_hi, y_hi, q_hi = mu_mid, mid["y"], q_mid

        if abs(mu_hi - mu_lo) < cfg.fold_mu_tol:
            break

    mu_est = 0.5 * (mu_lo + mu_hi)
    y_guess = 0.5 * (y_lo + y_hi)
    est = _equilibrium_stab_tangent(spec, cfg, mu_est, y_guess)

    return {
        "type": "FOLD",
        "mu_est": float(mu_est),
        "mu_lo": float(mu_lo),
        "mu_hi": float(mu_hi),
        "refine_steps": int(k + 1),
        "success": bool(est.get("success", False)),
        "y_est": est.get("y", None),
        "dmu_ds_est": float(est.get("dmu_ds", float("nan"))),
        "jac_smin_est": float(est.get("jac_smin", float("nan"))),
        "eig_max_real_est": float(est.get("eig_max_real", float("nan"))),
    }


def continue_equilibria(
    spec: EquilibriumSpec,
    cfg: ContinuationConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Equilibrium continuation with:
      - pseudo-arclength predictor-corrector
      - stability via eigenvalues of Jacobian
      - HOPF localization
      - FOLD localization
      - adaptive step-size ds with retry
      - event rows include equilibrium state (store-y)
    """
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    # Initial equilibrium at mu_start
    init = _equilibrium_stab_tangent(spec, cfg, cfg.mu_start, np.asarray(spec.y_guess, dtype=float))
    if not init["success"]:
        report = {
            "kind": "equilibrium_continuation",
            "param": spec.param_name,
            "runtime_s": float(time.time() - t0),
            "init_success": False,
            "init": {k: v for k, v in init.items() if k != "eigvals"},
            "events": events,
        }
        return rows, report

    y = init["y"]
    mu = float(init["mu"])
    dy_ds = init["dy_ds"]
    dmu_ds = float(init["dmu_ds"])

    prev: Optional[Dict[str, Any]] = None
    good_streak = 0
    ds = float(cfg.ds)

    def in_range(mu_val: float) -> bool:
        return (mu_val <= cfg.mu_stop) if (cfg.mu_start < cfg.mu_stop) else (mu_val >= cfg.mu_stop)

    step = 0
    while step < cfg.max_steps and in_range(mu):
        params = dict(spec.params)
        params[spec.param_name] = mu

        # Current stability and diagnostics at accepted point
        cur = _equilibrium_stab_tangent(spec, cfg, mu, y)
        if not cur["success"]:
            # Should not happen for accepted point; record and stop.
            fail_row = {
                "step": int(step),
                "mu": float(mu),
                "success": False,
                "newton_iter": int(cur.get("newton_iter", 0)),
                "residual_norm": float(cur.get("residual_norm", float("nan"))),
                "corr_iter": "",
                "corr_residual": "",
                "eig_max_real": float("nan"),
                "stable": False,
                "jac_smin": float("nan"),
                "event": f"ACCEPTED_POINT_FAIL:{cur.get('message','')}",
            }
            for i in range(len(y)):
                fail_row[f"y_{i}"] = float("nan")
            rows.append(fail_row)
            break

        # Main row for this accepted point
        row = {
            "step": int(step),
            "mu": float(mu),
            "success": True,
            "newton_iter": int(cur["newton_iter"]),
            "residual_norm": float(cur["residual_norm"]),
            "corr_iter": "",
            "corr_residual": "",
            "eig_max_real": float(cur["eig_max_real"]),
            "stable": bool(cur["stable"]),
            "jac_smin": float(cur["jac_smin"]),
            "event": "",
        }
        for i, v in enumerate(np.asarray(cur["y"], dtype=float).reshape(-1)):
            row[f"y_{i}"] = float(v)
        rows.append(row)

        # Event detection against previous accepted point
        if prev is not None:
            # HOPF: sign change in eig_max_real and presence of complex pair (heuristic)
            r0 = float(prev["eig_max_real"])
            r1 = float(cur["eig_max_real"])
            crossed = (r0 < 0.0 and r1 > 0.0) or (r0 > 0.0 and r1 < 0.0)
            if crossed:
                imag = np.abs(np.imag(cur["eigvals"]))
                has_complex = bool(np.any(imag > cfg.hopf_imag_min))
                if has_complex:
                    hopf = _localize_hopf(spec, cfg, prev, cur)
                    events.append(hopf)
                    if hopf.get("success") and hopf.get("y_est") is not None:
                        y_est = np.asarray(hopf["y_est"], dtype=float)
                        loc_row = {
                            "step": int(step),
                            "mu": float(hopf["mu_est"]),
                            "success": True,
                            "newton_iter": "",
                            "residual_norm": "",
                            "corr_iter": "",
                            "corr_residual": "",
                            "eig_max_real": float(hopf["eig_max_real_est"]),
                            "stable": False,
                            "jac_smin": float(hopf["jac_smin_est"]),
                            "event": "HOPF_LOC",
                        }
                        for i, v in enumerate(y_est):
                            loc_row[f"y_{i}"] = float(v)
                        rows.append(loc_row)

            # FOLD: sign change in dmu/ds with near-singular Jacobian
            q0 = float(prev["dmu_ds"])
            q1 = float(cur["dmu_ds"])
            fold_cross = (np.sign(q0) != np.sign(q1))
            near_sing = (min(float(prev["jac_smin"]), float(cur["jac_smin"])) < max(cfg.svd_sing_tol, 1e-300) * 1e3)
            if fold_cross and near_sing:
                fold = _localize_fold(spec, cfg, prev, cur)
                events.append(fold)
                if fold.get("success") and fold.get("y_est") is not None:
                    y_est = np.asarray(fold["y_est"], dtype=float)
                    loc_row = {
                        "step": int(step),
                        "mu": float(fold["mu_est"]),
                        "success": True,
                        "newton_iter": "",
                        "residual_norm": "",
                        "corr_iter": "",
                        "corr_residual": "",
                        "eig_max_real": float(fold.get("eig_max_real_est", float("nan"))),
                        "stable": False,
                        "jac_smin": float(fold.get("jac_smin_est", float("nan"))),
                        "event": "FOLD_LOC",
                    }
                    for i, v in enumerate(y_est):
                        loc_row[f"y_{i}"] = float(v)
                    rows.append(loc_row)

        # --- adaptive step with retry loop ---
        accepted_next = False
        retry = 0
        while (not accepted_next) and (retry <= cfg.step_retry_max):
            # predictor
            y_pred = np.asarray(cur["y"], dtype=float) + ds * np.asarray(cur["dy_ds"], dtype=float)
            mu_pred = float(cur["mu"]) + ds * float(cur["dmu_ds"])

            # corrector
            corr = corrector_pseudo_arclength(
                spec=spec,
                cfg=cfg,
                y_pred=y_pred,
                mu_pred=mu_pred,
                dy_ds=np.asarray(cur["dy_ds"], dtype=float),
                dmu_ds=float(cur["dmu_ds"]),
                y_init=y_pred,
                mu_init=mu_pred,
            )

            if corr.success and corr.n_iter <= cfg.iter_bad:
                # accept step
                y_next = np.asarray(corr.y, dtype=float)
                mu_next = float(corr.mu)

                # update tangent at new accepted point
                nxt = _equilibrium_stab_tangent(spec, cfg, mu_next, y_next)
                if not nxt["success"]:
                    # treat as failed, shrink ds and retry
                    ds = max(cfg.ds_min, ds * cfg.ds_shrink)
                    retry += 1
                    continue

                # annotate the last "accepted point" row with corrector diagnostics
                rows[-1]["corr_iter"] = int(corr.n_iter)
                rows[-1]["corr_residual"] = float(corr.residual_norm)

                prev = cur
                y = y_next
                mu = mu_next
                dy_ds = nxt["dy_ds"]
                dmu_ds = float(nxt["dmu_ds"])

                # ds adaptation based on performance
                if corr.n_iter <= cfg.iter_good:
                    good_streak += 1
                else:
                    good_streak = 0

                if good_streak >= cfg.good_streak_for_grow:
                    ds = min(cfg.ds_max, ds * cfg.ds_grow)
                    good_streak = 0

                accepted_next = True
            else:
                # shrink ds and retry
                ds = max(cfg.ds_min, ds * cfg.ds_shrink)
                retry += 1

        if not accepted_next:
            # record failure row and stop
            fail_row = {
                "step": int(step + 1),
                "mu": float(mu),
                "success": False,
                "newton_iter": "",
                "residual_norm": "",
                "corr_iter": "",
                "corr_residual": "",
                "eig_max_real": float("nan"),
                "stable": False,
                "jac_smin": float("nan"),
                "event": f"STEP_FAIL: retry_exhausted ds={ds}",
            }
            for i in range(len(y)):
                fail_row[f"y_{i}"] = float("nan")
            rows.append(fail_row)
            break

        step += 1

    report = {
        "kind": "equilibrium_continuation",
        "param": spec.param_name,
        "mu_start": float(cfg.mu_start),
        "mu_stop": float(cfg.mu_stop),
        "runtime_s": float(time.time() - t0),
        "n_points": int(len(rows)),
        "final_ds": float(ds),
        "events": [
            {
                **{k: v for k, v in ev.items() if k != "y_est"},
                "has_y_est": bool(ev.get("y_est") is not None),
            }
            for ev in events
        ],
    }
    return rows, report
