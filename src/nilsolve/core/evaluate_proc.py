import time
import multiprocessing as mp
import numpy as np
from scipy.integrate import solve_ivp

from nilsolve.models import CandidateResult, ProblemSpec
from nilsolve.core.quality import assess_solution_quality


def _worker(problem: ProblemSpec, method: str, rtol: float, atol: float, q: mp.Queue):
    t0 = time.time()
    try:
        sol = solve_ivp(
            fun=lambda t, y: problem.rhs(t, y, problem.params),
            t_span=problem.t_span,
            y0=problem.y0,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        runtime = time.time() - t0

        q.put((
            "ok",
            sol.t,
            sol.y,
            runtime,
            getattr(sol, "nfev", 0),
            getattr(sol, "njev", 0),
            getattr(sol, "nlu", 0),
            str(sol.message),
            bool(sol.success),
            int(sol.status),
        ))
    except Exception as e:
        runtime = time.time() - t0
        q.put(("err", runtime, str(e)))


def solve_one_candidate(
    problem: ProblemSpec,
    method: str,
    policy_name: str,
    rtol: float,
    atol: float,
    max_wall_s: float,
) -> CandidateResult:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker, args=(problem, method, rtol, atol, q))
    p.daemon = True

    parent_t0 = time.time()
    p.start()
    p.join(timeout=max_wall_s)

    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        p.join(timeout=0.2)

        if p.is_alive():
            try:
                p.kill()
            except Exception:
                pass
            p.join(timeout=0.2)

        runtime = time.time() - parent_t0
        return CandidateResult(
            method=method,
            policy_name=policy_name,
            rtol=rtol,
            atol=atol,
            max_step=None,
            success=False,
            status=-2,
            message=f"Timeout > {max_wall_s}s",
            runtime_s=float(runtime),
            nfev=0,
            njev=0,
            nlu=0,
            finite_ok=False,
            nonnegative_ok=False,
            quality_status="FAIL",
            quality_metrics=None,
            quality_notes=["Hard timeout"],
            invariant_drift=None,
            dt_halving_relerr=None,
            score=float("inf"),
        )

    runtime_parent = time.time() - parent_t0

    if q.empty():
        return CandidateResult(
            method=method,
            policy_name=policy_name,
            rtol=rtol,
            atol=atol,
            max_step=None,
            success=False,
            status=-3,
            message="No result returned",
            runtime_s=float(runtime_parent),
            nfev=0,
            njev=0,
            nlu=0,
            finite_ok=False,
            nonnegative_ok=False,
            quality_status="FAIL",
            quality_metrics=None,
            quality_notes=["No payload from worker"],
            invariant_drift=None,
            dt_halving_relerr=None,
            score=float("inf"),
        )

    payload = q.get()
    tag = payload[0]

    if tag == "err":
        _, runtime_child, msg = payload
        return CandidateResult(
            method=method,
            policy_name=policy_name,
            rtol=rtol,
            atol=atol,
            max_step=None,
            success=False,
            status=-99,
            message=str(msg),
            runtime_s=float(runtime_parent),
            nfev=0,
            njev=0,
            nlu=0,
            finite_ok=False,
            nonnegative_ok=False,
            quality_status="FAIL",
            quality_metrics=None,
            quality_notes=[str(msg)],
            invariant_drift=None,
            dt_halving_relerr=None,
            score=float("inf"),
        )

    # tag == "ok"
    _, t, y, runtime_child, nfev, njev, nlu, msg, success, status = payload

    finite_ok = bool(np.isfinite(y).all())

    # Run formal quality gate
    quality = assess_solution_quality(
        t=t,
        y=y.T,  # solve_ivp returns shape (d,n); quality expects (n,d)
        f_rhs=lambda tt, yy: problem.rhs(tt, yy, problem.params),
        atol=atol,
        rtol=rtol,
        state_names=problem.state_names,
        nonnegative=problem.nonnegative_indices if problem.require_nonnegative else None,
        invariants=problem.invariants,
    )

    # Legacy flags (kept for compatibility)
    nonnegative_ok = quality.status != "FAIL"

    # Scoring: penalize WARN, exclude FAIL
    if quality.status == "FAIL":
        score = float("inf")
    else:
        penalty = 1.5 if quality.status == "WARN" else 1.0
        score = penalty * (float(runtime_parent) + float(nfev) * 1e-3)

    return CandidateResult(
        method=method,
        policy_name=policy_name,
        rtol=rtol,
        atol=atol,
        max_step=None,
        success=bool(success) and quality.status != "FAIL",
        status=int(status),
        message=str(msg),
        runtime_s=float(runtime_parent),
        nfev=int(nfev),
        njev=int(njev),
        nlu=int(nlu),
        finite_ok=finite_ok,
        nonnegative_ok=nonnegative_ok,
        quality_status=quality.status,
        quality_metrics=quality.metrics,
        quality_notes=quality.notes,
        invariant_drift=None,
        dt_halving_relerr=None,
        score=score,
    )
