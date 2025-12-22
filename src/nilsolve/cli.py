from __future__ import annotations

import sys
import multiprocessing as mp
import importlib.util
import time
from pathlib import Path

import typer
from scipy.integrate import solve_ivp

from nilsolve.api import rank_methods
from nilsolve.models import ProblemSpec
from nilsolve.core.io import write_solution_csv, write_report_json
from nilsolve.core.plotting import plot_solution
from nilsolve.core.evaluate_proc import solve_one_candidate

from nilsolve.core.cont.spec import EquilibriumSpec, ContinuationConfig
from nilsolve.core.cont.run import continue_equilibria
from nilsolve.core.cont.io import write_branch_csv, write_report_json as write_cont_report_json
from nilsolve.core.cont.plotting import plot_branch_png


if sys.platform == "darwin":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass


app = typer.Typer(add_completion=False)
cont_app = typer.Typer(add_completion=False)
app.add_typer(cont_app, name="cont")


def parse_set(items: list[str] | None) -> dict[str, float]:
    params: dict[str, float] = {}
    if not items:
        return params
    for it in items:
        if "=" not in it:
            raise typer.BadParameter("Use --set key=value")
        k, v = it.split("=", 1)
        params[k] = float(v)
    return params


def load_system(path: str):
    path = Path(path).resolve()
    name = f"nilsolve_user_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load system file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


@app.command()
def rank(
    system: str = typer.Option(...),
    set: list[str] = typer.Option(None),
    budget: str = typer.Option("standard"),
    max_wall_s: float = typer.Option(20.0),
    pilot_cap: float = typer.Option(5.0),
):
    mod, path = load_system(system)
    typer.echo(f"Loaded system module: {mod.__name__}")
    typer.echo(f"Loaded system path:   {path}")

    if not hasattr(mod, "build_problem"):
        raise typer.BadParameter("System must define build_problem()")

    prob: ProblemSpec = mod.build_problem()
    overrides = parse_set(set)
    if overrides:
        prob.params.update(overrides)
        typer.echo(f"Applied parameter overrides: {overrides}")

    leaderboard, selected = rank_methods(
        problem=prob,
        budget=budget,
        max_wall_s=max_wall_s,
        pilot_cap=pilot_cap,
        verbose=True,
    )

    typer.echo("nilsolve leaderboard (best → worst)")
    for i, r in enumerate(leaderboard, 1):
        typer.echo(
            f"{i:2d}. {r.method:<6} {r.policy_name:<8} "
            f"ok={str(r.success)[0]} status={r.status:<3} "
            f"t={r.runtime_s:>6.3f}s nfev={r.nfev:<8} score={r.score:.4g} msg={r.message}"
        )

    if selected:
        typer.echo(f"Selected: {selected.method} + {selected.policy_name}")


@app.command()
def solve(
    system: str = typer.Option(...),
    set: list[str] = typer.Option(None),
    budget: str = typer.Option("standard"),
    max_wall_s: float = typer.Option(20.0),
    pilot_cap: float = typer.Option(5.0),
    out_prefix: str = typer.Option("outputs/nilsolve"),
    no_plot: bool = typer.Option(False),
):
    mod, path = load_system(system)
    typer.echo(f"Loaded system module: {mod.__name__}")
    typer.echo(f"Loaded system path:   {path}")

    if not hasattr(mod, "build_problem"):
        raise typer.BadParameter("System must define build_problem()")

    prob: ProblemSpec = mod.build_problem()
    overrides = parse_set(set)
    if overrides:
        prob.params.update(overrides)
        typer.echo(f"Applied parameter overrides: {overrides}")

    leaderboard, selected = rank_methods(
        problem=prob,
        budget=budget,
        max_wall_s=max_wall_s,
        pilot_cap=pilot_cap,
        verbose=False,
    )
    if selected is None:
        raise RuntimeError("No solver candidates produced a result.")

    typer.echo(f"Selected: {selected.method} + {selected.policy_name}")

    t0 = time.time()
    sol = solve_ivp(
        fun=lambda t, y: prob.rhs(t, y, prob.params),
        t_span=prob.t_span,
        y0=prob.y0,
        method=selected.method,
        rtol=selected.rtol,
        atol=selected.atol,
    )
    runtime = time.time() - t0

    prefix = Path(out_prefix)
    csv_path = Path(str(prefix) + "_solution.csv")
    png_path = Path(str(prefix) + "_solution.png")
    rep_path = Path(str(prefix) + "_report.json")

    names = getattr(mod, "state_names", None)
    write_solution_csv(csv_path, sol.t, sol.y, names=names)

    payload = {
        "system_path": str(path),
        "params": dict(prob.params),
        "selected": {
            "method": selected.method,
            "policy_name": selected.policy_name,
            "rtol": selected.rtol,
            "atol": selected.atol,
        },
        "solve": {
            "success": bool(sol.success),
            "status": int(sol.status),
            "message": str(sol.message),
            "runtime_s": float(runtime),
            "nfev": int(getattr(sol, "nfev", 0)),
            "njev": int(getattr(sol, "njev", 0)),
            "nlu": int(getattr(sol, "nlu", 0)),
        },
        "leaderboard": [r.__dict__ for r in leaderboard],
    }
    write_report_json(rep_path, payload)

    if not no_plot:
        try:
            plot_solution(png_path, sol.t, sol.y, names=names)
        except Exception as e:
            typer.echo(f"Plot skipped: {e}")


@app.command()
def scan(
    system: str = typer.Option(...),
    mu: str = typer.Option(...),
    start: float = typer.Option(...),
    stop: float = typer.Option(...),
    n: int = typer.Option(41),
    out_csv: str = typer.Option("outputs/scan.csv"),
    reuse_decision_from: str = typer.Option(None),
    no_rerank: bool = typer.Option(False),
    per_point_timeout_s: float = typer.Option(5.0),
):
    import numpy as _np

    mod, path = load_system(system)
    typer.echo(f"Loaded system module: {mod.__name__}")
    typer.echo(f"Loaded system path:   {path}")

    if not hasattr(mod, "build_problem"):
        raise typer.BadParameter("System must define build_problem()")

    prob: ProblemSpec = mod.build_problem()

    if reuse_decision_from:
        from nilsolve.core.decision import load_decision
        d = load_decision(reuse_decision_from)
        method, rtol, atol = d.method, d.rtol, d.atol
    else:
        if no_rerank:
            raise typer.BadParameter("--no-rerank requires --reuse-decision-from")
        leaderboard, selected = rank_methods(problem=prob, verbose=False)
        if selected is None:
            raise RuntimeError("No solver candidates produced a result.")
        method, rtol, atol = selected.method, selected.rtol, selected.atol

    mus = _np.linspace(start, stop, n).astype(float)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = "mu_name,mu_value,success,status,message,runtime_s,nfev,njev,nlu,finite_ok,nonnegative_ok\n"
    rows = [header]

    for v in mus:
        prob.params[mu] = float(v)

        res = solve_one_candidate(
            problem=prob,
            method=method,
            policy_name="scan",
            rtol=rtol,
            atol=atol,
            max_wall_s=per_point_timeout_s,
        )

        msg = str(res.message).replace('"', '""')

        line = (
            f"{mu},{float(v):.18g},"
            f"{int(res.success)},{int(res.status)},"
            f"\"{msg}\","
            f"{res.runtime_s:.18g},{res.nfev},{res.njev},{res.nlu},"
            f"{int(res.finite_ok)},{int(res.nonnegative_ok)}\n"
        )
        rows.append(line)

    out_path.write_text("".join(rows), encoding="utf-8")
    typer.echo(f"Wrote scan CSV: {out_path}")


@cont_app.command("eq")
def cont_eq(
    system: str = typer.Option(..., help="System python file defining build_equilibrium_problem()"),
    param: str = typer.Option(..., help="Continuation parameter name (mu)"),
    mu_start: float = typer.Option(...),
    mu_stop: float = typer.Option(...),
    ds: float = typer.Option(0.05),
    max_steps: int = typer.Option(400),
    out_prefix: str = typer.Option("outputs/cont"),
    y_index_plot: int = typer.Option(0, help="Which state component to plot vs mu"),
    no_plot: bool = typer.Option(False),
):
    mod, path = load_system(system)
    typer.echo(f"Loaded system module: {mod.__name__}")
    typer.echo(f"Loaded system path:   {path}")

    if not hasattr(mod, "build_equilibrium_problem"):
        raise typer.BadParameter("System must define build_equilibrium_problem() -> EquilibriumSpec")

    spec: EquilibriumSpec = mod.build_equilibrium_problem()
    if spec.param_name != param:
        # Allow override from CLI
        spec.param_name = param

    cfg = ContinuationConfig(
        mu_start=float(mu_start),
        mu_stop=float(mu_stop),
        ds=float(ds),
        max_steps=int(max_steps),
    )

    rows, report = continue_equilibria(spec, cfg)

    prefix = Path(out_prefix)
    csv_path = Path(str(prefix) + "_branch.csv")
    rep_path = Path(str(prefix) + "_report.json")
    png_path = Path(str(prefix) + "_branch.png")

    write_branch_csv(csv_path, rows, state_names=spec.state_names)
    write_cont_report_json(rep_path, report)

    if (not no_plot):
        try:
            plot_branch_png(png_path, rows, state_names=spec.state_names, y_index=int(y_index_plot))
        except Exception as e:
            typer.echo(f"Plot skipped: {e}")

    typer.echo(f"Wrote branch CSV:   {csv_path}")
    typer.echo(f"Wrote report JSON:  {rep_path}")
    if not no_plot:
        typer.echo(f"Wrote branch PNG:   {png_path}")


if __name__ == "__main__":
    app()
