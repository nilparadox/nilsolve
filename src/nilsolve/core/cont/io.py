from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json


def write_branch_csv(path: Path, rows: List[Dict[str, Any]], state_names: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Determine y dimension from first row
    y_keys = [k for k in rows[0].keys() if k.startswith("y_")]
    y_keys = sorted(y_keys, key=lambda s: int(s.split("_")[1]))

    base_cols = [
        "step", "mu", "success",
        "newton_iter", "residual_norm",
        "corr_iter", "corr_residual",
        "eig_max_real", "stable", "jac_smin", "event"
    ]
    if state_names:
        y_cols = [f"{nm}" for nm in state_names]
    else:
        y_cols = [f"y{i}" for i in range(len(y_keys))]

    header = base_cols + y_cols
    out = [",".join(header) + "\n"]

    def _fmt(x: Any) -> str:
        if x is None:
            return ""
        if x == "":
            return ""
        return str(x)

    for r in rows:
        y_vals = [r.get(k, float("nan")) for k in y_keys]

        event_str = str(r.get("event", ""))
        event_str = event_str.replace('"', '""')  # CSV quoting

        line_vals = [
            _fmt(r.get("step", "")),
            _fmt(r.get("mu", "")),
            int(bool(r.get("success", False))),
            _fmt(r.get("newton_iter", "")),
            _fmt(r.get("residual_norm", "")),
            _fmt(r.get("corr_iter", "")),
            _fmt(r.get("corr_residual", "")),
            _fmt(r.get("eig_max_real", "")),
            int(bool(r.get("stable", False))),
            _fmt(r.get("jac_smin", "")),
            f"\"{event_str}\"",
        ]
        line_vals += [f"{float(v):.18g}" for v in y_vals]
        out.append(",".join(map(str, line_vals)) + "\n")

    path.write_text("".join(out), encoding="utf-8")


def write_report_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
