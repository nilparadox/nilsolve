from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json


@dataclass(frozen=True)
class SolverDecision:
    method: str
    rtol: float
    atol: float


def load_decision(report_json_path: str | Path) -> SolverDecision:
    p = Path(report_json_path)
    data: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    sel = data.get("selected") or {}
    return SolverDecision(
        method=str(sel["method"]),
        rtol=float(sel["rtol"]),
        atol=float(sel["atol"]),
    )
