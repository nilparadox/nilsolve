from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence, Dict, Any
import json

import numpy as np


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_report_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_solution_csv(path: Path, t: np.ndarray, y: np.ndarray, names: Optional[Sequence[str]] = None) -> None:
    """
    Writes a wide CSV:
        t, y0, y1, ... (or named columns if provided)
    y is (n, m) where n=dim, m=len(t)
    """
    ensure_parent(path)
    if y.ndim != 2:
        raise ValueError("y must be 2D array shaped (dim, n_points)")

    dim, n = y.shape
    if t.shape[0] != n:
        raise ValueError("t and y length mismatch")

    if names is None:
        names = [f"y{i}" for i in range(dim)]
    if len(names) != dim:
        raise ValueError("names length must match y dimension")

    header = "t," + ",".join(names)
    rows = np.column_stack([t, y.T])
    lines = [header] + [",".join(f"{v:.18g}" for v in row) for row in rows]
    path.write_text("\n".join(lines), encoding="utf-8")
