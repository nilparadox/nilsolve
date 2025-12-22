from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def plot_scan1d(
    csv_path: str,
    out_png: str,
    x_col: str = "mu",
    y_col: str = "amp",
    title: str = "scan1d",
) -> Tuple[int, int]:
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    pairs: List[Tuple[float, float]] = []

    for row in rows:
        if row.get("success", "").lower() not in ("true", "1", "yes", "y"):
            continue
        xv = row.get(x_col, None)
        yv = row.get(y_col, None)
        if xv is None or yv is None:
            continue
        try:
            x = float(xv)
            y = float(yv)
        except Exception:
            # e.g., period_est is blank/None => skip
            continue
        # skip NaNs cleanly
        if x != x or y != y:
            continue
        pairs.append((x, y))

    if not pairs:
        raise ValueError(f"No plottable numeric (x,y) pairs for {x_col} vs {y_col} in {csv_path}")

    # Sort by x to make curves readable
    pairs.sort(key=lambda t: t[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return (len(rows), len(pairs))
