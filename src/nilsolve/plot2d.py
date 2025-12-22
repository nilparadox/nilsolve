from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def plot_scan2d_heatmap(
    csv_path: str,
    out_png: str,
    x_col: str = "mu2",
    y_col: str = "mu1",
    z_col: str = "amp",
    title: str = "scan2d heatmap",
) -> Tuple[int, int]:
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    # collect unique grid values
    xs = sorted({float(r[x_col]) for r in rows if r.get("success", "").lower() in ("true", "1", "yes", "y") and r.get(x_col) is not None})
    ys = sorted({float(r[y_col]) for r in rows if r.get("success", "").lower() in ("true", "1", "yes", "y") and r.get(y_col) is not None})

    if not xs or not ys:
        raise ValueError("No successful rows to plot.")

    x_index = {v: i for i, v in enumerate(xs)}
    y_index = {v: i for i, v in enumerate(ys)}

    Z = np.full((len(ys), len(xs)), np.nan, dtype=float)

    used = 0
    for r in rows:
        if r.get("success", "").lower() not in ("true", "1", "yes", "y"):
            continue
        try:
            xv = float(r[x_col])
            yv = float(r[y_col])
            zv = float(r[z_col])
        except Exception:
            continue
        Z[y_index[yv], x_index[xv]] = zv
        used += 1

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    plt.figure()
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[min(xs), max(xs), min(ys), max(ys)],
    )
    plt.colorbar(im, label=z_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

    return (len(rows), used)
