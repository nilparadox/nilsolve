from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import numpy as np


def plot_solution(path: Path, t: np.ndarray, y: np.ndarray, names: Optional[Sequence[str]] = None) -> None:
    """
    If dim=2 -> phase portrait (y0 vs y1).
    Else -> time series (t vs yi).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise RuntimeError("matplotlib is not installed. Install it or use --no-plot.")

    if y.ndim != 2:
        raise ValueError("y must be 2D array shaped (dim, n_points)")

    dim, n = y.shape
    if names is None:
        names = [f"y{i}" for i in range(dim)]

    path.parent.mkdir(parents=True, exist_ok=True)

    if dim == 2:
        plt.figure()
        plt.plot(y[0, :], y[1, :])
        plt.xlabel(names[0])
        plt.ylabel(names[1])
        plt.title("Phase portrait")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
    else:
        plt.figure()
        for i in range(dim):
            plt.plot(t, y[i, :], label=names[i])
        plt.xlabel("t")
        plt.ylabel("state")
        plt.title("Trajectory")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
