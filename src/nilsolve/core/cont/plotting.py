from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


def plot_branch_png(
    path: Path,
    rows: List[Dict[str, Any]],
    state_names: Optional[Sequence[str]] = None,
    y_index: int = 0,
) -> None:
    """
    Simple bifurcation diagram: mu vs y[y_index].
    Uses matplotlib if available; otherwise raises ImportError.
    """
    import matplotlib.pyplot as plt

    if not rows:
        return

    mu = np.array([float(r["mu"]) for r in rows], dtype=float)
    yk = np.array([float(r.get(f"y_{y_index}", np.nan)) for r in rows], dtype=float)
    stable = np.array([bool(r.get("stable", False)) for r in rows], dtype=bool)

    plt.figure()
    plt.plot(mu[stable], yk[stable], marker="o", linestyle="None")
    plt.plot(mu[~stable], yk[~stable], marker="x", linestyle="None")
    plt.xlabel("mu")
    ylabel = state_names[y_index] if (state_names and y_index < len(state_names)) else f"y[{y_index}]"
    plt.ylabel(ylabel)
    plt.title("Equilibrium continuation branch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
